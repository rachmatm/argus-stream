package stream

import (
	"bufio"
	"context"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Package stream orchestrates the ffmpeg + Python subprocess pair that turns
// an HLS URL into raw RGB frames plus detection metadata. The orchestration
// is purely goroutine + channel based: one goroutine reads ffmpeg stdout
// into fixed-size frames, two more drain the subprocess stderr pipes, and a
// fourth fans a copy of every Nth frame into Python's stdin. Cancellation
// propagates through the context passed to Start: cancelling it kills both
// child processes via exec.CommandContext.

// DetectEveryN throttles how often ffmpeg frames are forwarded to Python for
// YOLO inference. At 30 fps and N=5 that's ~6 Hz of inference, which is
// roughly what YOLOv8 Nano can sustain on a modern CPU. All 30 fps frames
// still reach the frontend via OnFrameReady -- only the detection step is
// throttled.
const DetectEveryN = 5

// Pipeline holds the running subprocesses and the user-supplied callbacks
// fired by the goroutines started in Start().
type Pipeline struct {
	HlsURL string // HLS playlist URL passed to ffmpeg -i

	OnMetaReady  func(jsonStr string) // called once per META:{json} line from Python stderr
	OnFrameReady func(frame []byte)   // called once per decoded ffmpeg frame (~30 Hz)

	ffmpegCmd *exec.Cmd // child process for ffmpeg; bound to ctx via CommandContext
	pythonCmd *exec.Cmd // child process for inference/detection_engine.py; bound to ctx via CommandContext
}

func NewPipeline(url string, metaCb func(string), frameCb func([]byte)) *Pipeline {
	return &Pipeline{HlsURL: url, OnMetaReady: metaCb, OnFrameReady: frameCb}
}

// resolvePythonBin prefers the project's venv interpreter (so this doesn't
// silently depend on whether the venv happens to be active in whatever shell
// launched `wails dev`), falling back to whatever "python3" resolves to on
// PATH if no venv is found. Set ARGUS_PYTHON_BIN to override explicitly.
func resolvePythonBin() string {
	if override := os.Getenv("ARGUS_PYTHON_BIN"); override != "" {
		return override
	}

	candidates := []string{
		"venv/bin/python3",          // Linux/macOS venv
		"venv\\Scripts\\python.exe", // Windows venv
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			abs, err := filepath.Abs(c)
			if err == nil {
				return abs
			}
			return c
		}
	}

	log.Println("[DEBUG] no venv found at ./venv, falling back to PATH python3 - this will fail if ultralytics isn't installed system-wide")
	return "python3"
}

// Start spins up the ffmpeg + Python subprocess pair and wires them together
// with four goroutines. The function returns once the processes are spawned;
// all subsequent work happens in the goroutines started below.
//
// Lifecycle: cancelling ctx (e.g. Wails shutdown) kills both child processes
// because exec.CommandContext installs a goroutine that sends SIGKILL the
// moment ctx.Done() fires.
func (p *Pipeline) Start(ctx context.Context) error {
	pythonBin := resolvePythonBin()
	log.Println("[DEBUG] using python interpreter:", pythonBin)

	// Python is spawned FIRST so its stdin pipe exists before ffmpeg starts
	// producing frames. exec.CommandContext binds the child to ctx: when
	// ctx is cancelled, Go sends SIGKILL to the child automatically.
	p.pythonCmd = exec.CommandContext(ctx, pythonBin, "inference/detection_engine.py")

	// Two separate pipes -- Stdin (we write to) and Stderr (we read from).
	// A single pipe cannot be both read and written by the parent.
	pyIn, _ := p.pythonCmd.StdinPipe()
	pyErr, _ := p.pythonCmd.StderrPipe()

	// Start() returns immediately; the child runs concurrently with this
	// function. We check the error and bail early if it failed to launch.
	if err := p.pythonCmd.Start(); err != nil {
		log.Println("[DEBUG] python3 failed to start:", err)
		return err
	}
	log.Println("[DEBUG] python3 process started, pid:", p.pythonCmd.Process.Pid)

	//   Python side - inference/detection_engine.py

	//   WIDTH, HEIGHT, CHANNELS = 640, 480, 3
	//   FRAME_SIZE = WIDTH * HEIGHT * CHANNELS      # = 921,600 bytes

	//   raw_bytes = sys.stdin.buffer.read(FRAME_SIZE)   # blocks until 921,600 bytes arrive
	//   if len(raw_bytes) != FRAME_SIZE:
	//       break                                         # EOF - exit
	//   rgb_frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))

	//   The contract (each link matters)

	//   +------------------------+-----------------------+--------------------------------------------------+
	//   |         Link           |        Value          |                       Why                        |
	//   +------------------------+-----------------------+--------------------------------------------------+
	//   | -pix_fmt rgb24         | 3 bytes/pixel         | Must match CHANNELS = 3 and dtype uint8          |
	//   +------------------------+-----------------------+--------------------------------------------------+
	//   | -s 640x480             | 640 x 480 pixels      | Must match the (HEIGHT, WIDTH, CHANNELS) reshape |
	//   |                        |                       |  order                                           |
	//   +------------------------+-----------------------+--------------------------------------------------+
	//   | 640 x 480 x 3          | 921,600 bytes/frame   | Must equal FRAME_SIZE on both sides              |
	//   +------------------------+-----------------------+--------------------------------------------------+
	//   | Raw pipe:1 (no         | byte stream, no       | No length prefix -> positional synchronization   |
	//   | container)             | headers               |  only                                            |
	//   +------------------------+-----------------------+--------------------------------------------------+
	// Same CommandContext pattern for ffmpeg. The arg list is the entire
	// ffmpeg pipeline: HLS demux -> decode -> rescale -> raw RGB24 -> stdout.
	// The matching Python contract lives in inference/detection_engine.py.
	p.ffmpegCmd = exec.CommandContext(ctx, "ffmpeg", "-i", p.HlsURL, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "30", "pipe:1")

	// Read ffmpeg's stdout (raw frames) and stderr (ffmpeg log lines)
	// through separate pipes. StdoutPipe must be read from exactly one
	// goroutine -- the ffmpeg reader loop below -- otherwise frames
	// interleave and the protocol desyncs.
	ffmpegOut, _ := p.ffmpegCmd.StdoutPipe()
	ffmpegErr, _ := p.ffmpegCmd.StderrPipe()
	if err := p.ffmpegCmd.Start(); err != nil {
		log.Println("[DEBUG] ffmpeg failed to start:", err)
		return err
	}
	log.Println("[DEBUG] ffmpeg process started, pid:", p.ffmpegCmd.Process.Pid)

	// GOROUTINE #1: ffmpeg stderr drainer.
	// ffmpeg writes its own log lines (codec info, frame counters, errors)
	// to stderr. If nobody reads it, the pipe buffer fills up and ffmpeg
	// blocks on write. Scanner.Buffer(1 MB) raises the default 64 KB line
	// limit so verbose codec messages don't get truncated.
	go func() {
		scanner := bufio.NewScanner(ffmpegErr)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			log.Println("[ffmpeg]", scanner.Text())
		}
	}()

	// GOROUTINE #2: Python stderr drainer + metadata extractor.
	// detection_engine.py emits detection results as META:{json} lines on
	// stderr (stderr is chosen because stdout would interleave with any
	// future passthrough frames). Anything else is treated as a debug log
	// line. This goroutine is the only thing that parses the metadata
	// protocol between Python and Go.
	go func() {
		scanner := bufio.NewScanner(pyErr)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "META:") {
				p.OnMetaReady(strings.TrimPrefix(line, "META:"))
			} else {
				log.Println("[python]", line)
			}
		}
		log.Println("[DEBUG] python stderr scanner exited (process likely closed stderr/exited)")
	}()

	// detectCh is the bridge between the ffmpeg reader loop (producer) and
	// Python's stdin (consumer). Capacity 2 is intentional: combined with
	// the non-blocking send below, it gives drop-on-overflow backpressure
	// rather than blocking the ffmpeg reader. If Python is slow, we drop
	// detection slots rather than letting frames pile up in the channel.
	detectCh := make(chan []byte, 2)

	// GOROUTINE #3: Python stdin pumper.
	// Pulls frames off detectCh and writes them to Python's stdin pipe.
	// The loop terminates when detectCh is closed by GOROUTINE #4 below --
	// a closed channel drains naturally and `range` ends.
	go func() {
		for frame := range detectCh {
			if _, err := pyIn.Write(frame); err != nil {
				log.Println("[DEBUG] failed writing frame to python stdin:", err)
				return
			}
		}
	}()

	// GOROUTINE #4: ffmpeg reader loop (the heart of the pipeline).
	// This single goroutine owns ffmpegOut. It pulls fixed-size frames out
	// of the pipe and fans each one out to two sinks:
	//   1. Always:  p.OnFrameReady  -> the frontend (via WebSocket) at 30 Hz.
	//   2. Every Nth: detectCh      -> Python for inference (~6 Hz).
	go func() {
		frameSize := 640 * 480 * 3 // = 921,600 bytes; MUST match FRAME_SIZE in detection_engine.py
		buf := make([]byte, frameSize)
		framesRead := 0

		for {
			// io.ReadFull reads EXACTLY frameSize bytes (or returns err).
			// Unlike io.Read, it does not return short reads -- this is
			// what makes the fixed-size framing protocol work. If we ever
			// get fewer bytes, ffmpeg has ended and we exit.
			_, err := io.ReadFull(ffmpegOut, buf)
			if err != nil {
				log.Println("[DEBUG] ffmpeg reader loop exiting after", framesRead, "frames, err:", err)
				close(detectCh) // signals GOROUTINE #3 to exit cleanly
				return
			}
			framesRead++
			if framesRead == 1 {
				log.Println("[DEBUG] first raw frame read from ffmpeg stdout")
			}

			// DEFENSIVE COPY: `buf` is reused on every iteration, so we
			// MUST copy before handing the bytes to any other goroutine
			// (OnFrameReady -> WebSocket writer, or detectCh consumer ->
			// pyIn.Write). Sharing buf would mean the next ffmpeg read
			// overwrites the previous frame mid-flight, corrupting both
			// the WebSocket frame and the Python input.
			frameCopy := make([]byte, frameSize)
			copy(frameCopy, buf)
			p.OnFrameReady(frameCopy)

			// Throttled detection fan-out. The `select { default: }` is
			// non-blocking: if detectCh is full (Python is slower than the
			// ffmpeg reader), we DROP this detection slot rather than
			// stalling the 30 Hz frame pipeline. Detection quality degrades
			// under load, but the video feed stays smooth.
			if framesRead%DetectEveryN == 0 {
				select {
				case detectCh <- frameCopy:
				default:
				}
			}
		}
	}()

	return nil
}

// Close kills both child processes. Process.Kill() sends SIGKILL -- neither
// ffmpeg nor detection_engine.py has signal handlers for graceful shutdown,
// so there is nothing to wait for. The goroutines started in Start() exit
// on their own once their pipes return EOF (which SIGKILL triggers).
func (p *Pipeline) Close() {
	if p.ffmpegCmd != nil && p.ffmpegCmd.Process != nil {
		p.ffmpegCmd.Process.Kill()
	}
	if p.pythonCmd != nil && p.pythonCmd.Process != nil {
		p.pythonCmd.Process.Kill()
	}
}
