# Argus

> Real-time object detection on live video streams.

A desktop application that ingests a live HLS video stream, runs YOLOv8 object detection, and renders tracked objects with labeled bounding boxes in a native desktop window.

This project is primarily a **knowledge exercise** in building a streaming, multi-process pipeline. It explores:

- **Goroutines and channels** in Go for concurrent subprocess management, frame piping, and backpressure handling.
- **WebSocket** (Gorilla) as a low-overhead binary transport for streaming raw RGB frames and metadata to the frontend in real time.
- **Inter-process communication** between Go and Python via stdin/stdout pipes, using a fixed-size framing protocol (921,600 bytes per frame).
- **Process orchestration**: `ffmpeg` (HLS decode) -> `Python` (YOLOv8 + IoU tracker) -> `Go` (Wails shell) -> browser frontend over WebSocket.

**Why this stack:**

- **Go** powers the application shell, the subprocess plumbing, and the WebSocket server. Goroutines and a small memory footprint keep CPU and RAM usage low even while a 30 fps frame stream and periodic YOLO inference are running concurrently.
- **Python** handles object detection and tracking. It was chosen for the mature ML ecosystem (`ultralytics`, OpenCV, NumPy); porting YOLO into Go would dwarf the rest of the project.
- **Wails v2** wraps the whole thing as a native desktop app - Go on the backend, HTML/TypeScript on the frontend, no browser in the tab bar. It is included in this repo purely as the demo shell for the streaming pipeline.

Use it as a reference for goroutine patterns, WebSocket frame protocols, and Go/Python IPC.


## Brief

The system tracks 8 COCO classes: **person, bicycle, car, motorcycle, bus, truck, dog, cat**. Each detected object receives a stable track ID that persists across frames, so a running unique-object counter is shown in the UI.


## Architecture

```
+-----------------+   raw RGB24   +-------------------+  processed RGB24  +----------+
|     ffmpeg      | ----stdin---> |  Python (YOLOv8)  | ---stdout pipe--> |  Go App  |
|  (HLS decode)   |               |  detection + IoU  |                   | (Wails)  |
+-----------------+               |  tracker         |  META: JSON on    +----+-----+
                                  +-------------------+  stderr                |
                                                                                |
                                                                          WebSocket (binary)
                                                                          ws://localhost:8083
                                                                                |
                                                                          +-----v------+
                                                                          |  Frontend  |
                                                                          |  (Canvas)  |
                                                                          +------------+
```

### Data flow

1. **Go** spawns `ffmpeg` to decode the HLS URL into raw RGB24 frames (640x480 @ 30fps).
2. **Go** pipes raw frames into a **Python** subprocess (`inference/detection_engine.py`) via stdin.
3. **Python** runs YOLOv8 Nano inference every 5th frame, maintains an IoU-based tracker for stable object IDs, and emits detection metadata as `META:{json}` lines on stderr.
4. **Python** writes processed (passthrough) RGB frames to stdout, which **Go** reads back.
5. **Go** combines frame bytes + JSON metadata into a binary packet: `[4-byte header length][JSON metadata][raw pixel bytes]`.
6. **Go** sends packets over a local WebSocket (`:8083/stream`) to the **frontend**.
7. **Frontend** decodes the frame onto a `<canvas>`, parses the metadata, and draws color-coded bounding boxes with class labels and track IDs.


## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Wails v2** (Go + web frontend) | Native desktop window with a lightweight Go backend; avoids Electron's overhead while keeping a modern web UI. |
| **ffmpeg subprocess** for stream decoding | Battle-tested HLS demuxer; handles network errors, codec negotiation, and reconnection without custom code. |
| **YOLOv8 Nano** (`yolov8n.pt`, ~6MB) | Smallest/fastest YOLOv8 variant; sufficient for real-time CPU inference. Auto-downloads on first run. |
| **Custom IoU tracker** | Lightweight, no external tracking library dependency. Matches detections across frames by bounding-box overlap (threshold 0.3), assigns stable IDs, expires stale tracks after 10 missed cycles. |
| **Python subprocess via stdin/stdout pipes** | Avoids CGo/FFI complexity; Python handles its own dependency tree independently. Go and Python communicate through well-defined binary and text protocols. |
| **Detection every 5th frame** | Reduces CPU load - YOLOv8 Nano on CPU cannot sustain 30fps; running inference at ~6fps while passing through all frames keeps the video feed smooth. |
| **WebSocket for Go->frontend** | Binary WebSocket frames carry both pixel data and metadata in a single message, avoiding HTTP overhead and enabling real-time rendering. |
| **Vanilla TypeScript frontend** | No framework overhead for a single-canvas UI; Vite handles bundling and dev server. |


## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | [Wails v2](https://wails.io/) |
| Backend | Go 1.22, Gorilla WebSocket |
| Stream decoding | ffmpeg |
| Object detection | [ultralytics](https://docs.ultralytics.com/) YOLOv8 Nano |
| Inference runtime | Python 3.10-3.11, OpenCV (headless), NumPy |
| Frontend | Vanilla TypeScript, Vite, HTML5 Canvas |


## Prerequisites

| Tool | Min Version | Check |
|---|---|---|
| Go | 1.22+ | `go version` |
| Node.js | 18+ | `node -v` |
| npm | 9+ | `npm -v` |
| Python | 3.10-3.11 | `python3 --version` |
| ffmpeg | any recent | `ffmpeg -version` |
| Wails CLI | v2.8+ | `wails version` |
| Git | any | `git --version` |

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y build-essential git curl ffmpeg python3 python3-venv python3-pip \
  pkg-config libwebkit2gtk-4.1-dev libgtk-3-dev

# Go
curl -LO https://go.dev/dl/go1.22.2.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin:$(go env GOPATH)/bin' >> ~/.bashrc
source ~/.bashrc

# Node.js (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 18

# Wails CLI
go install github.com/wailsapp/wails/v2/cmd/wails@latest

wails doctor
```

### Windows

```powershell
winget install -e --id GoLang.Go
winget install -e --id OpenJS.NodeJS.LTS
winget install -e --id Python.Python.3.11
winget install -e --id Gyan.FFmpeg
winget install -e --id Git.Git

go install github.com/wailsapp/wails/v2/cmd/wails@latest

wails doctor
```

### macOS

```bash
brew install go node python ffmpeg
go install github.com/wailsapp/wails/v2/cmd/wails@latest
wails doctor
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd argus
```

### 2. Python inference environment

This project uses [uv](https://github.com/astral-sh/uv) for Python and dependency management.

Install uv (one-time):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtualenv and install dependencies. The venv is created at `./venv` (not uv's default `.venv`) because that is the path the Go pipeline looks up at startup in `stream/pipeline.go`:

```bash
uv venv venv
uv pip install -r inference/requirements.txt --python venv/bin/python3
```

> The `ultralytics` package auto-downloads the YOLOv8 Nano model (~6MB) on first run.

Verify (no shell activation required - `uv run` invokes the venv interpreter directly):

```bash
uv run --python venv/bin/python3 python -c "from ultralytics import YOLO; print('ultralytics OK')"
```

### 3. Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Go dependencies

```bash
go mod tidy
```

---

## Running in Development

```bash
wails dev #or use the -tags to fix issue mentioned in Troubleshooting section below.  wails dev -tags webkit2_41 
```

This starts:
- The Go backend with an embedded WebSocket server on `localhost:8083`
- A Vite dev server with hot-reload for the frontend
- A native desktop window

1. Paste an HLS stream URL (`.m3u8`) into the input field.
2. Click **Start Pipeline Feed**.
3. Watch tracked objects appear with color-coded bounding boxes and a running unique-object counter.

### Test streams

Use any public HLS endpoint e.g. https://cctvjss.jogjakota.go.id/malioboro/Malioboro_2_Depan_Toko_Subur.stream/playlist.m3u8

---

## Building for Production

### Build the frontend

```bash
cd frontend && npm run build && cd ..
```

### Build the desktop binary

```bash
# Linux
wails build -platform linux/amd64

# Windows
wails build -platform windows/amd64

# Both
wails build -platform linux/amd64,windows/amd64
```

Output lands in `build/bin/`.

### Bundling Python for end users

End users won't have a dev venv. Two options:

**Option A - PyInstaller sidecar (recommended):**

```bash
pip install pyinstaller
pyinstaller --onefile inference/detection_engine.py --name detection_engine
```

Place the resulting binary alongside the Wails binary and update `stream/pipeline.go` to invoke it directly instead of `python3 inference/detection_engine.py`.

**Option B - Ship an install script** that creates the venv and installs dependencies on the target machine.

---

## Project Structure

```
argus/
|-- main.go                          # Wails app entry point
|-- app.go                           # App struct, WebSocket server, pipeline orchestration
|-- go.mod / go.sum                  # Go module dependencies
|-- wails.json                       # Wails CLI manifest
|-- stream/
|   `-- pipeline.go                  # ffmpeg + Python subprocess management, frame piping
|-- inference/
|   |-- detection_engine.py          # YOLOv8 detection, IoU tracker, stdin/stdout protocol
|   `-- requirements.txt             # Python dependencies
|-- frontend/
|   |-- index.html                   # UI layout (input, canvas, counter)
|   |-- src/
|   |   `-- main.ts                  # WebSocket client, frame rendering, bbox overlay
|   |-- wailsjs/                     # Auto-generated Go<->JS bindings
|   |-- package.json
|   `-- tsconfig.json
`-- .gitignore
```

---

## Configuration

Detection parameters are defined at the top of `inference/detection_engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `yolov8n.pt` | YOLOv8 model variant (nano = fastest) |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum detection confidence |
| `TRACKED_CLASSES` | `person, bicycle, car, motorcycle, bus, truck, dog, cat` | COCO classes to track |
| `DETECT_EVERY_N_FRAMES` | `5` | Run inference every Nth frame |
| `TRACK_EXPIRY_CYCLES` | `10` | Cycles before removing an unmatched track |
| `IOU_MATCH_THRESHOLD` | `0.3` | Minimum IoU to associate a detection with an existing track |

The Python interpreter used by the pipeline can be overridden with the `ARGUS_PYTHON_BIN` environment variable. By default it looks for `venv/bin/python3`, then falls back to `python3` on `PATH`.

---

## WebSocket Protocol

The frontend receives binary WebSocket messages with this format:

```
[4 bytes: header length (big-endian uint32)]
[N bytes: JSON metadata]
[remaining bytes: raw RGB24 pixel data, 640x480x3 = 921,600 bytes]
```

JSON metadata shape:

```json
{
  "totalCount": 42,
  "objects": [
    { "id": 1, "box": [120, 80, 50, 120], "label": "person" },
    { "id": 3, "box": [300, 200, 180, 100], "label": "car" }
  ]
}
```

- `box` is `[x, y, width, height]` in pixel coordinates.
- `totalCount` is the cumulative unique objects seen this session.
- `id` is a stable track ID that persists across frames.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Blank canvas, no video | `ffmpeg` not on `PATH` | Verify `ffmpeg -version` in the same shell running `wails dev` |
| Python process exits immediately | venv not found / dependencies missing | Ensure `venv/` exists with `pip install -r inference/requirements.txt` |
| No bounding boxes, video plays | YOLO model download failed | Check network access; model auto-downloads on first run (~6MB) |
| `wails dev` fails with WebView/GTK errors (Linux) | Missing system libs | `sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev`, then run `wails dev -tags webkit2_41` https://github.com/wailsapp/wails/issues/3345 |
| Stream fails to connect | Invalid or unreachable HLS URL | Verify the `.m3u8` URL loads in any HSL player e.g. https://livepush.io/hlsplayer/index.html |
