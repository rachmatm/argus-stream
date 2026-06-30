package main

import (
	"context"
	"argus/stream"
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

// App is the Wails app shell. The WebSocket implementation in this file has
// three responsibilities:
//
//  1. Listen for one client connection on :8083/stream.
//  2. Upgrade HTTP -> WebSocket when the frontend connects (handleWS).
//  3. Push two streams down that single socket:
//     - Text frames: detection metadata (JSON), one per Python inference.
//     - Binary frames: raw 921,600-byte RGB frames, one per ffmpeg frame.
//
// All writes are serialized through wsMu because *websocket.Conn is NOT
// safe for concurrent WriteMessage calls (Gorilla docs make this explicit).
type App struct {
	ctx      context.Context
	pipe     *stream.Pipeline
	upgrader websocket.Upgrader // configured in NewApp; controls the HTTP->WS handshake
	wsConn   *websocket.Conn   // current client; nil until the first /stream upgrade
	wsMu     sync.Mutex        // guards wsConn -- the writers below grab this before each WriteMessage

	frameSendCount int
	metaSendCount  int
}

func NewApp() *App {
	return &App{
		// CheckOrigin: return true is permissive -- fine for a local desktop
		// app where the page is loaded by Wails itself (same origin). Tighten
		// this if a browser ever connects cross-origin.
		upgrader: websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }},
	}
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx

	// Register the WebSocket endpoint. The path "/stream" is what the
	// frontend (frontend/src/main.ts) connects to after the user clicks
	// "Start Pipeline Feed".
	http.HandleFunc("/stream", a.handleWS)

	// ListenAndServe blocks, so it runs in its own goroutine -- otherwise
	// startup() would never return and Wails' init would hang. We log the
	// error but don't propagate it: the GUI is already up by this point.
	go func() {
		log.Println("[DEBUG] WebSocket server listening on :8083/stream")
		if err := http.ListenAndServe(":8083", nil); err != nil {
			log.Println("[DEBUG] http.ListenAndServe error:", err)
		}
	}()
}

func (a *App) StartVideoPipeline(url string) string {
	log.Println("[DEBUG] StartVideoPipeline called with url:", url)

	if a.pipe != nil {
		log.Println("[DEBUG] Closing existing pipeline before starting a new one")
		a.pipe.Close()
	}

	a.frameSendCount = 0
	a.metaSendCount = 0

	a.pipe = stream.NewPipeline(url, func(jsonStr string) {
		// METADATA WRITER (closure #1).
		// Fired once per Python detection cycle (~6 Hz when DETECT_EVERY_N=5
		// at 30 fps). Sent as a TEXT frame so the frontend can JSON.parse()
		// it directly. The wsMu lock serializes this against the frame writer
		// below -- two concurrent WriteMessage calls on the same conn would
		// interleave bytes and corrupt the stream.
		a.wsMu.Lock()
		defer a.wsMu.Unlock()

		if a.wsConn == nil {
			return
		}

		if err := a.wsConn.WriteMessage(websocket.TextMessage, []byte(jsonStr)); err != nil {
			log.Println("[DEBUG] WriteMessage (meta) failed:", err)
			// Drop the conn ref so subsequent writes no-op until handleWS
			// installs a new one. There is no explicit conn.Close() here --
			// the next WriteMessage would fail anyway with a broken pipe.
			a.wsConn = nil
			return
		}

		a.metaSendCount++
		if a.metaSendCount == 1 || a.metaSendCount%60 == 0 {
			log.Println("[DEBUG] Metadata sent over WebSocket, count:", a.metaSendCount)
		}
	}, func(frameBytes []byte) {
		// FRAME WRITER (closure #2).
		// Fired every time ffmpeg produces a decoded frame (~30 Hz). Sent as
		// a BINARY frame so the 921,600 RGB bytes ship raw -- no base64, no
		// JSON wrapping, no per-frame overhead.
		a.wsMu.Lock()
		defer a.wsMu.Unlock()

		if a.wsConn == nil {
			if a.frameSendCount%60 == 0 {
				log.Println("[DEBUG] OnFrameReady fired but wsConn is nil - no client connected yet")
			}
			a.frameSendCount++
			return
		}

		if err := a.wsConn.WriteMessage(websocket.BinaryMessage, frameBytes); err != nil {
			log.Println("[DEBUG] WriteMessage (frame) failed:", err)
			a.wsConn = nil
			return
		}

		a.frameSendCount++
		if a.frameSendCount == 1 || a.frameSendCount%60 == 0 {
			log.Println("[DEBUG] Frame sent over WebSocket, count:", a.frameSendCount)
		}
	})

	if err := a.pipe.Start(a.ctx); err != nil {
		log.Println("[DEBUG] pipe.Start error:", err)
		return "Pipeline failed to start: " + err.Error()
	}

	log.Println("[DEBUG] Pipeline started successfully")
	return "Pipeline Activated"
}

func (a *App) handleWS(w http.ResponseWriter, r *http.Request) {
	// handleWS is the HTTP -> WebSocket upgrade handler, registered at
	// /stream in startup(). Each call flips one incoming HTTP request into
	// a full-duplex WebSocket connection. Note: only ONE client is supported
	// at a time -- a second connection silently overwrites the first.
	log.Println("[DEBUG] Incoming WebSocket upgrade request from:", r.RemoteAddr)

	// Upgrade performs the HTTP 101 Switching Protocols handshake. After
	// this returns successfully, the underlying TCP conn speaks WebSocket
	// framing instead of HTTP. The third argument (nil here) is a
	// http.Header for any custom response headers in the handshake.
	conn, err := a.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("[DEBUG] WebSocket upgrade failed:", err)
		return
	}

	log.Println("[DEBUG] WebSocket upgrade succeeded")

	// Publish the new conn under wsMu. The writers in StartVideoPipeline
	// read a.wsConn under the same mutex, so this swap is race-free.
	a.wsMu.Lock()
	a.wsConn = conn
	a.wsMu.Unlock()
}

func (a *App) shutdown(ctx context.Context) {
	if a.pipe != nil {
		a.pipe.Close()
	}
}
