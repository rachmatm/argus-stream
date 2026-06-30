import sys
import json
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from frame_debug import debug_enabled, log_frame_boundaries

WIDTH, HEIGHT, CHANNELS = 640, 480, 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
TRACKED_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat"}

TRACK_EXPIRY_CYCLES = 10
IOU_MATCH_THRESHOLD = 0.3

_model = YOLO(MODEL_NAME)


def on_new_track(track: "Track") -> None:
    pass


@dataclass
class Track:
    id: int
    box: list
    label: str
    missed_cycles: int = 0


class Tracker:
    def __init__(self):
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._lock = threading.Lock()
        self.total_unique_count = 0

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        union_area = (aw * ah) + (bw * bh) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections: list) -> list:
        with self._lock:
            unmatched_track_ids = set(self._tracks.keys())
            matched_indices = set()

            for tid in list(self._tracks.keys()):
                track = self._tracks[tid]
                best_iou, best_idx = 0.0, -1
                for i, (box, label) in enumerate(detections):
                    if i in matched_indices or label != track.label:
                        continue
                    iou = self._iou(track.box, box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i

                if best_iou >= IOU_MATCH_THRESHOLD:
                    track.box = detections[best_idx][0]
                    track.missed_cycles = 0
                    matched_indices.add(best_idx)
                    unmatched_track_ids.discard(tid)

            for tid in unmatched_track_ids:
                self._tracks[tid].missed_cycles += 1
                if self._tracks[tid].missed_cycles > TRACK_EXPIRY_CYCLES:
                    del self._tracks[tid]

            for i, (box, label) in enumerate(detections):
                if i in matched_indices:
                    continue
                new_track = Track(id=self._next_id, box=box, label=label)
                self._tracks[new_track.id] = new_track
                self._next_id += 1
                self.total_unique_count += 1

                try:
                    on_new_track(new_track)
                except Exception:
                    pass

            return list(self._tracks.values())


_tracker = Tracker()


def main():
    frame_index = 0
    debug = debug_enabled()
    while True:
        raw_bytes = sys.stdin.buffer.read(FRAME_SIZE)
        if len(raw_bytes) != FRAME_SIZE:
            break

        if debug:
            log_frame_boundaries(raw_bytes, frame_index)
            frame_index += 1

        rgb_frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        detections = []
        try:
            results = _model.predict(bgr_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = _model.names[cls_id]
                    if label not in TRACKED_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], label))
        except Exception:
            pass

        tracks = _tracker.update(detections)

        payload = {
            "totalCount": _tracker.total_unique_count,
            "objects": [
                {"id": t.id, "box": t.box, "label": t.label}
                for t in tracks
            ],
        }
        sys.stderr.write(f"META:{json.dumps(payload)}\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
