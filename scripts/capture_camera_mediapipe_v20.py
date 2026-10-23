"""Capture a real webcam sequence of MediaPipe 21 hand landmarks.

Run this script with the existing ``D:\\conda\\envs\\linker_hand`` Python,
which already contains OpenCV and MediaPipe.  The output is a raw camera
landmark file, not MANO parameters; v20 fitting consumes it in a later step.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _physical_handedness(raw_label: str | None, detection_frame_mirrored: bool) -> str | None:
    """Convert MediaPipe's mirrored-input label to the physical hand label.

    MediaPipe's handedness classifier assumes a mirrored selfie image. When
    the frame actually passed to MediaPipe is unmirrored, its label must be
    exchanged. ``detection_frame_mirrored`` refers to that actual frame, not
    to the camera driver's raw output.
    """
    if detection_frame_mirrored:
        return raw_label
    if raw_label == "Left":
        return "Right"
    if raw_label == "Right":
        return "Left"
    return raw_label


def _draw_landmarks(frame, points, cv2, mirror_x: bool) -> None:
    """Draw the 21 points and a minimal hand skeleton without mp.solutions."""
    height, width = frame.shape[:2]
    coords = []
    for point in points:
        x_norm = 1.0 - float(point.x) if mirror_x else float(point.x)
        x = int(round(max(0.0, min(1.0, x_norm)) * (width - 1)))
        y = int(round(max(0.0, min(1.0, float(point.y))) * (height - 1)))
        coords.append((x, y))
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, coords[start], coords[end], (0, 220, 0), 2, cv2.LINE_AA)
    for index, (x, y) in enumerate(coords):
        cv2.circle(frame, (x, y), 5, (0, 80, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, str(index), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--display", action="store_true", help="show a preview window; press q to stop")
    parser.add_argument("--until-q", action="store_true",
                        help="with --display, keep capturing until q is pressed")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--input-mirrored", action="store_true",
                        help="use when the camera driver already returns a mirrored frame")
    parser.add_argument("--hand-landmarker-model", type=Path,
                        help="MediaPipe hand_landmarker.task; omit only with --allow-raw-only")
    parser.add_argument("--allow-raw-only", action="store_true",
                        help="capture webcam frames even when the MediaPipe task model is unavailable")
    args = parser.parse_args()
    if args.seconds <= 0 and not args.until_q:
        raise ValueError("seconds must be positive")
    if args.until_q and not args.display:
        raise ValueError("--until-q requires --display so q can stop the capture")

    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    landmarker = None
    vision = None
    if args.hand_landmarker_model is not None:
        model_path = args.hand_landmarker_model.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        from mediapipe.tasks.python import vision as mp_vision
        vision = mp_vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
    elif not args.allow_raw_only:
        raise RuntimeError("MediaPipe task model is required; pass --hand-landmarker-model or --allow-raw-only")
    frames = []
    started = time.perf_counter()
    frame_id = 0
    last_timestamp_ms = -1
    while args.until_q or time.perf_counter() - started < args.seconds:
            ok, capture_frame = cap.read()
            if not ok:
                continue
            # The VIDEO mode API requires a strictly increasing timestamp for
            # every frame.  Compute it before invoking detect_for_video(); the
            # previous version used ``timestamp`` before this assignment on
            # the first frame.
            timestamp = time.perf_counter() - started
            timestamp_ms = max(last_timestamp_ms + 1, int(timestamp * 1000.0))
            last_timestamp_ms = timestamp_ms
            # Keep a separate detection frame and preview frame.  By default
            # OpenCV supplies an unmirrored frame; --input-mirrored handles
            # cameras/drivers that already mirror their output.
            detection_frame = (cv2.flip(capture_frame, 1)
                               if args.input_mirrored else capture_frame)
            display_frame = (cv2.flip(capture_frame, 1)
                             if not args.no_mirror else capture_frame.copy())
            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            result = None
            if landmarker is not None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)
            record = {
                "frame_id": frame_id,
                "timestamp_s": float(timestamp),
                "image_width": int(capture_frame.shape[1]),
                "image_height": int(capture_frame.shape[0]),
                "handedness": None,
                "handedness_raw_mediapipe": None,
                "world_landmarks_m": None,
                "image_landmarks_normalized": None,
            }
            if result is not None and result.hand_landmarks:
                image_points = result.hand_landmarks[0]
                record["image_landmarks_normalized"] = [[float(p.x), float(p.y), float(p.z)] for p in image_points]
                if result.hand_world_landmarks:
                    world = result.hand_world_landmarks[0]
                    record["world_landmarks_m"] = [[float(p.x), float(p.y), float(p.z)] for p in world]
                if result.handedness:
                    raw_label = result.handedness[0][0].category_name
                    record["handedness_raw_mediapipe"] = raw_label
                    record["handedness"] = _physical_handedness(
                        raw_label, detection_frame_mirrored=not args.input_mirrored)
                if args.display:
                    _draw_landmarks(display_frame, image_points, cv2,
                                    mirror_x=((not args.no_mirror) != args.input_mirrored))
            frames.append(record)
            frame_id += 1
            if args.display:
                cv2.putText(display_frame, f"frames={frame_id} hand={record['handedness']}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
                cv2.imshow("v20 webcam MediaPipe capture - press q to stop", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if landmarker is not None:
        landmarker.close()
    cap.release()
    if args.display:
        cv2.destroyAllWindows()
    detected = [frame for frame in frames if frame["image_landmarks_normalized"] is not None]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v20_real_webcam_mediapipe_capture",
        "status": "completed_real_webcam_landmark_capture" if landmarker is not None else "completed_real_webcam_raw_capture_only",
        "camera_index": args.camera_index,
        "requested_seconds": None if args.until_q else args.seconds,
        "stop_mode": "until_q" if args.until_q else "duration",
        "mirror_display": not args.no_mirror,
        "detection_frame_mirrored": not bool(args.input_mirrored),
        "handedness_convention": "physical_anatomical_label; raw label is swapped only when detection input is unmirrored",
        "units": {"world_landmarks": "m", "image_landmarks": "normalized"},
        "frame_count": len(frames),
        "detected_frame_count": len(detected),
        "detector": "MediaPipe HandLandmarker task" if landmarker is not None else None,
        "handedness_counts": {
            label: sum(frame["handedness"] == label for frame in detected)
            for label in ["Left", "Right", None]
        },
        "frames": frames,
        "limitations": [
            "MediaPipe world landmarks are not MANO theta/beta parameters.",
            "The saved handedness is the physical anatomical label; handedness_raw_mediapipe preserves the classifier output.",
            "Depth orientation should be checked before fitting MANO_LEFT.",
            "No R mapping or L20 q mapping is performed by this capture script.",
            "Raw-only mode proves camera access but contains no hand landmarks.",
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "frame_count": len(frames),
                      "detected_frame_count": len(detected), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
