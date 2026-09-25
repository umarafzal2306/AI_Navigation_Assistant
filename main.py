# ============================================================
# main.py — AI Navigation Assistant Entry Point
# ============================================================
#
# Real-time pipeline:
#   Webcam → YOLOv8 Detection → MiDaS Depth → Direction →
#   ByteTrack Tracking → Decision Engine → Voice Alerts
#
# Keyboard shortcuts:
#   Q / ESC  — Quit
#   H        — Toggle language (English ↔ Hindi)
#   D        — Trigger LM Studio scene description
#
# Usage:
#   python main.py
#   python main.py --language hi
#   python main.py --model yolov8n.pt
# ============================================================

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

from utils.logger import get_logger, setup_file_logging
from utils.frame_utils import resize_frame, annotate_frame, draw_status_bar, frame_to_base64
from modules.detector import ObjectDetector
from modules.depth_estimator import DepthEstimator
from modules.tracker import ObjectTracker
from modules.direction import get_direction
from modules.decision_engine import DecisionEngine
from modules.voice import VoiceEngine
from modules.lm_studio_client import LMStudioClient
from modules.camera import CameraStream

log = get_logger("main")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        log.warning(f"Config file not found: {config_path} — using defaults")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    log.info(f"Configuration loaded from {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (override config.yaml values)."""
    parser = argparse.ArgumentParser(
        description="AI Navigation Assistant for Visually Impaired People"
    )
    parser.add_argument(
        "--language", "-l",
        choices=["en", "hi"],
        default=None,
        help="Default voice language: en (English) or hi (Hindi)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="YOLOv8 model path (e.g., yolov8n.pt, yolov8x.pt)",
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default=None,
        help="Camera device index (e.g. 0) or IP stream URL (e.g. http://192.168.1.15:8080/video)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the video display window",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable local LM Studio LLM brain",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml file",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to config dict."""
    if args.language:
        config.setdefault("voice", {})["default_language"] = args.language

    if args.model:
        config.setdefault("detection", {})["model_path"] = args.model

    if args.camera is not None:
        cam_val = int(args.camera) if args.camera.isdigit() else args.camera
        config.setdefault("camera", {})["device_index"] = cam_val

    if args.no_display:
        config.setdefault("display", {})["show_window"] = False

    if args.no_llm:
        config.setdefault("lm_studio", {})["enabled"] = False

    return config


def main():
    """Run the AI Navigation Assistant pipeline."""
    # ---- Parse args & load config ----
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    # Setup logging
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    main_logger = get_logger("main", log_level)

    if log_config.get("log_to_file", False):
        setup_file_logging(main_logger, log_config.get("log_file", "nav_assistant.log"))

    main_logger.info("=" * 60)
    main_logger.info("  AI Navigation Assistant — Starting Up")
    main_logger.info("=" * 60)

    # ---- Initialize all modules ----
    main_logger.info("Initializing modules...")

    # 1. Object Detector (YOLOv8)
    detector = ObjectDetector(config)

    # 2. Depth Estimator (MiDaS)
    depth_estimator = DepthEstimator(config)

    # 3. Object Tracker
    tracker = ObjectTracker(config)

    # 4. Decision Engine
    decision_engine = DecisionEngine(config)
    decision_engine.set_tracker(tracker)

    # 5. Voice Engine
    voice_engine = VoiceEngine(config)
    voice_engine.start()

    # 6. LM Studio Client (local LLM brain)
    lm_client = LMStudioClient(config)
    lm_client.set_voice_engine(voice_engine)

    # Direction config
    dir_config = config.get("direction", {})
    left_boundary = dir_config.get("left_boundary", 0.33)
    right_boundary = dir_config.get("right_boundary", 0.66)

    # Camera config
    cam_config = config.get("camera", {})
    cam_index = cam_config.get("device_index", 0)
    if isinstance(cam_index, str) and cam_index.isdigit():
        cam_index = int(cam_index)
    frame_width = cam_config.get("frame_width", 640)
    frame_height = cam_config.get("frame_height", 480)

    # Display config
    display_config = config.get("display", {})
    show_window = display_config.get("show_window", True)
    window_name = display_config.get("window_name", "AI Navigation Assistant")

    # Performance config
    perf_config = config.get("performance", {})
    process_every_n = max(1, perf_config.get("process_every_n_frames", 3))
    main_logger.info(f"Frame skipping: processing every {process_every_n} frame(s)")

    # ---- Open webcam (threaded async reader) ----
    main_logger.info(f"Opening camera (source={cam_index})...")
    # Suppress harmless MJPEG decoder warnings from IP camera streams
    os.environ.setdefault("OPENCV_LOG_LEVEL", "0")
    try:
        cv2.setLogLevel(0)  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Older OpenCV builds don't expose setLogLevel — env var covers it

    cam = CameraStream(cam_index, frame_width, frame_height)

    if not cam.is_opened():
        main_logger.error("Failed to open webcam! Check camera connection.")
        voice_engine.speak("Camera not found. Please check your webcam connection.", "en")
        voice_engine.stop()
        sys.exit(1)

    cam.start()
    main_logger.info(f"Camera opened — {frame_width}x{frame_height} (async reader)")

    # ---- Startup greeting ----
    language = voice_engine.get_language()
    greeting = lm_client.get_startup_greeting(language)
    if greeting:
        main_logger.info(f"Startup greeting: {greeting}")
        voice_engine.speak(greeting, language)

    # ---- FPS tracking ----
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    # ---- Language toggle cooldown (prevents double-fire from key repeat) ----
    last_lang_toggle_time = 0.0
    LANG_TOGGLE_COOLDOWN = 2.0  # seconds

    # ---- Last annotated state (for LM Studio) ----
    last_frame_base64 = ""
    last_detections = []

    # ---- Frame counter for skipping ----
    frame_index = 0

    # ---- Cached state for skipped frames ----
    cached_tracked_objects = []
    cached_annotated_frame = None

    main_logger.info(
        f"Pipeline running — press Q to quit, H to toggle language, D for scene description"
    )

    # ============================================================
    # MAIN LOOP
    # ============================================================
    try:
        while True:
            loop_start = time.time()

            # ---- 1. Capture frame (async — always the latest) ----
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Resize to standard dimensions
            frame = resize_frame(frame, frame_width, frame_height)
            frame_index += 1

            # ---- Frame skip: only run AI on every Nth frame ----
            if frame_index % process_every_n == 0:

                # ---- 2. YOLOv8 Detection ----
                detections = detector.detect(frame)

                # ---- 3. MiDaS Depth Estimation (CONDITIONAL) ----
                # Only run the heavy depth model if objects were detected
                if len(detections) > 0:
                    depth_map = depth_estimator.estimate(frame)

                    # ---- 4. Enrich detections with direction + distance ----
                    for det in detections:
                        det["direction"] = get_direction(
                            det["center_x"], frame_width, left_boundary, right_boundary
                        )
                        det["distance_m"] = depth_estimator.get_distance(depth_map, det["bbox"])

                # ---- 5. Update tracker ----
                tracked_objects = tracker.update(detections, frame_width)
                cached_tracked_objects = tracked_objects

                # ---- 6. Decision engine → alerts ----
                alerts = decision_engine.evaluate(tracked_objects)

                # ---- 7. Send alerts to voice engine ----
                language = voice_engine.get_language()
                for alert in alerts:
                    message = alert.get_message(language)
                    is_urgent = alert.level == "urgent"
                    voice_engine.speak(message, language, urgent=is_urgent)
                    main_logger.info(
                        f"[{alert.level.upper()}] {message} "
                        f"(dist={alert.tracked_object.get('distance_m', '?'):.1f}m, "
                        f"id={alert.tracked_object.get('track_id', '?')})"
                    )

                # ---- 8. Annotate frame ----
                for obj in tracked_objects:
                    if "alert_level" not in obj:
                        dist = obj.get("distance_m", 999)
                        if dist < decision_engine.urgent_threshold:
                            obj["alert_level"] = "urgent"
                        elif dist < decision_engine.warning_threshold:
                            obj["alert_level"] = "warning"
                        elif dist < decision_engine.info_threshold:
                            obj["alert_level"] = "info"
                        else:
                            obj["alert_level"] = "silent"

                annotated_frame = annotate_frame(frame, tracked_objects)
                cached_annotated_frame = annotated_frame

                # ---- 12. Auto-trigger LM Studio scene description ----
                if lm_client.should_auto_trigger() and len(detections) > 0:
                    frame_b64 = frame_to_base64(annotated_frame)
                    lm_client.describe_scene_async(frame_b64, detections, language)
                    lm_client.mark_triggered()

            else:
                # Skipped frame: reuse cached annotations on the new frame
                annotated_frame = (
                    annotate_frame(frame, cached_tracked_objects)
                    if cached_tracked_objects
                    else frame
                )

            # ---- 9. Status bar (always drawn) ----
            language = voice_engine.get_language()
            annotated_frame = draw_status_bar(
                annotated_frame,
                language=language,
                fps=current_fps,
                num_objects=len(cached_tracked_objects),
                groq_status=lm_client.get_status(),
            )

            # ---- 10. Display (always shown) ----
            if show_window:
                cv2.imshow(window_name, annotated_frame)

            # ---- 13. Keyboard input ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                main_logger.info("Quit command received")
                break

            elif key == ord("h"):  # Toggle language (with cooldown guard)
                now = time.time()
                if now - last_lang_toggle_time >= LANG_TOGGLE_COOLDOWN:
                    last_lang_toggle_time = now
                    new_lang = voice_engine.toggle_language()
                    main_logger.info(f"Language toggled to: {new_lang}")
                else:
                    main_logger.debug("Language toggle ignored — cooldown active")

            elif key == ord("d"):  # LM Studio scene description
                main_logger.info("Manual scene description triggered")
                frame_b64 = frame_to_base64(annotated_frame)
                lm_client.describe_scene_async(frame_b64, [], language)
                lm_client.mark_triggered()

            # ---- FPS calculation ----
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()

    except KeyboardInterrupt:
        main_logger.info("Keyboard interrupt — shutting down")

    finally:
        # ---- Cleanup ----
        main_logger.info("Shutting down...")
        cam.stop()
        cv2.destroyAllWindows()
        voice_engine.stop()
        main_logger.info("AI Navigation Assistant stopped. Goodbye!")


if __name__ == "__main__":
    main()
