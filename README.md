# 🦯 AI Navigation Assistant for Visually Impaired People

A real-time AI-powered navigation assistant that uses a laptop webcam to detect objects, estimate distances, and deliver **bilingual (English + Hindi) voice guidance** to visually impaired users.

Built with **YOLOv8**, **MiDaS depth estimation**, **ByteTrack** object tracking, and **Claude claude-opus-4-6** for intelligent scene descriptions.

---

## ✨ Features

- 🎯 **Real-time object detection** — YOLOv8 with 30+ navigation-relevant classes
- 📏 **Monocular depth estimation** — MiDaS v2.1 approximates distance to each object
- 🧭 **Directional awareness** — Objects classified as LEFT / CENTER / RIGHT
- 🔄 **Persistent tracking** — ByteTrack assigns IDs, prevents repeated alerts
- 🚨 **Smart alert system** — URGENT / WARNING / INFO levels with priority scoring
- 🗣️ **Bilingual voice output** — English (pyttsx3, offline) + Hindi (gTTS)
- 🤖 **AI scene descriptions** — Claude claude-opus-4-6 generates natural language summaries
- 🛡️ **Anti-hallucination safeguards** — 6-layer verification prevents false alerts

---

## 🏗️ Architecture

```
┌──────────────┐
│  Webcam Feed │  (OpenCV, 30fps)
└──────┬───────┘
       │ frame (640×480)
       ▼
┌──────────────┐    ┌──────────────────┐
│  YOLOv8      │    │  MiDaS Depth     │
│  Detector    │    │  Estimator       │
└──────┬───────┘    └────────┬─────────┘
       │ detections          │ depth_map
       ▼                     ▼
┌──────────────────────────────────────┐
│  Direction + Depth Sampling          │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  ByteTrack Object Tracker            │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Decision Engine (Rule-Based)        │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Voice Engine (Async TTS Thread)     │
└──────────────────────────────────────┘
       ↕
┌─────────────────────┐
│  Claude API (Async)  │  ← 'D' key or auto every 30s
└─────────────────────┘
```

---

## 📁 Project Structure

```
ai-nav-assistant/
├── main.py                     # Entry point — full pipeline
├── config.yaml                 # All tunable parameters
├── requirements.txt            # Python dependencies
├── .env.example                # API key template
├── modules/
│   ├── detector.py             # YOLOv8 wrapper
│   ├── depth_estimator.py      # MiDaS wrapper
│   ├── tracker.py              # ByteTrack object tracking
│   ├── direction.py            # LEFT/CENTER/RIGHT zones
│   ├── decision_engine.py      # Alert rules + priority logic
│   ├── voice.py                # Async bilingual TTS
│   └── claude_client.py        # Claude API wrapper
├── utils/
│   ├── logger.py               # Structured colored logging
│   ├── frame_utils.py          # Frame annotation + encoding
│   └── calibration.py          # Depth-to-meters mapping
└── models/
    └── yolov8x.pt              # YOLOv8 model weights
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd ai-nav-assistant

# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key (optional, for Claude features)

```bash
copy .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run

```bash
python main.py
```

### Command-Line Options

```bash
python main.py --language hi          # Start in Hindi
python main.py --model yolov8n.pt     # Use lighter model (faster on CPU)
python main.py --camera 1             # Use different webcam
python main.py --no-display           # Headless mode (no video window)
python main.py --no-claude            # Disable Claude API
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit the application |
| `H` | Toggle language (English ↔ Hindi) |
| `D` | Trigger Claude scene description |

---

## 🛡️ Anti-Hallucination Safeguards

| Layer | Technique | Threshold |
|-------|-----------|-----------|
| 1 | Confidence thresholding | ≥ 80% |
| 2 | Bounding box area filter | ≥ 1% of frame |
| 3 | Class whitelist | Navigation-relevant only |
| 4 | Multi-frame verification | 3 consecutive frames |
| 5 | Tracking stability | 5+ tracked frames |
| 6 | Distance variance check | ≤ 20% variance |

Claude is used **only** for scene descriptions — **never** for stop/go decisions.

---

## ⚙️ Configuration

Edit `config.yaml` to tune:

- **Detection**: model, confidence threshold, target classes
- **Depth**: calibration scale/offset, smoothing frames
- **Tracking**: cooldown period, re-announce thresholds
- **Decision**: distance thresholds for URGENT/WARNING/INFO
- **Voice**: language, speech rate, volume
- **Claude**: model, auto-trigger interval, enable/disable

---

## 📊 Performance Targets

| Stage | Target Latency |
|-------|---------------|
| Frame Capture | ~10ms |
| YOLOv8 (CPU) | 15–30ms |
| MiDaS Depth (CPU) | 20–40ms |
| Tracking | ~5ms |
| Decision Engine | < 2ms |
| TTS | Async (non-blocking) |
| **Total** | **< 100ms** |

---

## 📄 License

This project is for educational and accessibility research purposes.
