# 🚀 How to Setup and Run the AI Navigation Assistant

This guide explains how to set up and run the AI Navigation Assistant on your local machine.

---

## 📋 Prerequisites
Before you start, make sure you have the following installed:
1. **Python 3.8 to 3.11** (Python 3.12+ might have compatibility issues with some ML dependencies like PyTorch/torchvision depending on OS packages).
2. A working **Webcam** (built-in or USB).
3. (Optional) An **Anthropic API Key** if you want to use the Claude scene description features.

---

## ⚙️ Step-by-Step Setup

### Step 1: Navigate to Project Directory
Open your terminal/command prompt and navigate to the project directory:
```bash
cd AI_Navigation_Assistant
```

### Step 2: Create a Virtual Environment (Recommended)
This isolates the project dependencies so they do not conflict with other Python packages on your computer.

*   **Windows**:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### Step 3: Install Dependencies
Install all the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```
> *Note: This will download and install PyTorch, OpenCV, YOLOv8 (ultralytics), MiDaS (timm/torchhub), FastAPI, and other packages. It might take a few minutes.*

### Step 4: Configure API Key (Optional)
If you want to use Claude for generating intelligent voice descriptions of the scene:
1. Make a copy of `.env.example` and name it `.env`:
   *   **Windows (CMD/PowerShell)**:
       ```powershell
       copy .env.example .env
       ```
   *   **macOS / Linux**:
       ```bash
       cp .env.example .env
       ```
2. Open the newly created `.env` file in a text editor and add your Anthropic key:
   ```env
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

---

## 🏃‍♂️ Running the Project

You can run the project in two different modes: **Terminal/Window Mode** or **Web Dashboard Mode**.

### Option A: Standard Window Mode (main.py)
This runs the pipeline locally and opens an OpenCV window with the webcam feed, annotations, and voice alerts.

```bash
python main.py
```

#### Command-Line Options:
*   **Run with Hindi voice outputs**:
    ```bash
    python main.py --language hi
    ```
*   **Use a lighter model (faster on CPU)**:
    ```bash
    python main.py --model yolov8n.pt
    ```
*   **Run without opening a visual window**:
    ```bash
    python main.py --no-display
    ```
*   **Run without Claude integration**:
    ```bash
    python main.py --no-claude
    ```

---

### Option B: Web Dashboard Mode (server.py)
This starts a FastAPI server. It processes the camera feed in a background thread and streams the video, telemetry, and live alerts directly to a beautiful web interface.

1. Start the server:
   ```bash
   python server.py
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:8765
   ```
3. Use the web control panel to toggle the camera, mute sounds, change detection thresholds, or swap models dynamically.

---

## ⌨️ Control Keys (When Window is Active)
If running standard mode (`main.py`) or local window in server mode:
*   Press **`Q`** or **`ESC`** to exit/quit.
*   Press **`H`** to swap languages instantly between English and Hindi.
*   Press **`D`** to manually trigger a detailed AI Scene Description using Claude.
