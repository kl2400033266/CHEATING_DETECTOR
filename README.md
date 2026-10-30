# CHEATING_DETECTOR

Beginner-friendly Flask + YOLOv8 exam-monitoring demo.

## Folder structure

CHEATING_DETECTOR/
- static/
  - style.css
- templates/
  - index.html
- uploads/
- app.py
- README.md
- requirements.txt
- requirements-local.txt
- vercel.json
- yolov8n.pt
- yolo11n-pose.pt

The `.venv/` and `__pycache__/` folders are local runtime folders and should
not be uploaded to GitHub. Model files are also ignored by Git because
Ultralytics can download them when the application starts.

## Setup in VS Code / Windows

Open the project folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python app.py
```

Open:

http://127.0.0.1:5000

## Vercel deployment

Vercel uses the lightweight `requirements.txt` so the dashboard can deploy
within the serverless function size limit. Full camera detection and telemetry
analysis require the local dependencies in `requirements-local.txt`, because
the YOLO/PyTorch stack is too large for a Vercel Function.

## Public deployment with live AI detection

Vercel cannot host the YOLO/PyTorch backend because its Python Function bundle
limit is 500 MB. Use the included `render.yaml` to deploy the full backend to
Render:

1. Create a Render Web Service from this repository.
2. Select the `render.yaml` Blueprint configuration.
3. Wait for the service to finish installing `requirements-local.txt`.
4. Copy the backend URL, for example
   `https://eduguard-ai-backend.onrender.com`.
5. In the Vercel project settings, add the environment variable
   `AI_BACKEND_URL` with that backend URL.
6. Redeploy Vercel and refresh the browser.

The live camera dashboard runs object, pose, and sound detection in the browser
using TensorFlow.js, COCO-SSD, MoveNet, and the Web Audio API, so monitoring
does not depend on a slow or sleeping backend request. Microphone samples are
analyzed locally and are not uploaded. It detects phones, books, multiple
people, microphone sound, and head turns lasting more than ten seconds.
Microphone permission and browser speech recognition are required to start a
monitored session. Only a confident human speech transcript issues a warning;
background sounds are ignored. Camera, object, and pose monitoring continues
during speech monitoring. Use Chrome or Edge for this feature.
The Render backend retains the
YOLO endpoints for uploaded-video processing and API use, and can also process
telemetry. Keep `yolov8n.pt` and `yolo11n-pose.pt` available for those backend
features.

## Telemetry analysis

The web app also supports the notebook's telemetry pipeline. Upload two CSV files
from the **Telemetry analysis** section:

- `video_telemetry.csv`: `timestamp`, `eye_gaze_angle`, `audio_db`
- `system_events.csv`: `timestamp`, `tab_switches`, `is_cheating`

The app aligns asynchronous events with `merge_asof`, fills missing signals,
creates 10-second rolling features, trains a class-balanced Random Forest, and
reports confusion matrices and precision at both 0.50 and the strict 0.90
cheating-probability threshold.

The notebook analysis is implemented in these steps:

1. Parse and sort both timestamp columns.
2. Align asynchronous system events to video records with `merge_asof` using
   the nearest previous event.
3. Forward-fill eye-gaze values, interpolate audio values, and fill missing
   event labels with zero.
4. Create 10-record rolling gaze and audio features.
5. Train a class-balanced Random Forest with an 80/20 stratified split.
6. Compare the default 0.50 decision threshold with the strict 0.90 threshold.

The current validation dataset produced:

```text
Rows used: 9,991
Default precision (0.50): 88.74%
Strict precision (0.90): 100.00%
```

The analysis result is returned by `POST /analyze-data` as JSON and is shown
directly in the telemetry panel.

## Note

The system only flags detected objects such as phones/books for human review.
It should not be used as automatic proof that a person cheated.

During live monitoring, the exam also stops immediately when more than one
person is detected. A sustained head-turn posture stops the exam after more
than 10 continuous seconds. The timer resets when the head returns to normal;
brief separate turns are not added together. These are automated review safeguards and should be
validated by a human invigilator.
