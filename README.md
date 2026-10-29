# CHEATING_DETECTOR

Beginner-friendly Flask + YOLOv8 exam-monitoring demo.

## Folder structure

CHEATING_DETECTOR/
- static/style.css
- templates/index.html
- uploads/
- app.py
- README.md
- requirements.txt
- yolov8n.pt (downloaded automatically by Ultralytics if missing)

## Setup in VS Code / Windows

Open the project folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open:

http://127.0.0.1:5000

## Telemetry analysis

The web app also supports the notebook's telemetry pipeline. Upload two CSV files
from the **Telemetry analysis** section:

- `video_telemetry.csv`: `timestamp`, `eye_gaze_angle`, `audio_db`
- `system_events.csv`: `timestamp`, `tab_switches`, `is_cheating`

The app aligns asynchronous events with `merge_asof`, fills missing signals,
creates 10-second rolling features, trains a class-balanced Random Forest, and
reports confusion matrices and precision at both 0.50 and the strict 0.90
cheating-probability threshold.

## Note

The system only flags detected objects such as phones/books for human review.
It should not be used as automatic proof that a person cheated.

During live monitoring, the exam also stops immediately when more than one
person is detected. A sustained head-turn posture stops the exam after more
than 10 continuous seconds. The timer resets when the head returns to normal;
brief separate turns are not added together. These are automated review safeguards and should be
validated by a human invigilator.
