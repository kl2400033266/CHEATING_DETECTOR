from flask import Flask, render_template, request, Response, jsonify
import base64
import os
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
configured_backend_url = os.environ.get("AI_BACKEND_URL", "").strip()
AI_BACKEND_URL = configured_backend_url.rstrip("/")
if not AI_BACKEND_URL and os.environ.get("VERCEL"):
    AI_BACKEND_URL = "https://cheating-detector-5jt3.onrender.com"


@app.after_request
def allow_frontend_requests(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    np = None
    YOLO = None

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, precision_score
except ImportError:
    pd = None
    RandomForestClassifier = None
    train_test_split = None
    confusion_matrix = None
    precision_score = None

model = None
pose_model = None

# Simple demo rule: flag objects that may deserve review during an exam.
# This does NOT automatically determine that a student cheated.
REVIEW_OBJECTS = {"cell phone", "book"}
TELEMETRY_FEATURES = [
    "eye_gaze_angle",
    "audio_db",
    "tab_switches",
    "gaze_rolling_10s",
    "audio_rolling_10s",
]


def get_detection_models():
    global model, pose_model
    if YOLO is None:
        raise RuntimeError(
            "Live detection is unavailable in this deployment. "
            "Run locally with the full requirements.txt file."
        )
    if model is None:
        model = YOLO("yolov8n.pt")
    if pose_model is None:
        pose_model = YOLO("yolo11n-pose.pt")
    return model, pose_model


def process_video(video_path):
    if cv2 is None:
        raise RuntimeError("Video detection dependencies are not installed.")
    detection_model, _ = get_detection_models()
    cap = cv2.VideoCapture(str(video_path))
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = detection_model(frame, verbose=False)
        annotated = results[0].plot()

        detected = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = detection_model.names[cls_id]
            detected.append(name)

        review = any(name in REVIEW_OBJECTS for name in detected)
        label = "REVIEW EVENT" if review else "Monitoring"
        cv2.putText(
            annotated, label, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255) if review else (0, 180, 0), 2
        )

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()


def detect_frame(frame):
    detection_model, pose_detection_model = get_detection_models()
    results = detection_model(frame, verbose=False, conf=0.15, imgsz=640)
    annotated = results[0].plot()
    detections = [
        {
            "name": detection_model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 2),
        }
        for box in results[0].boxes
    ]
    detected = [item["name"] for item in detections]
    person_count = sum(name == "person" for name in detected)
    pose_results = pose_detection_model(frame, verbose=False, conf=0.15, imgsz=640)
    head_rotated = False
    if pose_results[0].keypoints is not None:
        keypoint_confidences = pose_results[0].keypoints.conf
        for index, keypoints in enumerate(pose_results[0].keypoints.xy):
            if len(keypoints) < 7:
                continue
            if keypoint_confidences is not None:
                confidence = keypoint_confidences[index]
                if min(float(confidence[0]), float(confidence[5]), float(confidence[6])) < 0.25:
                    continue
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            shoulder_width = float(abs(right_shoulder[0] - left_shoulder[0]))
            if shoulder_width <= 0:
                continue
            shoulder_center_x = float((left_shoulder[0] + right_shoulder[0]) / 2)
            horizontal_offset = abs(float(nose[0]) - shoulder_center_x)
            head_rotated = horizontal_offset / shoulder_width > 0.25
            if head_rotated:
                break
    review = any(name in REVIEW_OBJECTS for name in detected)
    cv2.putText(
        annotated,
        "MULTIPLE PEOPLE" if person_count > 1 else (
            "HEAD TURN DETECTED" if head_rotated else (
                "REVIEW EVENT" if review else "Monitoring"
            )
        ),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if (review or person_count > 1 or head_rotated) else (0, 180, 0),
        2,
    )
    return annotated, detected, detections, person_count, head_rotated


@app.route("/detect-frame", methods=["POST"])
def detect_frame_route():
    if cv2 is None or np is None:
        return jsonify({
            "error": "Live detection is unavailable on Vercel because AI model "
            "dependencies exceed the serverless size limit. Run the app locally."
        }), 503
    image = request.files.get("frame")
    if image is None:
        return jsonify({"error": "No camera frame received"}), 400

    encoded = np.frombuffer(image.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid camera frame"}), 400

    annotated, detected, detections, person_count, head_rotated = detect_frame(frame)
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        return jsonify({"error": "Could not encode detected frame"}), 500
    review = any(name in REVIEW_OBJECTS for name in detected)
    return jsonify({
        "image": base64.b64encode(buffer.tobytes()).decode("ascii"),
        "review": review,
        "detected": detected,
        "detections": detections,
        "person_count": person_count,
        "multiple_people": person_count > 1,
        "head_rotated": head_rotated,
    })


def prepare_telemetry(video_df, system_df):
    """Align telemetry streams and create the notebook's rolling features."""
    required_video = {"timestamp", "eye_gaze_angle", "audio_db"}
    required_system = {"timestamp", "tab_switches", "is_cheating"}
    missing_video = required_video - set(video_df.columns)
    missing_system = required_system - set(system_df.columns)
    if missing_video or missing_system:
        missing = sorted(missing_video | missing_system)
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    video_df = video_df.copy()
    system_df = system_df.copy()
    video_df["timestamp"] = pd.to_datetime(video_df["timestamp"], errors="coerce")
    system_df["timestamp"] = pd.to_datetime(system_df["timestamp"], errors="coerce")
    video_df = video_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    system_df = system_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if video_df.empty or system_df.empty:
        raise ValueError("Both CSV files must contain valid timestamp values.")

    merged = pd.merge_asof(
        video_df,
        system_df[["timestamp", "tab_switches", "is_cheating"]],
        on="timestamp",
        direction="backward",
    )
    merged["eye_gaze_angle"] = merged["eye_gaze_angle"].ffill()
    merged["audio_db"] = merged["audio_db"].interpolate().bfill().ffill()
    merged[["tab_switches", "is_cheating"]] = (
        merged[["tab_switches", "is_cheating"]].fillna(0)
    )
    merged["gaze_rolling_10s"] = (
        merged["eye_gaze_angle"].rolling(window=10, min_periods=10).mean()
    )
    merged["audio_rolling_10s"] = (
        merged["audio_db"].rolling(window=10, min_periods=10).max()
    )
    return merged.dropna(subset=TELEMETRY_FEATURES + ["is_cheating"])


def analyze_telemetry(video_df, system_df):
    if any(item is None for item in (
        pd, RandomForestClassifier, train_test_split,
        confusion_matrix, precision_score
    )):
        raise RuntimeError(
            "Telemetry analysis is unavailable in this lightweight deployment. "
            "Run locally with the full requirements.txt file."
        )
    merged = prepare_telemetry(video_df, system_df)
    if merged["is_cheating"].nunique() < 2:
        raise ValueError("is_cheating must contain both 0 and 1 classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        merged[TELEMETRY_FEATURES],
        merged["is_cheating"].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=merged["is_cheating"].astype(int),
    )
    classifier = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)[:, 1]
    default_predictions = (probabilities >= 0.50).astype(int)
    strict_predictions = (probabilities >= 0.90).astype(int)

    return {
        "rows_used": int(len(merged)),
        "default_confusion_matrix": confusion_matrix(
            y_test, default_predictions, labels=[0, 1]
        ).tolist(),
        "strict_confusion_matrix": confusion_matrix(
            y_test, strict_predictions, labels=[0, 1]
        ).tolist(),
        "default_precision": round(float(precision_score(
            y_test, default_predictions, zero_division=0
        )), 4),
        "strict_precision": round(float(precision_score(
            y_test, strict_predictions, zero_division=0
        )), 4),
        "threshold": 0.90,
    }

@app.route("/")
def index():
    return render_template("index.html", api_base_url=AI_BACKEND_URL)

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video selected"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No video selected"}), 400

    safe_name = Path(file.filename).name
    path = UPLOAD_FOLDER / safe_name
    file.save(path)
    return jsonify({"video": safe_name})


@app.route("/analyze-data", methods=["POST"])
def analyze_data():
    if "video_telemetry" not in request.files or "system_events" not in request.files:
        return jsonify({
            "error": "Upload both video_telemetry.csv and system_events.csv."
        }), 400
    try:
        video_df = pd.read_csv(request.files["video_telemetry"])
        system_df = pd.read_csv(request.files["system_events"])
        return jsonify(analyze_telemetry(video_df, system_df))
    except (ValueError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400

@app.route("/video/<filename>")
def video(filename):
    path = UPLOAD_FOLDER / Path(filename).name
    if not path.exists():
        return "Video not found", 404
    return Response(
        process_video(path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
