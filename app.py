import json
import os
import sys
import gc

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, render_template, request, jsonify, send_file, Response
from pipeline import GymFormCheckerPipeline

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.dirname(os.path.abspath(__file__))
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"success": False, "error": "Video file exceeds the 15 MB limit. Please upload a smaller video clip."}), 413

# Initialize pipeline instance (cached in memory)
pipeline = GymFormCheckerPipeline()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sample_results")
def get_sample_results():
    """Returns the pre-analyzed squat1_analysis.json for instant demo testing."""
    json_path = os.path.join(app.config["UPLOAD_FOLDER"], "squat1_analysis.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data, "video_url": "/video/squat1_annotated.mp4"})
    return jsonify({"success": False, "error": "Sample analysis file not found."}), 404

@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """Uploads and analyzes a new workout video."""
    exercise_override = request.form.get("exercise")
    if exercise_override == "auto":
        exercise_override = None

    use_sample = request.form.get("use_sample") == "true"

    if use_sample:
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], "squat1.mp4")
    else:
        if "video" not in request.files:
            return jsonify({"success": False, "error": "No video file uploaded."}), 400
        file = request.files["video"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename."}), 400

        video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(video_path)

    try:
        results = pipeline.process_video(
            video_path=video_path,
            exercise_override=exercise_override,
            generate_video=False
        )
        orig_filename = os.path.basename(video_path)
        return jsonify({
            "success": True,
            "data": results,
            "video_url": f"/video/{orig_filename}",
            "original_video_url": f"/video/{orig_filename}"
        })
    except Exception as e:
        print(f"Pipeline error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        gc.collect()

@app.route("/video/<filename>")
def stream_video(filename):
    """Streams video file with proper MIME type."""
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        # Fallback to squat1.mp4 if annotated video isn't found
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "squat1.mp4")
    return send_file(file_path, mimetype="video/mp4")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 FormAnalyzer Web Server Running!")
    print("👉 Open your browser at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
