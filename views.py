from flask import request, jsonify, send_file
import os
import io
import inflect
import uuid
import gc
import json
from google_drive_downloader import GoogleDriveDownloader as gdd

from app import app, DATA_FOLDER, RESULTS_FOLDER
from hifigan import load_hifigan_model
from synthesize import load_model, synthesize


VOCODER_FILES = {
    "hifigan.pt": "15-CfChiUdX2zay0kZmQNuXd0jRsqfmhq",
    "config.json": "1sJ71OLN6FcP7sY4vsKTrm0SJnp4flDy2",
}
HIFIGAN_MODEL = "hifigan.pt"
HIFIGAN_CONFIG = "config.json"

with open("voices.json") as f:
    VOICES = json.load(f)


def get_model_name(voice_name):
    return voice_name.replace(" ", "_") + ".pt"


def check_files():
    files = os.listdir(DATA_FOLDER)

    for name, id in VOCODER_FILES.items():
        if name not in files:
            gdd.download_file_from_google_drive(file_id=id, dest_path=os.path.join(DATA_FOLDER, name))

    for name, id in VOICES.items():
        if name not in files:
            gdd.download_file_from_google_drive(file_id=id, dest_path=os.path.join(DATA_FOLDER, get_model_name(name)))


# Synthesis
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
inflect_engine = inflect.engine()
VOCODER = None
MODELS = None


@app.route("/", methods=["GET"])
def index():
    global VOCODER, MODELS
    check_files()
    gc.collect()

    if not VOCODER:
        VOCODER = load_hifigan_model(os.path.join(DATA_FOLDER, HIFIGAN_MODEL), os.path.join(DATA_FOLDER, HIFIGAN_CONFIG))

    if not MODELS:
        MODELS = {name: load_model(os.path.join(DATA_FOLDER, get_model_name(name))) for name in VOICES}

    voice_name = request.args.get("name")
    if not voice_name:
        return jsonify({"error": "No name given"}), 400

    text = request.args.get("text")
    if not text:
        return jsonify({"error": "No text given"}), 400

    if voice_name not in MODELS:
        return jsonify({"error": "Voice not found"}), 400

    id = str(uuid.uuid4())
    graph_path = os.path.join(RESULTS_FOLDER, f"{id}.png")
    audio_path = os.path.join(RESULTS_FOLDER, f"{id}.wav")
    synthesize(MODELS[voice_name], VOCODER, text, inflect_engine, graph=graph_path, audio=audio_path)

    return jsonify({"graph": f"results/{id}.png", "audio": f"results/{id}.wav"})


@app.route("/results/<path:path>", methods=["GET"])
def results(path):
    mimetype = "image/png" if path.endswith("png") else "audio/wav"

    if not os.path.isfile(os.path.join(RESULTS_FOLDER, path)):
        return jsonify({"error": "File not found"}), 400

    with open(os.path.join(RESULTS_FOLDER, path), "rb") as f:
        return send_file(io.BytesIO(f.read()), attachment_filename=path, mimetype=mimetype)
