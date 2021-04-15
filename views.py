from flask import request, jsonify, send_file
import os
import io
import inflect
import uuid
import gc
from google_drive_downloader import GoogleDriveDownloader as gdd

from app import app, DATA_FOLDER, RESULTS_FOLDER
from synthesize import load_vocoder, load_model, synthesize


REQUIRED_FILES = {
    "L128_small_pretrain.pt": "1BKCVK781QTvmkYneK9eP9_ZDKBzcTOkk",
    "David_Attenborough.pt": "1-Rq_oj3pluFE4vfIOiUpI1CrkFT62LOm"
}


def get_model_name(voice_name):
    return voice_name.replace(" ", "_") + ".pt"


def check_files():
    files = os.listdir(DATA_FOLDER)

    for name, id in REQUIRED_FILES.items():
        if name not in files:
            gdd.download_file_from_google_drive(file_id=id, dest_path=os.path.join(DATA_FOLDER, name))


# Synthesis
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
check_files()
inflect_engine = inflect.engine()
WAVEGLOW = None
MODELS = None


@app.route("/", methods=["GET"])
def index():
    global WAVEGLOW, MODELS
    check_files()
    gc.collect()

    if not WAVEGLOW:
        WAVEGLOW = load_vocoder(os.path.join(DATA_FOLDER, "L128_small_pretrain.pt"))

    if not MODELS:
        MODELS = {"David Attenborough": load_model(os.path.join(DATA_FOLDER, "David_Attenborough.pt"))}

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
    synthesize(MODELS[voice_name], WAVEGLOW, text, inflect_engine, graph=graph_path, audio=audio_path)

    return jsonify({"graph": f"results/{id}.png", "audio": f"results/{id}.wav"})


@app.route("/results/<path:path>", methods=["GET"])
def results(path):
    mimetype = "image/png" if path.endswith("png") else "audio/wav"

    if not os.path.isfile(os.path.join(RESULTS_FOLDER, path)):
        return jsonify({"error": "File not found"}), 400

    with open(os.path.join(RESULTS_FOLDER, path), "rb") as f:
        return send_file(io.BytesIO(f.read()), attachment_filename=path, mimetype=mimetype)
