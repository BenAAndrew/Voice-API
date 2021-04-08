import os
import inflect
import matplotlib.pyplot as plt
import IPython.display as ipd
from tacotron2_model import Tacotron2
import torch
import numpy as np
import glow

import matplotlib

matplotlib.use("Agg")

from clean_text import clean_text


SYMBOLS = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def load_model(model_path):
    model = Tacotron2()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["state_dict"])
    return model


def load_waveglow(waveglow_path):
    waveglow = torch.load(waveglow_path)["model"]
    for k in waveglow.convinv:
        k.float()
    return waveglow


def generate_graph(alignments, filepath):
    data = alignments.float().data.cpu().numpy()[0].T
    plt.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.savefig(filepath)


def generate_audio(mel, waveglow, filepath, sample_rate=22050):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = ipd.Audio(audio, rate=sample_rate)
    with open(filepath, "wb") as f:
        f.write(audio.data)


def text_to_sequence(text):
    sequence = np.array([[SYMBOL_TO_ID[s] for s in text if s in SYMBOL_TO_ID]])
    return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def synthesize(model, waveglow_model, text, inflect_engine, graph=None, audio=None):
    text = clean_text(text, inflect_engine)
    sequence = text_to_sequence(text)
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)

    if graph:
        generate_graph(alignments, graph)

    if audio:
        generate_audio(mel_outputs_postnet, waveglow_model, audio)
