import os
import inflect
import matplotlib.pyplot as plt
import IPython.display as ipd
from tacotron2_model import Tacotron2
import torch
import numpy as np
import glow
from scipy.io.wavfile import write

import matplotlib

matplotlib.use("Agg")

from clean_text import clean_text


SYMBOLS = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
MAX_WAV_VALUE = 32768.0
SIGMA = 1.0


def load_model(model_path):
    model = Tacotron2()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


def load_vocoder(vocoder_path):
    squeezewave = torch.load(vocoder_path, map_location=torch.device('cpu'))['model']
    squeezewave = squeezewave.remove_weightnorm(squeezewave)
    return squeezewave


def generate_graph(alignments, filepath):
    data = alignments.float().data.cpu().numpy()[0].T
    plt.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.savefig(filepath)


def generate_audio(mel, vocoder, filepath, sample_rate=22050):
    with torch.no_grad():
        audio = vocoder.infer(mel, sigma=SIGMA).float()
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')

    write(filepath, sample_rate, audio)


def text_to_sequence(text):
    sequence = np.array([[SYMBOL_TO_ID[s] for s in text if s in SYMBOL_TO_ID]])
    return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def synthesize(model, vocoder, text, inflect_engine, graph=None, audio=None):
    text = clean_text(text, inflect_engine)
    sequence = text_to_sequence(text)
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)

    if graph:
        generate_graph(alignments, graph)

    if audio:
        generate_audio(mel_outputs_postnet, vocoder, audio)
