import inflect
import matplotlib.pyplot as plt
from torch import autograd, from_numpy
import numpy as np
from scipy.io.wavfile import write

import matplotlib

matplotlib.use("Agg")

from clean_text import clean_text
from hifigan import generate_audio_hifigan


SYMBOLS = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def generate_graph(alignments, filepath):
    data = alignments.float().data.cpu().numpy()[0].T
    plt.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.savefig(filepath)


def text_to_sequence(text):
    sequence = np.array([[SYMBOL_TO_ID[s] for s in text if s in SYMBOL_TO_ID]])
    return autograd.Variable(from_numpy(sequence)).cpu().long()


def synthesize(model, vocoder, text, inflect_engine, graph=None, audio=None):
    text = clean_text(text, inflect_engine)
    sequence = text_to_sequence(text)
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)

    if graph:
        generate_graph(alignments, graph)

    if audio:
        generate_audio_hifigan(vocoder, mel_outputs_postnet, audio)
