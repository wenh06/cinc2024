import argparse
import logging
import os
import pickle
import random
import shutil
import uuid
import warnings
from collections import namedtuple
from pathlib import Path
from sys import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import spacy
import tensorflow as tf
import validators
from bs4 import BeautifulSoup, Comment
from constants import CACHE_DIR, HANDWRITTEN_TEXT_DIR, save_img_ext
from PIL import Image

(CACHE_DIR / "hw_text").mkdir(exist_ok=True, parents=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)

# prevent tensorflow from allocating the entire GPU memory at once
gpus = tf.config.experimental.list_physical_devices("GPU")
n_gpus = len(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_parser():
    description = "Create a corpus for medical corpus"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-l", "--link", type=str, required=True)
    parser.add_argument("-n", "--num_words", type=int, required=True)
    parser.add_argument("-x_offset", dest="x_offset", type=int, default=0)
    parser.add_argument("-y_offset", dest="y_offset", type=int, default=0)
    parser.add_argument("-hws", dest="handwriting_size_factor", type=float, default=0.2)
    parser.add_argument("-s", dest="source_dir", type=str, required=True)
    parser.add_argument("-i", dest="input_file", type=str, required=True)
    parser.add_argument("-o", dest="output_dir", type=str, required=True)
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        default=os.path.join(os.path.join(HANDWRITTEN_TEXT_DIR, "pretrained"), "model-29"),
    )
    parser.add_argument("--text", dest="text", type=str, default=None)
    parser.add_argument("--style", dest="style", type=int, default=None)
    parser.add_argument("--bias", dest="bias", type=float, default=1.0)
    parser.add_argument("--force", dest="force", action="store_true", default=False)
    parser.add_argument("--animation", dest="animation", action="store_true", default=False)
    parser.add_argument("--noinfo", dest="info", action="store_false", default=True)
    parser.add_argument("--save", dest="save", type=str, default=None)

    return parser


# Sample random from a multivariate normal
def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho], [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


# Plot the handwriting
def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    # Traverse through the handwritten text points and plot strokes
    for e in range(len(points)):
        if points[e, 2] == 1.0:
            strokes += [points[b : e + 1, :2].copy()]
            b = e + 1
    return strokes


# Get cumulative sum of points
def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


# Code snippet from https://github.com/Grzego/handwriting-generation
def sample_text(sess, args_text, translation, force, bias, style=None):
    # fmt: off
    fields = [
        "coordinates", "sequence", "bias", "e", "pi", "mu1", "mu2", "std1", "std2",
        "rho", "window", "kappa", "phi", "finish", "zero_states",
    ]
    vs = namedtuple("Params", fields)(*[tf.compat.v1.get_collection(name)[0] for name in fields])
    # fmt: on

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0.0, 0.0, 1.0])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0]  # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text]  # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len
        e, pi, mu1, mu2, std1, std2, rho, finish, phi, window, kappa = sess.run(
            [vs.e, vs.pi, vs.mu1, vs.mu2, vs.std1, vs.std2, vs.rho, vs.finish, vs.phi, vs.window, vs.kappa],
            feed_dict={
                vs.coordinates: coord[None, None, ...],
                vs.sequence: sequence_prime if is_priming else sequence,
                vs.bias: bias,
            },
        )

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not force and finish[0, 0] > 0.8:
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.0

    return phi_data, window_data, kappa_data, stroke_data, coords


# Main function to add handwritten text to ecg
def get_handwritten(
    link,
    num_words,
    input_file,
    output_dir,
    x_offset=0,
    y_offset=0,
    handwriting_size_factor=0.2,
    model_path=None,
    text=None,
    style=None,
    bias=1.0,
    force=False,
    animation=False,
    noinfo=True,
    save=None,
    bbox=False,
):
    if len(list((CACHE_DIR / "en_core_sci_sm").rglob("config.cfg"))) == 1:
        en_core_sci_sm_model_dir = str(list((CACHE_DIR / "en_core_sci_sm").rglob("config.cfg"))[0].parent)
        en_core_sci_sm_model = spacy.load(en_core_sci_sm_model_dir)
    else:
        en_core_sci_sm_model = None
    if en_core_sci_sm_model is None:
        warnings.warn("No spacy model named 'en_core_sci_sm', skipping handwritting text generation")
        if Path(input_file).parent.expanduser().resolve() != Path(output_dir).expanduser().resolve():
            shutil.copy(input_file, output_dir)
        outfile = os.path.join(output_dir, Path(input_file).name)
        return outfile

    if model_path is None:
        model_path = str((HANDWRITTEN_TEXT_DIR / "pretrained/model-29").resolve())

    # Use 'Agg' mode to prevent accumulation of figures
    matplotlib.use("Agg")
    filename = input_file
    nlp = en_core_sci_sm_model

    # Extract n medical terms
    if validators.url(link):
        # Parse URL
        r = requests.get(link)

        if platform == "darwin":
            soup = BeautifulSoup(r.content, "html5lib")
        else:
            soup = BeautifulSoup(r.content, "lxml")

        medicalText = ""
        for text in soup.body.find_all(string=True):
            if text.parent.name not in ["script", "meta", "link", "style"] and not isinstance(text, Comment) and text != "\n":
                medicalText = medicalText + text.strip()
        # Extract medical terms using space biomedical library
        doc = nlp(medicalText)
    else:
        # Extract medical terms from .txt files
        if link == "":
            link = HANDWRITTEN_TEXT_DIR / "Biomedical.txt"
        with open(link, "r") as f:
            # Extract lines from the file
            text = ""
            for line in f.readlines():
                text = text + " " + line
            # Extract medical terms using biomedical library
        doc = nlp(text)
        # Choose n random words from the extracted list
    words = random.choices(doc.ents, k=num_words)

    # Load the pretrained RNN model for handwritten text generation
    # with open(os.path.join(os.path.join("HandwrittenText", "data"), "translation.pkl"), "rb") as file:
    #     translation = pickle.load(file)
    translation = pickle.loads((HANDWRITTEN_TEXT_DIR / "data/translation.pkl").read_bytes())
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ""

    # Configure machine
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    # Create session
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(model_path + ".meta")
        saver.restore(sess, model_path)
        # Generate n handwritten words from the selected words
        numw = len(words)
        fig, ax = plt.subplots(numw, 1)
        i = 0

        # Iterate through the words and select style file
        for text in words:
            med_text = str(text)
            style = None
            if style is not None:
                style = None
                # with open(os.path.join(os.path.join("HandwrittenText", "data"), "styles.pkl"), "rb") as file:
                #     styles = pickle.load(file)
                styles = pickle.loads((HANDWRITTEN_TEXT_DIR / "data/styles.pkl").read_bytes())
                if style > len(styles[0]):
                    raise ValueError("Requested style is not in style list")
                style = [styles[0][style], styles[1][style]]
            phi_data, window_data, kappa_data, stroke_data, coords = sample_text(
                sess, med_text, translation, force, bias, style
            )
            # Plot strokes of handwritten text
            strokes = np.array(stroke_data)
            strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)

            # Generate subplots for each handwritten text
            for stroke in split_strokes(cumsum(np.array(coords))):
                ax[i].plot(stroke[:, 0], -stroke[:, 1])
            ax[i].set_aspect("equal")
            ax[i].set_axis_off()
            i = i + 1
        # Save the plot as HandwrittenText.png
        hw_text_file = CACHE_DIR / "hw_text" / f"HandwrittenText-{uuid.uuid4()}{save_img_ext}"
        fig.savefig(hw_text_file, dpi=1200)
        img_path = filename
        file_head, file_tail = os.path.splitext(filename)
        boxed_file = file_head + "-boxed" + file_tail

        # Load the ecg image
        img_ecg = Image.open(img_path)
        # Convert from RGBA to RGB
        img_ecg = img_ecg.convert("RGB")
        # Load the generated handwritten text image
        img_handwritten = Image.open(hw_text_file)
        # Convert the generated handwritten text image to RGB
        img_handwritten = img_handwritten.convert("RGB")
        # Resize the handwritten text image
        img_length = int(np.floor(img_ecg.size[0] * handwriting_size_factor))
        img_width = int(np.floor(img_ecg.size[1] * handwriting_size_factor))

        # Load handwritten text image into a numpy array
        img_handwritten = img_handwritten.resize((img_length, img_width))
        img_handwritten_arr = np.asarray(img_handwritten).copy()
        # Convert to a black and white image mask
        img_handwritten_arr[img_handwritten_arr[:, :, 1] != 255] = 0
        img_handwritten_arr[img_handwritten_arr == 255] = 1
        # Convert to an array
        img_handwritten_arr = np.asarray(img_handwritten_arr).copy()
        img_ecg_arr = np.asarray(img_ecg).copy()
        # Shift the handwritten text by specified offset
        img_cropped = (
            img_ecg_arr[
                x_offset : img_handwritten_arr.shape[0] + x_offset,
                y_offset : img_handwritten_arr.shape[1] + y_offset,
                : img_handwritten_arr.shape[2],
            ]
            * img_handwritten_arr
        )
        # Apply cropped image
        img_ecg_arr[
            x_offset : img_handwritten_arr.shape[0] + x_offset,
            y_offset : img_handwritten_arr.shape[1] + y_offset,
            : img_handwritten_arr.shape[2],
        ] = img_cropped
        # Save final image
        img_final = Image.fromarray(img_ecg_arr)
        head, tail = os.path.split(filename)
        img_final.save(os.path.join(output_dir, tail))

        # Load the ecg image
        plt.close("all")
        plt.close(fig)
        plt.clf()
        plt.cla()

        img_ecg.close()
        img_handwritten.close()

        os.remove(hw_text_file)
        outfile = os.path.join(output_dir, tail)

        del sess, saver, en_core_sci_sm_model, nlp

        # clear session and memory
        # try:
        #     tf.keras.backend.clear_session(free_memory=True)
        # except Exception:
        #     tf.keras.backend.clear_session()

        return outfile
