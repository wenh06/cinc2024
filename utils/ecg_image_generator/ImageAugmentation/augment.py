import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np

try:
    from imgaug import augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBoxesOnImage
except ImportError:
    iaa = None
    BoundingBoxesOnImage = None
from constants import lead_bounding_box_dir_name, text_bounding_box_dir_name
from helper_functions import read_bounding_box_txt, write_bounding_box_txt
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_directory", type=str, required=True)
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-o", "--output_directory", type=str, required=True)
    parser.add_argument("-r", "--rotate", type=int, default=25)
    parser.add_argument("-n", "--noise", type=int, default=25)
    parser.add_argument("-c", "--crop", type=float, default=0.01)
    parser.add_argument("-t", "--temperature", type=int, default=6500)
    return parser


# Main function for running augmentations
def get_augment(
    input_file, output_directory, rotate=25, noise=25, crop=0.01, temperature=6500, bbox=False, store_text_bounding_box=False
):
    if iaa is None:
        print("imgaug is not installed, or some libraries are missing. Skipping augmentation.")
        return os.path.splitext(os.path.split(input_file)[1])[0]
    filename = input_file
    image = Image.open(filename)

    image = np.array(image)

    lead_bbs = []
    leadNames_bbs = []

    if bbox:
        head, tail = os.path.split(filename)
        f, extn = os.path.splitext(tail)
        txt_file = os.path.join(head, lead_bounding_box_dir_name, f + ".txt")
        lead_bbs = read_bounding_box_txt(txt_file)
        lead_bbs = BoundingBoxesOnImage(lead_bbs, shape=image.shape)

    if store_text_bounding_box:

        head, tail = os.path.split(filename)
        f, extn = os.path.splitext(tail)
        txt_file = os.path.join(head, text_bounding_box_dir_name, f + ".txt")
        leadNames_bbs = read_bounding_box_txt(txt_file)
        leadNames_bbs = BoundingBoxesOnImage(leadNames_bbs, shape=image.shape)

    images = [image[:, :, :3]]
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    # Augment in a sequential manner. Create an augmentation object
    seq = iaa.Sequential(
        [
            iaa.Affine(rotate=rot),
            iaa.AdditiveGaussianNoise(scale=(noise, noise)),
            iaa.Crop(percent=crop_sample),
            iaa.ChangeColorTemperature(temperature),
        ]
    )

    seq_bbox = iaa.Sequential([iaa.Affine(rotate=-rot), iaa.Crop(percent=crop_sample)])

    images_aug = seq(images=images)

    if bbox:
        temp, augmented_lead_bbs = seq_bbox(images=images, bounding_boxes=lead_bbs)

    if store_text_bounding_box:
        temp, augmented_leadName_bbs = seq_bbox(images=images, bounding_boxes=leadNames_bbs)

    head, tail = os.path.split(filename)

    f = os.path.join(output_directory, tail)
    plt.imsave(fname=f, arr=images_aug[0])

    if bbox:
        head, tail = os.path.split(filename)
        f, extn = os.path.splitext(tail)
        txt_file = os.path.join(head, lead_bounding_box_dir_name, f + ".txt")
        write_bounding_box_txt(augmented_lead_bbs, txt_file)

    if store_text_bounding_box:
        head, tail = os.path.split(filename)
        f, extn = os.path.splitext(tail)
        txt_file = os.path.join(head, text_bounding_box_dir_name, f + ".txt")
        write_bounding_box_txt(augmented_leadName_bbs, txt_file)

    return f
