# -*- coding: utf-8 -*-
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import color, transform

base_path = 'C:/Users/rpmsp/Desktop/Doutorado Organizado/Disciplinas/[PDI] Fotos Fissuras Revestimento Ceramico/Teste 6/'

def num_images():
    return len(os.listdir(base_path))

def load_image(index, path=base_path, grayscale=True):
    files = sorted(os.listdir(path))
    image = plt.imread(path + files[index])
    
    if grayscale:
        image = to_grayscale(image)
    
    image = image - np.min(image)
    image = 255*image/np.max(image)

    return image.astype(np.uint8)

def to_grayscale(image):
    return 0.3*image[...,0]+0.59*image[...,1]+0.11*image[...,2]


def highlight(image, mask):
    col_img = color.gray2rgb(image)
    col_mask = color.gray2rgb(mask)
    return np.where(col_mask==255, [[[0, 255, 0]]], col_img)

def cut(img, shape):
    crop_height, crop_width = shape
    height, width = img.shape

    start_x = width//2 - crop_width//2
    start_y = height//2 - crop_height//2

    return img[
        start_y : start_y + crop_height,
        start_x : start_x + crop_width,
    ]

def periodic_lines(shape, angle, freq, phase, line_width):
    D = np.ceil(np.linalg.norm(shape)).astype(np.int)
    
    _, Y = np.meshgrid(range(D), range(D))
    
    period = abs(int(1/freq))
    gradients = np.mod(Y + phase, period)
    lines = np.where(gradients < line_width, 1, 0)
    rotated = transform.rotate(lines, -angle, preserve_range=True)
    return cut(rotated, shape)
    
def grid(shape, angles, freqs, phases, line_width):
    assert len(angles) == len(freqs) == len(phases)
    result = np.zeros(shape)
    for angle, freq, phase in zip(angles, freqs, phases):
        lines = periodic_lines(shape, angle, freq, phase, line_width)
        result = np.logical_or(result, lines)
    return np.where(result, 1, 0)


'''
def attempt1(img):
    threshold = filters.threshold_otsu(img)
    binary = np.where(img > threshold, 255, 0)
    dilated = segmentation.boundaries.dilation(binary)
    eroded = segmentation.boundaries.erosion(dilated)
    result = eroded - binary
    return result


'''