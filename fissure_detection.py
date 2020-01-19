
import numpy as np
import pandas as pd
import skimage
from skimage import filters, morphology, transform, exposure
from IPython.display import display, clear_output
import scipy as sp
from matplotlib import pyplot as plt
import utils

def detect_fissures(img):
    pimg = np.array(img)
    limiar = utils.define_threshold(img)
    if limiar > 0 and limiar < 0.05:
        offset = 10
    elif limiar > 0.05:
        offset = 25

    pimg = utils.define_skeleton(img, 45, offset)
    if limiar > 0.05:
        pimg = utils.filtragem(pimg)
    runs = 30
    lines = []
    for _ in range(runs):
        lines.extend(transform.probabilistic_hough_line(pimg))

    directions = np.array([utils.angle(point1, point2) for point1, point2 in lines])
    directions = np.where(directions < 0, directions + 180, directions)
    hist = np.histogram(directions, range=[0,180], bins=180)
    sort_indexes = np.argsort(hist[0])
    hist = hist[0][sort_indexes], hist[1][sort_indexes]

    a1, a2 = hist[1][-2:]

    rot_pimg1 = skimage.transform.rotate(pimg, a1-90, resize=True)
    rot_pimg2 = skimage.transform.rotate(pimg, a2-90, resize=True)

    width = 3
    kernel1 = np.pad(np.ones([rot_pimg1.shape[0], width]), 1, mode='constant', constant_values=-1)
    kernel2 = np.pad(np.ones([rot_pimg2.shape[0], width]), 1, mode='constant', constant_values=-1)
    corr1 = sp.ndimage.correlate(rot_pimg1, kernel1, mode='constant')
    corr2 = sp.ndimage.correlate(rot_pimg2, kernel2, mode='constant')

    corr_rot1 = utils.cut(skimage.transform.rotate(corr1, 90-a1, resize=True), pimg.shape)
    corr_rot2 = utils.cut(skimage.transform.rotate(corr2, 90-a2, resize=True), pimg.shape)
    
    thresholds = [filters.threshold_isodata, filters.threshold_li, filters.threshold_mean,
                filters.threshold_minimum, filters.threshold_otsu, filters.threshold_triangle,
                filters.threshold_yen]

    best_fitness = -np.inf
    best_mask = None
    for thr in thresholds:
        binary1 = np.where(corr_rot1 > thr(corr1), 1, 0)
        binary2 = np.where(corr_rot2 > thr(corr2), 1, 0)
        selem = np.ones((5,5))
        binary_dilated1 = skimage.morphology.dilation(binary1, selem=selem)
        binary_dilated2 = skimage.morphology.dilation(binary2, selem=selem)
        mask = np.logical_or(binary_dilated1, binary_dilated2)
        fitness_value = fitness(mask, pimg)
        if fitness_value > best_fitness:
            best_fitness = fitness_value
            best_mask = mask
    mask = best_mask

    rachaduras = (1 - mask)*pimg
    # rachaduras_rgba = np.where(rachaduras[..., np.newaxis] == 1, [255,0,0,255], [0,0,0,0])
    return rachaduras


def plot_img_grid(image_grid):
    grid_height, grid_width = len(image_grid), len(image_grid[0])
    fig, axes = plt.subplots(grid_height, grid_width, squeeze=False, figsize=1.5*np.array(plt.rcParams['figure.figsize']))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(image_grid[i][j])
    fig.tight_layout(pad=0)


def fitness(mask, binary_image):
    binary_image = 2*binary_image - 1
    return np.mean(mask*binary_image)
