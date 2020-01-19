# -*- coding: utf-8 -*-
import os
from matplotlib import pyplot as plt
import numpy as np
import skimage
from skimage import color, transform, exposure, filters, morphology, segmentation
import scipy as sp
from scipy import ndimage
import os
import errno

#base_path = './Fotos Fissuras Revestimento Ceramico/images/'
base_path = './Teste 6/'

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

def angle(point1, point2):
    z1 = np.complex(*point1)
    z2 = np.complex(*point2)
    return np.angle(z2-z1, deg=True)

def invert(img):
    return 1 - img

def define_threshold(imagem):
    sobel_vertical = filters.sobel_v(imagem)
    sobel_horizontal = filters.sobel_h(imagem)
    h,c = imagem.shape
    return np.sum(np.abs(sobel_vertical + sobel_horizontal))/(h*c)

def define_skeleton(imagem, block_size, offset):
    binary_adaptive = imagem > filters.threshold_local(imagem, block_size, offset=offset)
    binary_adaptive = invert(binary_adaptive)
    binary_adaptive = morphology.skeletonize(binary_adaptive)
    return binary_adaptive

def filtragem(imagem):
    filtro = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    h,c = imagem.shape
    imagem_filtrada = imagem.copy()
    for i in range(2,h-2):
        for j in range(2,c-2):
            if np.sum(filtro*imagem[i-2:i+3,j-2:j+3]) != 5 or imagem[i,j] == 0:
                imagem_filtrada[i,j] = 0
    return imagem_filtrada

def plot_img_grid(image_grid):
    grid_height, grid_width = len(image_grid), len(image_grid[0])
    fig, axes = plt.subplots(grid_height, grid_width, squeeze=False, figsize=1.5*np.array(plt.rcParams['figure.figsize']))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(image_grid[i][j])
    fig.tight_layout(pad=0)

def create_parent_directories(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def savefig(name, path):
    plt.savefig(path+name+'.png', format='png')
    plt.show()

def attempt3(img_index):
    img = load_image(img_index)
    path = './outputs/'+str(img_index)+'/'
    create_parent_directories(path)
    
    plt.imshow(img)
    savefig('1-Imagem original', path)
    
    pimg = np.array(img)

    limiar = define_threshold(img)
    print(limiar)

    if limiar > 0 and limiar < 0.05:
        offset = 10
    elif limiar > 0.05:
        offset = 25

    pimg = define_skeleton(img, 45, offset)
    plt.imshow(pimg)
    savefig('2-esqueleto1', path)
    if limiar > 0.05:
        pimg = filtragem(pimg)    

    plt.imshow(pimg)
    savefig('3-esqueleto1', path)

    runs = 30
    lines = []
    for _ in range(runs):
        lines.extend(transform.probabilistic_hough_line(pimg))
    print('Média de {} linhas detectadas em {} execuções'.format(len(lines)/runs, runs))
    plt.imshow(pimg)
    for line in lines:
            plt.plot(*zip(*line), c='r')
    savefig('4-detecção de retas', path)

    directions = np.array([angle(point1, point2) for point1, point2 in lines])
    directions = np.where(directions < 0, directions + 180, directions)
    hist = np.histogram(directions, range=[0,180], bins=180)
    sort_indexes = np.argsort(hist[0])
    hist = hist[0][sort_indexes], hist[1][sort_indexes]
    plt.bar(hist[1], hist[0])
    savefig('5-histograma de direções', path)

    a1, a2 = hist[1][-2:]
    print(a1, a2)
    rot_pimg1 = transform.rotate(pimg, a1-90, resize=True)
    rot_pimg2 = transform.rotate(pimg, a2-90, resize=True)

    plot_img_grid([[rot_pimg1, rot_pimg2]])
    savefig("6-Imagem esqueletizada rotacionada", path)
    width = 3
    kernel1 = np.pad(np.ones([rot_pimg1.shape[0], width]), 1, mode='constant', constant_values=-1)
    kernel2 = np.pad(np.ones([rot_pimg2.shape[0], width]), 1, mode='constant', constant_values=-1)
    corr1 = ndimage.correlate(rot_pimg1, kernel1, mode='constant')
    corr2 = ndimage.correlate(rot_pimg2, kernel2, mode='constant')
    
    plot_img_grid([[corr1, corr2]])
    savefig("7-Aplicação do filtro", path)
    
    corr_rot1 = cut(transform.rotate(corr1, 90-a1, resize=True), pimg.shape)
    corr_rot2 = cut(transform.rotate(corr2, 90-a2, resize=True), pimg.shape)
    plot_img_grid([[corr_rot1, corr_rot2]])
    savefig("8-Imagem filtrada rotacionada a posição original", path)
    
    def fitness(mask, binary_image):
        binary_image = 2*binary_image - 1
        return np.mean(mask*binary_image)

    thresholds = [filters.threshold_isodata, filters.threshold_li, filters.threshold_mean,
              filters.threshold_minimum, filters.threshold_otsu, filters.threshold_triangle,
              filters.threshold_yen]
    best_fitness = -np.inf
    best_mask = None
    for thr in thresholds:
        binary1 = np.where(corr_rot1 > thr(corr1), 1, 0)
        binary2 = np.where(corr_rot2 > thr(corr2), 1, 0)
        selem = np.ones((5,5))
        binary_dilated1 = morphology.dilation(binary1, selem=selem)
        binary_dilated2 = morphology.dilation(binary2, selem=selem)
        mask = np.logical_or(binary_dilated1, binary_dilated2)
        fitness_value = fitness(mask, pimg)
        if fitness_value > best_fitness:
            best_fitness = fitness_value
            best_mask = mask
    mask = best_mask
    plt.imshow(mask)
    savefig("9-Mascara", path)
    cor_rachadura = [255,0,0]
    rachaduras = (1 - mask)*pimg
    rachaduras_dilatadas = morphology.dilation(rachaduras)
    rachaduras_rgba = np.where(rachaduras[..., np.newaxis] == 1, cor_rachadura+[255], [0,0,0,0])
    rachaduras_rgba_dilatadas = np.where(rachaduras_dilatadas[..., np.newaxis] == 1, cor_rachadura+[255], [0,0,0,0])
    
    plt.imshow(img)
    plt.imshow(rachaduras_rgba)
    savefig("10-rachaduras detectadas", path)
    
    plt.imshow(img)
    plt.imshow(rachaduras_rgba_dilatadas)
    savefig("11-rachaduras detectadas (dilatadas)", path)
    
    plt.imshow(rachaduras)
    savefig("12-apenas rachaduras", path)
    
    return rachaduras

def attempt2(img):
    pimg = img
    pimg = exposure.adjust_sigmoid(pimg, gain=100)
    thr = filters.thresholding.threshold_otsu(pimg)
    pimg = pimg>thr
    num_white = np.sum(pimg)
    num_black = np.prod(pimg.shape) - num_white
    if num_white > num_black:
        pimg = 1 - pimg
    pimg = morphology.skeletonize(pimg)

    lines = transform.probabilistic_hough_line(pimg)

    directions = np.array([angle(point1, point2) for point1, point2 in lines])
    directions = np.where(directions < 0, directions + 180, directions)

    hist = np.histogram(directions, range=[0,180], bins=180)
    sort_indexes = np.argsort(hist[0])
    hist = hist[0][sort_indexes], hist[1][sort_indexes]

    a1, a2 = hist[1][-2:]

    rot_pimg1 = transform.rotate(pimg, a1-90, resize=True)
    rot_pimg2 = transform.rotate(pimg, a2-90, resize=True)
    width = 3
    kernel1 = np.pad(np.ones([rot_pimg1.shape[0], width]), 1, mode='constant', constant_values=-1)
    kernel2 = np.pad(np.ones([rot_pimg2.shape[0], width]), 1, mode='constant', constant_values=-1)
    corr1 = ndimage.correlate(rot_pimg1, kernel1, mode='constant')
    corr2 = ndimage.correlate(rot_pimg2, kernel2, mode='constant')
    corr_rot1 = cut(transform.rotate(corr1, 90-a1, resize=True), pimg.shape)
    corr_rot2 = cut(transform.rotate(corr2, 90-a2, resize=True), pimg.shape)

    binary1 = np.where(corr_rot1 > filters.threshold_yen(corr1), 1, 0)
    binary2 = np.where(corr_rot2 > filters.threshold_yen(corr2), 1, 0)
    selem = np.ones((10,10))
    binary_dilated1 = morphology.dilation(binary1, selem=selem)
    binary_dilated2 = morphology.dilation(binary2, selem=selem)

    mask = np.logical_or(binary_dilated1, binary_dilated2)
    return (1 - mask)*pimg

def attempt1(img):
    threshold = filters.threshold_otsu(img)
    binary = np.where(img > threshold, 255, 0)
    dilated = segmentation.boundaries.dilation(binary)
    eroded = segmentation.boundaries.erosion(dilated)
    result = eroded - binary
    return result
