import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from skimage import morphology, exposure, filters, transform
from scipy import optimize, ndimage
from optimization.fss import FSS
from optimization.pso import PSO
import utils
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.cmap'] = 'gray'

img = utils.load_image(9)
print('Imagem original [1]')
plt.imshow(img)
plt.show()
pimg = img
pimg = exposure.adjust_sigmoid(pimg, gain=100)

print("\nAlto contraste [1.1]")
plt.imshow(pimg)
plt.show()
#pimg = filters.sobel(pimg)
thr = filters.thresholding.threshold_otsu(pimg)
pimg = pimg>thr

print("\nOtsu [1.2]")
plt.imshow(pimg)
plt.show()
num_white = np.sum(pimg)
num_black = np.prod(pimg.shape) - num_white
if num_white > num_black:
    pimg = 1 - pimg
    print("\nInversão para que as rachaduras sejam brancas e o fundo preto [1.3]")
    plt.imshow(pimg)
    plt.show()
    
print("\nEsqueletização [1.4]")
pimg = morphology.skeletonize(pimg)
plt.imshow(pimg)

lines = transform.probabilistic_hough_line(pimg)
print('\n - {} linhas detectadas na Transformada de Hough'.format(len(lines)))
plt.imshow(pimg)
for line in lines:
        plt.plot(*zip(*line), c='r')
        
directions = np.rad2deg(np.arctan([(y2-y1)/((x2-x1) + 0.001) for (x1,y1), (x2,y2) in lines]))
directions = np.where(directions < 0, directions + 180, directions)
hist = np.histogram(directions, range=[0,180], bins=180)
sort_indexes = np.argsort(hist[0])
hist = hist[0][sort_indexes], hist[1][sort_indexes]

plt.figure()
plt.bar(hist[1], hist[0])        

a1, a2 = hist[1][-2:]
print(" - Ângulos mais frequentes: "+str(a1)+", "+str(a2))

print('Exemplo de grade com frequência e fase especificada')
plt.imshow(utils.grid(pimg.shape, [a1,a2], [0.02, 0.01], [1,3], 3))
plt.show()

width = 5
img_fit = np.where(pimg == 1, width, -1/width)

def plot_img_grid(image_grid):
    grid_height, grid_width = len(image_grid), len(image_grid[0])
    fig, axes = plt.subplots(grid_height, grid_width, squeeze=False, figsize=1.5*np.array(plt.rcParams['figure.figsize']))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(image_grid[i][j])
    fig.tight_layout(pad=0)

print("Imagem esqueletizada")
plt.imshow(pimg)
plt.show()
rot_pimg1 = transform.rotate(pimg, a1-90, resize=True)
rot_pimg2 = transform.rotate(pimg, a2-90, resize=True)
print("Imagem esqueletizada rotacionada")
plot_img_grid([[rot_pimg1, rot_pimg2]])
plt.show()
width = 3
kernel1 = np.pad(np.ones([rot_pimg1.shape[0], width]), 1, mode='constant', constant_values=-1)
kernel2 = np.pad(np.ones([rot_pimg2.shape[0], width]), 1, mode='constant', constant_values=-1)
corr1 = ndimage.correlate(rot_pimg1, kernel1, mode='constant')
corr2 = ndimage.correlate(rot_pimg2, kernel2, mode='constant')
print("Aplicação do filtro")
plot_img_grid([[corr1, corr2]])
plt.show()
print("Imagem filtrada rotacionada a posição original")
corr_rot1 = utils.cut(transform.rotate(corr1, 90-a1, resize=True), pimg.shape)
corr_rot2 = utils.cut(transform.rotate(corr2, 90-a2, resize=True), pimg.shape)
plot_img_grid([[corr_rot1, corr_rot2]])
plt.show()

binary1 = np.where(corr_rot1 > filters.threshold_yen(corr1), 1, 0)
binary2 = np.where(corr_rot2 > filters.threshold_yen(corr2), 1, 0)
print('Imagem filtrada binarizada')
plot_img_grid([[binary1, binary2]])
plt.show()
print('Imagem filtrada binarizada dilatada')
selem = np.ones((10,10))
binary_dilated1 = morphology.dilation(binary1, selem=selem)
binary_dilated2 = morphology.dilation(binary2, selem=selem)
plot_img_grid([[binary_dilated1, binary_dilated2]])
plt.show()
mask = np.logical_or(binary_dilated1, binary_dilated2)
print('Mascara')
plt.imshow(mask)
plt.show()

print('Original => Rachaduras')
rachaduras = (1 - mask)*pimg
plot_img_grid([[img, rachaduras]])
plt.savefig('teste.jpg', cmap='gray')
plt.show()