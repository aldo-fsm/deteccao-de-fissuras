############################## Importar módulos ##############################

import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from skimage import morphology, exposure, filters, transform, util
from scipy import optimize, ndimage
import utils
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.cmap'] = 'gray'

for k in range(1,10):

    img = utils.load_image(k)
    print('\nImagem original: ')

    plt.figure()
    plt.imshow(img)
    plt.show()

    ################### Binzarização e esqueletização da imagem ###################
    
    def define_threshold(imagem):
        sobel_vertical = filters.sobel_v(imagem)
        sobel_horizontal = filters.sobel_h(imagem)
        h,c = imagem.shape
        return np.sum(np.abs(sobel_vertical + sobel_horizontal))/(h*c)

    def define_skeleton(imagem, block_size, offset):
        binary_adaptive = filters.threshold_adaptive(imagem, block_size, offset=offset)
        binary_adaptive = util.invert(binary_adaptive)
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

    limiar = define_threshold(img)
    print(limiar)
        
    if limiar > 0 and limiar < 0.05:
        offset = 10
    elif limiar > 0.05:
        offset = 25
    
    pimg = define_skeleton(img, 45, offset)

    if limiar > 0.05:
        pimg = filtragem(pimg)    
    
    plt.figure()
    plt.imshow(pimg)
    plt.show()

    ################# Identificar linhas na imagem esqueletizada #################

    lines = transform.probabilistic_hough_line(pimg, threshold=0, line_length=25, line_gap=1)
    print('\n - {} linhas detectadas na Transformada de Hough'.format(len(lines)))
    plt.imshow(pimg)
    for line in lines:
        plt.plot(*zip(*line), c='r')
 
    ################### Identificar orientações mais frequentes ###################
       
    directions = np.rad2deg(np.arctan([(y2-y1)/((x2-x1) + 0.001) for (x1,y1), (x2,y2) in lines]))
    directions = np.where(directions < 0, directions + 180, directions)
    hist = np.histogram(directions, range=[0,180], bins=180)
    sort_indexes = np.argsort(hist[0])
    hist = hist[0][sort_indexes], hist[1][sort_indexes]

    plt.figure()
    plt.bar(hist[1], hist[0])        
    
    a1, a2 = hist[1][-2:]
    print(" - Ângulos mais frequentes: "+str(a1)+", "+str(a2))
    
    ##### Não sei #####

    def plot_img_grid(image_grid):
        grid_height, grid_width = len(image_grid), len(image_grid[0])
        fig, axes = plt.subplots(grid_height, grid_width, squeeze=False, figsize=1.5*np.array(plt.rcParams['figure.figsize']))
        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                ax.imshow(image_grid[i][j])
        fig.tight_layout(pad=0)

    rot_pimg1 = transform.rotate(pimg, a1-90, resize=True)
    rot_pimg2 = transform.rotate(pimg, a2-90, resize=True)
    print("\nImagem esqueletizada rotacionada: ")

    plot_img_grid([[rot_pimg1, rot_pimg2]])
    plt.show()

    width = 1
    
    if limiar > 0.05:
        width = 3
    
    #kernel1 = np.pad(np.ones([rot_pimg1.shape[0], width]), 1, mode='constant', constant_values=0)
    #kernel2 = np.pad(np.ones([rot_pimg2.shape[0], width]), 1, mode='constant', constant_values=0)
    
    kernel1 = np.ones([rot_pimg1.shape[0], width])
    kernel2 = np.ones([rot_pimg2.shape[0], width])
    
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
    selem = np.array([[0,1,0],[1,1,1],[0,1,0]])
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
    plt.savefig('Teste_'+str(k)+'.jpg', cmap='gray')
    plt.show()