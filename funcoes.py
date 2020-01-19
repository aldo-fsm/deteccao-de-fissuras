################################### Funções ###################################

from skimage.filters import sobel_v, sobel_h, threshold_adaptive
from skimage.morphology import skeletonize
from skimage.util import invert
import numpy as np

def define_threshold(imagem):
    
    sobel_vertical = sobel_v(imagem)
    sobel_horizontal = sobel_h(imagem)
    
    h,c = imagem.shape
    return np.sum(np.abs(sobel_vertical + sobel_horizontal))/(h*c)

def define_skeleton(imagem, block_size, offset):

    binary_adaptive = threshold_adaptive(imagem, block_size, offset=offset)
    binary_adaptive = invert(binary_adaptive)
    binary_adaptive = skeletonize(binary_adaptive)
    
    return binary_adaptive

def histograma_de_direcoes(lines):
    listaDeAngulos = []
    for line in lines:
        p0, p1 = line
        angulo = np.arctan((p1[1]-p0[1])/((p1[0]-p0[0])+0.001))
        angulo = np.abs(np.round(180*angulo/np.pi))
        listaDeAngulos.append(angulo)    
    y, x = np.histogram(listaDeAngulos, bins=180, range=(0,180))
    yOrdenado = np.sort(y)
    valorMaximo1 = yOrdenado[-1]
    valorMaximo2 = yOrdenado[-2]
    yComoLista = y.tolist()
    angulo1 = yComoLista.index(valorMaximo1)
    angulo2 = yComoLista.index(valorMaximo2)    
    return angulo1, angulo2

def histograma_de_direcoes_mod(lines):
    maior_tamanho = 0
    guarda_linha = []
    for line in lines:
        p0, p1 = line
        tamanho = (p1[1]-p0[1])**2 + (p1[0]-p0[0])**2
        if tamanho > maior_tamanho:
            maior_tamanho = tamanho
            guarda_linha = line
    p0, p1 = guarda_linha        
    angulo = np.arctan((p1[1]-p0[1])/((p1[0]-p0[0])+0.001))
    angulo = np.abs(np.round(180*angulo/np.pi))
    return angulo, angulo+90

def filtragem(imagem):
    filtro = np.array([[1,1,1],[1,1,1],[1,1,1]])
    h,c = imagem.shape
    imagem_filtrada = imagem.copy()
    for i in range(1,h-1):
        for j in range(1,c-1):
            if np.sum(filtro*imagem[i-1:i+2,j-1:j+2]) != 3 or imagem[i,j] == 0:
                imagem_filtrada[i,j] = 0
    return imagem_filtrada   
            
        
    