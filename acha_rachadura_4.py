###################### Código para identificar rachaduras ######################

import funcoes
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import probabilistic_hough_line

for i in range(1,11):

    im = plt.imread('Teste 4/'+str(i)+'.jpg')
    im = im[:,:,0]

    limiar = funcoes.define_threshold(im)
    print(limiar)
    
    if limiar > 0 and limiar < 0.05:
        offset = 10
    elif limiar > 0.05:
        offset = 25
    
    edges = funcoes.define_skeleton(im, 45, offset)
    
    edges = funcoes.filtragem(edges)
    
    nome = 'Teste 4/Esqueleto_'+str(i)+'.jpg'
    plt.imsave(nome, edges, cmap='gray')
    
    """plt.figure()
    lines = probabilistic_hough_line(edges, threshold=0, line_length=5, line_gap=5)

    for line in lines:
        p0, p1 = line
        angle = np.abs(np.round(180*np.arctan((p1[1]-p0[1])/((p1[0]-p0[0])+0.001))/np.pi))
        if angle > 88 or angle < 2:
            nada = []
        else:
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
    
    plt.imshow(im, cmap='gray')
    plt.savefig('Teste 4/Resultado_'+str(i)+'.jpg')
    plt.show()"""
    
    plt.figure()
    linhas = probabilistic_hough_line(edges, threshold=0, line_length=5, line_gap=5)
    orientacao1, orientacao2 = funcoes.histograma_de_direcoes_mod(linhas)
    if orientacao2 > 179:
        orientacao2-=180
    
    print(i)
    print(orientacao1)
    print(orientacao2)
    
    for line in linhas:
        p0, p1 = line
        angulo = np.arctan((p1[1]-p0[1])/((p1[0]-p0[0])+0.001))
        angulo = np.abs(np.round(180*angulo/np.pi))
        if i == 9:
            print(angulo)
        tolerancia = 5
        if (angulo > (orientacao1 - tolerancia) and  angulo < (orientacao1 + tolerancia)) or (angulo > (orientacao2 - tolerancia) and  angulo < (orientacao2 + tolerancia)):
            nada = []
        else:
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
        
    plt.imshow(im, cmap='gray')
    plt.savefig('Teste 4/Resultado_'+str(i)+'.jpg')
    plt.show()
    
