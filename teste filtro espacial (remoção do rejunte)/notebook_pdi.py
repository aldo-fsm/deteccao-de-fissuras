import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from skimage import morphology, exposure, filters, transform
from scipy import optimize
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

def auxGrid(freq1, phase1, freq2, phase2):
    return utils.grid(img_fit.shape, [a1,a2], [freq1, freq2], [phase1, phase2], width)

def denormalize(x):
    min_freq = 0.0016521614890856475
    high = np.array([0.5, 500, 0.5, 500])
    low = np.array([min_freq, 0, min_freq, 0])
    return (high-low)*np.abs(x) + low
  
def fitness(x):
    x = denormalize(x)
    freq1, phase1, freq2, phase2 = x
    grid_img = auxGrid(freq1, phase1, freq2, phase2)
    return np.mean(img_fit*grid_img)

freq1, phase1, freq2, phase2 = 0.005076465058218435, 173.01246882793018, 0.004892084622274588, 191.37292817679557
grid = utils.grid(pimg.shape, [a1,a2], [freq1, freq2], [phase1,phase2], 3)
plt.imshow(grid+pimg)

optimizer = FSS(10, 0.1, 0.01)
aquarium = [[0, 1], [0, 1], [0, 1], [0, 1]]
optimizer.initSwarm(20, aquarium, 50)
fitness_values = []
best_fitness = []
for i in range(optimizer.iterations):
    optimizer.step(fitness)
    '''
    print('================ ' + str(i) + ' ================')
    print(optimizer.best)
    df = pd.DataFrame(index=['Weights', 'fitness'], data=np.concatenate([optimizer.W[None, ...], optimizer.fitness[None, ...]], axis=0))
    #display(df)
    #print(df)
    '''
    clear_output()
    print(i, optimizer.best)
    fitness_values.append(optimizer.fitness)
    best_fitness.append(optimizer.best[0])
for fv in np.array(fitness_values).T:
    plt.plot(fv, color='b')
plt.plot(best_fitness, color='g')

bestGrid = auxGrid(*denormalize(optimizer.best[1]))
plt.imshow(pimg+(1-bestGrid))

result = optimize.differential_evolution(lambda x : -fitness(x), bounds=aquarium)
aaa = auxGrid(*denormalize(result.x))
plt.imshow(pimg+(1-aaa))

plt.imshow(utils.grid(img_fit.shape, [a1,a2], [freq1, freq2], [phase1, phase2], width))

plt.imshow(auxGrid(*denormalize(optimizer.best[1])))

swarm_size = 20
population = np.random.randn(swarm_size, 4)*.5
pso = PSO(population, 1, 1, 0.8)

for i in range(50):
    pso.minimize(lambda x : -fitness(x))
aaa = auxGrid(*denormalize(pso.gbest))
plt.imshow(pimg+(1-aaa)) 