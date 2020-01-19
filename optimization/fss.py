import numpy as np

class FSS:
    def __init__(self, w_scale, stepInd_initial, stepInd_final):
        self.w_scale = w_scale
        self.stepInd_initial = stepInd_initial
        self.stepInd_final = stepInd_final
        self.stepVol_initial = 2*stepInd_initial
        self.stepVol_final = 2*stepInd_final
    
    def initSwarm(self, swarm_size, aquarium, iterations):
        self.iteration = 0
        self.iterations = iterations
        self.allow_bad_move_prob = 1
        self.stepInd = self.stepInd_initial
        self.stepVol = self.stepVol_initial
        self.swarm_size = swarm_size
        self.num_dim = len(aquarium)
        self.aquarium = aquarium
        low, high = zip(*aquarium)
        self.X = np.random.uniform(low, high, size=(swarm_size, self.num_dim))
        self.W = np.ones(swarm_size)
        self.fitness = np.repeat(np.nan, swarm_size)
        self.best = None
    
    def evaluateFitness(self, X, fitness_function):
        return np.array([fitness_function(x) for x in X])
    
    def indMovement(self, fitness_function):
        temp = self.X + np.random.uniform(-1, 1, size=self.X.shape)*self.stepInd
        fitness = self.evaluateFitness(temp, fitness_function)
        improved = self.fitnessVariation(fitness) > 0
        accept_move = np.logical_or(improved, np.random.uniform(0, 1, size=self.swarm_size) < self.allow_bad_move_prob)#*np.exp(-self.iteration))
        X = np.where(accept_move[..., None], temp, self.X)
        fitness = np.where(accept_move, fitness, self.fitness)
        delta_f = self.fitnessVariation(fitness)
        delta_X = temp - X
        self.X = X
        self.fitness = fitness
        if self.stepInd > self.stepInd_final:
            self.stepInd = self.stepInd - (self.stepInd_initial-self.stepInd_final)/self.iterations
        if self.allow_bad_move_prob > 0:
            self.allow_bad_move_prob = self.allow_bad_move_prob - 1/self.iterations
        return delta_X, delta_f
    
    def fitnessVariation(self, fitness):
        delta_f = fitness - self.fitness
        delta_f = np.where(np.isnan(delta_f), 1E-7, delta_f)
        return delta_f
                      
    def feeding(self, delta_f):
        max_delta_f = np.max(np.abs(delta_f))
        if max_delta_f != 0:
            W = self.W + delta_f/max_delta_f
            W = np.where(self.W > self.w_scale, self.w_scale, W)
        else:
            W = self.W
        delta_W = W - self.W
        self.W = W
        return delta_W
    
    def colInstMovement(self, delta_X, delta_f):
        I = np.average(delta_X, weights=delta_f+1E-7, axis=0)
        self.X = self.X + I
    
    def colVolitMovement(self, delta_W):
        sign = -int(np.sum(delta_W)>0)
        B = self.barycenter()
        DE = np.linalg.norm(self.X-B, axis=1)[...,None]
        U = np.random.uniform(0, 1, size=(self.X.shape[0], 1))
        self.X = self.X + sign * self.stepVol * U * (self.X-B)/DE
        if self.stepVol > self.stepVol_final:
            self.stepVol = self.stepVol - (self.stepVol_initial-self.stepVol_final)/self.iterations
    
    def barycenter(self):
        return np.average(self.X, weights=self.W, axis=0)
        
    def step(self, fitness_function):
        # Individual movement
        delta_X, delta_f = self.indMovement(fitness_function)
        
        # Feeding
        delta_W = self.feeding(delta_f)
        
        # Collective-instinctive movement
        self.colInstMovement(delta_X, delta_f)
        
        # Collective-volitive movement
        self.colVolitMovement(delta_W)
        self.fitness = self.evaluateFitness(self.X, fitness_function)
        
        # Best solution
        best_index = np.argmax(self.fitness)
        best_fitness = self.fitness[best_index]
        best_X = self.X[best_index]
        if self.best is None or best_fitness > self.best[0]:
            self.best = best_fitness, best_X
        
        self.iteration += 1