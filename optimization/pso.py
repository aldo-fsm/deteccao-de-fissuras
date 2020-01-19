import numpy as np

class PSO:
    def __init__(self, population, c1, c2, inertia):
        self.population = population
        self.velocity = np.zeros(population.shape)
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia

        self.pbest = population
        self.pbest_cost = [np.inf]*len(population)

        self.gbest = np.zeros(population.shape[1])
        self.gbest_cost = np.inf
        self.iteration = 0
        
    def minimize(self, cost_function):
        r1 = np.random.uniform(0, 1, size=self.population.shape)
        r2 = np.random.uniform(0, 1, size=self.population.shape)
        self.velocity = self.inertia*self.velocity                  \
                        + self.c1*r1*(self.pbest-self.population)   \
                        + self.c2*r2*(self.gbest-self.population)
        self.population = self.population + self.velocity#/np.linalg.norm(self.velocity)
        
        costs = [cost_function(x) for x in self.population]
        for i in range(len(costs)):
            if costs[i] < self.pbest_cost[i]:
                self.pbest_cost[i] = costs[i]
                self.pbest[i] = self.population[i]
        minimum = np.argmin(self.pbest_cost)
        self.gbest = self.pbest[minimum]
        self.gbest_cost = self.pbest_cost[minimum]
        self.iteration += 1
        print(self.iteration, self.gbest_cost, np.mean(np.std(self.population, axis=0)))