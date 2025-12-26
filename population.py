import creature 
import numpy as np

class Population:
    def __init__(self, pop_size, gene_count):
        self.creatures = [creature.Creature(
                          gene_count=gene_count) 
                          for i in range(pop_size)]

    @staticmethod
    def get_fitness_map(fits, min_fitness=0.1):
        """
        Create cumulative fitness map for roulette wheel selection.

        Args:
            fits: List of fitness values
            min_fitness: Minimum fitness floor to ensure selection pressure
                        when all creatures have near-zero fitness

        Returns:
            Cumulative fitness map for selection
        """
        fitmap = []
        total = 0
        for f in fits:
            # Add minimum fitness floor to prevent random selection
            # when all creatures have ~0 fitness
            adjusted_f = f + min_fitness
            total = total + adjusted_f
            fitmap.append(total)
        return fitmap
    
    @staticmethod
    def select_parent(fitmap):
        r = np.random.rand() # 0-1
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i

