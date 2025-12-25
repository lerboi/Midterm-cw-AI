# Multi-threaded GA for mountain climbing
# If you on a Windows machine with any Python version
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# this multi-threaded version does not work
# please use test_ga_mountain.py (single-threaded) on those setups

import unittest
import population
import simulation
import genome
import creature
import numpy as np
import os

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=10,
                                    gene_count=3)
        sim = simulation.ThreadedSim(pool_size=1)

        for iteration in range(1000):
            sim.eval_population(pop, 2400)
            # Use climbing fitness (final position, ground contact, progress)
            fits = [cr.get_climbing_fitness()
                    for cr in pop.creatures]
            links = [len(cr.get_expanded_links())
                    for cr in pop.creatures]
            print(iteration, "max fitness:", np.round(np.max(fits), 3),
                  "mean:", np.round(np.mean(fits), 3), "mean links", np.round(np.mean(links)), "max links", np.round(np.max(links)))
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                # crossover and mutation
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            # elitism - keep the best creature
            max_fit = np.max(fits)
            for cr in pop.creatures:
                if cr.get_climbing_fitness() == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    os.makedirs("elite_creatures", exist_ok=True)
                    filename = "elite_creatures/elite_mountain_"+str(iteration)+".csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break

            pop.creatures = new_creatures

        self.assertNotEqual(fits[0], 0)

unittest.main()
