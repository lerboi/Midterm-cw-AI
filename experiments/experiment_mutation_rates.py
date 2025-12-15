import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np
import os

def save_results(filename, generation, max_fit, mean_fit, max_links, mean_links):
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # create file with headers doesnt exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write("generation,max_fitness,mean_fitness,max_links,mean_links\n")
    
    # Append data
    with open(filepath, 'a') as f:
        f.write(f"{generation},{max_fit},{mean_fit},{max_links},{mean_links}\n")

def run_mutation_experiments():
    mutation_configs = [
        (0.05, 0.1, 0.05),   # low mutation rates
        (0.1, 0.25, 0.1),    # medium (baseline)
        (0.25, 0.5, 0.25),   # high mutation rates
    ]
    
    for point_rate, shrink_rate, grow_rate in mutation_configs:
        print(f"\n{'='*60}")
        print(f"Testing Mutation Rates:")
        print(f"  Point: {point_rate}, Shrink: {shrink_rate}, Grow: {grow_rate}")
        print(f"{'='*60}\n")
        
        pop = population.Population(pop_size=10, gene_count=3)
        sim = simulation.Simulation()

        for iteration in range(500):
            for cr in pop.creatures:
                sim.run_creature(cr, 2400)            
            
            fits = [cr.get_hybrid_fitness() for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]
            
            max_fit = np.max(fits)
            mean_fit = np.mean(fits)
            max_links = np.max(links)
            mean_links = np.mean(links)
            
            print(f"Mutation(p={point_rate},s={shrink_rate},g={grow_rate}) - "
                  f"Gen {iteration}: max height: {np.round(max_fit, 3)}, "
                  f"mean: {np.round(mean_fit, 3)}, mean links: {np.round(mean_links)}")
            
            save_results(
                f"mutation_point_{point_rate}_shrink_{shrink_rate}_grow_{grow_rate}.csv",
                iteration,
                max_fit,
                mean_fit,
                max_links,
                mean_links
            )
            
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=point_rate, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
                dna = genome.Genome.grow_mutate(dna, rate=grow_rate)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            
            for cr in pop.creatures:
                if cr.get_hybrid_fitness() == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    
                    if iteration % 50 == 0:
                        os.makedirs("./elite_creatures", exist_ok=True)
                        filename = f"./elite_creatures/mutation_p{point_rate}_s{shrink_rate}_g{grow_rate}_gen_{iteration}.csv"
                        genome.Genome.to_csv(cr.dna, filename)
                    break
            
            pop.creatures = new_creatures
        
        print(f"\nCompleted mutation rates: point={point_rate}, shrink={shrink_rate}, grow={grow_rate}")
        print(f"Final max fitness: {max_fit}")
        print(f"Results saved to /results/mutation_point_{point_rate}_shrink_{shrink_rate}_grow_{grow_rate}.csv\n")

class TestGA(unittest.TestCase):
    def testMutationRates(self):
        run_mutation_experiments()

unittest.main()