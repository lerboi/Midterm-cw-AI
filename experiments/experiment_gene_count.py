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
    
    # append data
    with open(filepath, 'a') as f:
        f.write(f"{generation},{max_fit},{mean_fit},{max_links},{mean_links}\n")

def run_gene_experiments():
    gene_counts = [2, 3, 5, 10]
    
    for gene_count in gene_counts:
        print(f"\n{'='*60}")
        print(f"Testing Gene Count: {gene_count}")
        print(f"{'='*60}\n")
        
        pop = population.Population(pop_size=10, gene_count=gene_count)
        sim = simulation.Simulation()

        for iteration in range(500):
            for cr in pop.creatures:
                sim.run_creature(cr, 2400)            
            
            fits = [cr.get_max_height() for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]
            
            max_fit = np.max(fits)
            mean_fit = np.mean(fits)
            max_links = np.max(links)
            mean_links = np.mean(links)
            
            print(f"Genes {gene_count} - Gen {iteration}: max height: {np.round(max_fit, 3)}, "
                  f"mean: {np.round(mean_fit, 3)}, mean links: {np.round(mean_links)}")
            
            save_results(
                f"gene_count_{gene_count}.csv",
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
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            
            for cr in pop.creatures:
                if cr.get_max_height() == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    
                    if iteration % 50 == 0:
                        os.makedirs("./elite_creatures", exist_ok=True)
                        filename = f"./elite_creatures/genes_{gene_count}_gen_{iteration}.csv"
                        genome.Genome.to_csv(cr.dna, filename)
                    break
            
            pop.creatures = new_creatures
        
        print(f"\nCompleted gene count {gene_count}")
        print(f"Final max fitness: {max_fit}")
        print(f"Results saved to /results/gene_count_{gene_count}.csv\n")

class TestGA(unittest.TestCase):
    def testGeneCounts(self):
        run_gene_experiments()

unittest.main()