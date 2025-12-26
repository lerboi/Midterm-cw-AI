import unittest
import population
import simulation
import genome
import creature
import numpy as np

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        """
        FIXED WALKER GENETIC ALGORITHM

        This implements the PDF-suggested approach:
        "start with a fixed design robot, and just evolve the motor control parameters"

        Key changes from random creature evolution:
        1. All creatures start with identical fixed walker body
        2. Only motor control genes (waveform, amplitude, frequency) are evolved
        3. No shrink/grow mutations (body structure is preserved)
        4. Movement-first fitness rewards ANY displacement

        This dramatically reduces the search space and creates a fitness gradient
        that evolution can follow.
        """
        pop_size = 30

        # Create initial population with FIXED WALKER bodies
        # All creatures have identical body structure, only motor control differs
        creatures = []
        spec = genome.Genome.get_gene_spec()
        for i in range(pop_size):
            cr = creature.Creature(gene_count=2, use_fixed_walker=True)
            creatures.append(cr)

        sim = simulation.Simulation()

        for iteration in range(1000):
            # Run simulation for each creature
            for cr in creatures:
                sim.run_creature(cr, 2400)

            # Calculate fitness using movement-first function
            fits = [cr.get_climbing_fitness() for cr in creatures]
            links = [len(cr.get_expanded_links()) for cr in creatures]

            print(f"{iteration} max fitness: {np.round(np.max(fits), 3)} "
                  f"mean: {np.round(np.mean(fits), 3)} "
                  f"links: {np.round(np.mean(links))}")

            # Selection and reproduction with CONTROL-ONLY operators
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []

            for i in range(pop_size):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = creatures[p1_ind]
                p2 = creatures[p2_ind]

                # CONTROL-ONLY crossover - preserves body structure
                dna = genome.Genome.crossover_control_only(p1.dna, p2.dna)

                # CONTROL-ONLY mutation - only mutates motor genes
                # Higher mutation rate since search space is smaller
                dna = genome.Genome.point_mutate_control_only(dna, rate=0.4, amount=0.3)

                # NO shrink/grow mutations - body structure is fixed

                # Create new creature with mutated DNA
                cr = creature.Creature(gene_count=2, use_fixed_walker=True)
                cr.update_dna(dna)
                new_creatures.append(cr)

            # Elitism: keep the best creature
            max_fit = np.max(fits)
            for cr in creatures:
                if cr.get_climbing_fitness() == max_fit:
                    elite_cr = creature.Creature(gene_count=2, use_fixed_walker=True)
                    elite_cr.update_dna(cr.dna)
                    new_creatures[0] = elite_cr

                    # Save elite creature
                    import os
                    os.makedirs("elite_creatures", exist_ok=True)
                    filename = f"elite_creatures/elite_walker_{iteration}.csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break

            creatures = new_creatures

        # Verify evolution produced non-zero fitness
        self.assertGreater(np.max(fits), 0)

unittest.main()
