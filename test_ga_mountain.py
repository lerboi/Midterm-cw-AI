import unittest
import population
import simulation
import genome
import creature
import numpy as np

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        """
        SEEDED EVOLUTION GENETIC ALGORITHM

        This approach:
        1. SEEDS the initial population with a reasonable walker design
           (not random blobs that can't move)
        2. ALLOWS FULL EVOLUTION of both morphology AND motor control
        3. Creatures can evolve to find the optimal shape for climbing

        The fixed walker is just a STARTING POINT, not a constraint.
        Shape WILL change over generations as evolution finds better designs.
        """
        pop_size = 30

        # Create initial population SEEDED with walker design
        # This gives evolution a reasonable starting point
        creatures = []
        spec = genome.Genome.get_gene_spec()
        for i in range(pop_size):
            # Start with fixed walker body as SEED
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

            # Selection and reproduction with FULL EVOLUTION
            # Shape CAN and WILL change over generations
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []

            for i in range(pop_size):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = creatures[p1_ind]
                p2 = creatures[p2_ind]

                # FULL crossover - allows body structure to mix
                dna = genome.Genome.crossover(p1.dna, p2.dna)

                # FULL mutation - allows body shape to evolve
                dna = genome.Genome.point_mutate(dna, rate=0.25, amount=0.4)

                # Structural mutations - allows complexity to change
                dna = genome.Genome.shrink_mutate(dna, rate=0.05)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)

                # Create new creature with evolved DNA
                cr = creature.Creature(gene_count=2)
                cr.update_dna(dna)
                new_creatures.append(cr)

            # Elitism: keep the best creature unchanged
            max_fit = np.max(fits)
            for cr in creatures:
                if cr.get_climbing_fitness() == max_fit:
                    elite_cr = creature.Creature(gene_count=2)
                    elite_cr.update_dna(cr.dna)
                    new_creatures[0] = elite_cr

                    # Save elite creature
                    import os
                    os.makedirs("elite_creatures", exist_ok=True)
                    filename = f"elite_creatures/elite_gen_{iteration}.csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break

            creatures = new_creatures

        # Verify evolution produced non-zero fitness
        self.assertGreater(np.max(fits), 0)

unittest.main()
