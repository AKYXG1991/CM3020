# If you on a Windows machine with any Python version 
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# the multi-threaded version does not work
# so instead, you can use this version. 

import unittest
import population
import simulation
import genome
import creature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=10, 
                                    gene_count=3)
        #sim = simulation.ThreadedSim(pool_size=1)
        sim = simulation.Simulation()
        records = []

        for iteration in range(1000):
            # this is a non-threaded version 
            # where we just call run_creature instead
            # of eval_population
            for cr in pop.creatures:
                sim.run_creature(cr, 2400)            
            #sim.eval_population(pop, 2400)
            fits = [cr.get_height_climbed()
                    for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) 
                    for cr in pop.creatures]
            max_fit = float(np.max(fits))
            mean_fit = float(np.mean(fits))
            mean_link = float(np.mean(links))
            max_link = float(np.max(links))
            print(iteration, "fittest:", np.round(max_fit, 3),
                  "mean:", np.round(mean_fit, 3), "mean links", np.round(mean_link), "max links", np.round(max_link))
            records.append({
                "iteration": iteration,
                "fittest": max_fit,
                "mean_fitness": mean_fit,
                "mean_links": mean_link,
                "max_links": max_link,
            })
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                # now we have the parents!
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            # elitism
            max_fit = np.max(fits)
            for cr in pop.creatures:
                if cr.get_height_climbed() == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    filename = "elite_"+str(iteration)+".csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break
            
            pop.creatures = new_creatures

        df = pd.DataFrame(records)
        df.to_csv("ga_results.csv", index=False)
        summary = df.describe()
        summary.to_csv("ga_summary.csv")

        plt.figure()
        plt.plot(df["iteration"], df["fittest"], label="fittest")
        plt.plot(df["iteration"], df["mean_fitness"], label="mean")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ga_fitness.png")
        plt.close()

        plt.figure()
        plt.plot(df["iteration"], df["mean_links"], label="mean links")
        plt.plot(df["iteration"], df["max_links"], label="max links")
        plt.xlabel("iteration")
        plt.ylabel("links")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ga_links.png")
        plt.close()

        self.assertNotEqual(fits[0], 0)

unittest.main()
