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
        pop_sizes = [5, 10, 15]
        gene_counts = [1, 3, 5]

        all_records = []

        for pop_size in pop_sizes:
            for gene_count in gene_counts:
                pop = population.Population(pop_size=pop_size,
                                            gene_count=gene_count)
                sim = simulation.Simulation(use_gui=True,
                                            use_sandbox=True)
                records = []

                for iteration in range(1000):
                    for cr in pop.creatures:
                        sim.run_creature(cr, 2400)
                    fits = [cr.get_height_climbed() for cr in pop.creatures]
                    links = [len(cr.get_expanded_links())
                             for cr in pop.creatures]
                    max_fit = float(np.max(fits))
                    mean_fit = float(np.mean(fits))
                    mean_link = float(np.mean(links))
                    max_link = float(np.max(links))
                    print(pop_size, gene_count, iteration,
                          "fittest:", np.round(max_fit, 3),
                          "mean:", np.round(mean_fit, 3),
                          "mean links", np.round(mean_link),
                          "max links", np.round(max_link))
                    records.append({
                        "pop_size": pop_size,
                        "gene_count": gene_count,
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
                        dna = genome.Genome.crossover(p1.dna, p2.dna)
                        dna = genome.Genome.point_mutate(
                            dna, rate=0.1, amount=0.25)
                        dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                        dna = genome.Genome.grow_mutate(dna, rate=0.1)
                        cr = creature.Creature(1)
                        cr.update_dna(dna)
                        new_creatures.append(cr)
                    max_fit = np.max(fits)
                    for cr in pop.creatures:
                        if cr.get_height_climbed() == max_fit:
                            new_cr = creature.Creature(1)
                            new_cr.update_dna(cr.dna)
                            new_creatures[0] = new_cr
                            filename = "elite_" + str(pop_size) + "_" + \
                                       str(gene_count) + "_" + \
                                       str(iteration) + ".csv"
                            genome.Genome.to_csv(cr.dna, filename)
                            break

                    pop.creatures = new_creatures

                df = pd.DataFrame(records)
                df.to_csv(
                    f"ga_results_{pop_size}_{gene_count}.csv", index=False)
                summary = df.describe()
                summary.to_csv(
                    f"ga_summary_{pop_size}_{gene_count}.csv")

                plt.figure()
                plt.plot(df["fittest"], df["iteration"], label="fittest")
                plt.plot(df["mean_fitness"], df["iteration"], label="mean")
                plt.xlabel("fitness")
                plt.ylabel("iteration")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    f"ga_fitness_{pop_size}_{gene_count}.png")
                plt.close()

                all_records.extend(records)

        all_df = pd.DataFrame(all_records)
        for pop_size in pop_sizes:
            for gene_count in gene_counts:
                subset = all_df[(all_df["pop_size"] == pop_size) &
                                (all_df["gene_count"] == gene_count)]
                plt.plot(subset["fittest"], subset["iteration"],
                         label=f"p{pop_size}_g{gene_count}")
        plt.xlabel("fitness")
        plt.ylabel("iteration")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ga_fittest_all.png")
        plt.close()

        plt.figure()
        for pop_size in pop_sizes:
            for gene_count in gene_counts:
                subset = all_df[(all_df["pop_size"] == pop_size) &
                                (all_df["gene_count"] == gene_count)]
                plt.plot(subset["mean_fitness"], subset["iteration"],
                         label=f"p{pop_size}_g{gene_count}")
        plt.xlabel("fitness")
        plt.ylabel("iteration")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ga_mean_all.png")
        plt.close()

        self.assertTrue(len(all_records) > 0)


if __name__ == "__main__":
    unittest.main()
