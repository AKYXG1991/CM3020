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
import pybullet as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# mutation rates to test for point, shrink and grow mutations
MUTATION_RATES = [0.05, 0.1, 0.15, 0.2, 0.25]


class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop_size = 5
        gene_count = 3

        all_records = []

        for point_rate in MUTATION_RATES:
            for shrink_rate in MUTATION_RATES:
                for grow_rate in MUTATION_RATES:
                    pop = population.Population(pop_size=pop_size,
                                                gene_count=gene_count)
                    sim = simulation.Simulation(use_gui=True,
                                                use_sandbox=True)
                    records = []

                    # run a smaller number of iterations so the test completes more quickly
                    for iteration in range(20):
                        for cr in pop.creatures:
                            sim.run_creature(cr, 2400)
                        fits = [cr.get_height_climbed() for cr in pop.creatures]
                        links = [len(cr.get_expanded_links())
                                 for cr in pop.creatures]
                        max_fit = float(np.max(fits))
                        mean_fit = float(np.mean(fits))
                        mean_link = float(np.mean(links))
                        max_link = float(np.max(links))
                        print(point_rate, shrink_rate, grow_rate, iteration,
                              "fittest:", np.round(max_fit, 3),
                              "mean:", np.round(mean_fit, 3),
                              "mean links", np.round(mean_link),
                              "max links", np.round(max_link))
                        records.append({
                            "point_rate": point_rate,
                            "shrink_rate": shrink_rate,
                            "grow_rate": grow_rate,
                            "iteration": iteration,
                            "fittest": max_fit,
                            "mean_fitness": mean_fit,
                            "mean_links": mean_link,
                            "max_links": max_link,
                        })
                        fit_map = population.Population.get_fitness_map(fits)
                        new_creatures = []
                        for _ in range(len(pop.creatures)):
                            p1_ind = population.Population.select_parent(fit_map)
                            p2_ind = population.Population.select_parent(fit_map)
                            p1 = pop.creatures[p1_ind]
                            p2 = pop.creatures[p2_ind]
                            dna = genome.Genome.crossover(p1.dna, p2.dna)
                            dna = genome.Genome.point_mutate(
                                dna, rate=point_rate, amount=0.25)
                            dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
                            dna = genome.Genome.grow_mutate(dna, rate=grow_rate)
                            cr = creature.Creature(1)
                            cr.update_dna(dna)
                            new_creatures.append(cr)
                        max_fit = np.max(fits)
                        for cr in pop.creatures:
                            if cr.get_height_climbed() == max_fit:
                                new_cr = creature.Creature(1)
                                new_cr.update_dna(cr.dna)
                                new_creatures[0] = new_cr
                                filename = (
                                    f"elite_p{int(point_rate*100)}"
                                    f"_s{int(shrink_rate*100)}"
                                    f"_g{int(grow_rate*100)}_{iteration}.csv")
                                genome.Genome.to_csv(cr.dna, filename)
                                break

                        pop.creatures = new_creatures

                    df = pd.DataFrame(records)
                    fname = (
                        f"ga_results_p{int(point_rate*100)}"
                        f"_s{int(shrink_rate*100)}"
                        f"_g{int(grow_rate*100)}.csv")
                    df.to_csv(fname, index=False)
                    summary = df.describe()
                    sname = (
                        f"ga_summary_p{int(point_rate*100)}"
                        f"_s{int(shrink_rate*100)}"
                        f"_g{int(grow_rate*100)}.csv")
                    summary.to_csv(sname)

                    plt.figure()
                    plt.plot(df["iteration"], df["fittest"], label="fittest")
                    plt.plot(df["iteration"], df["mean_fitness"], label="mean")
                    plt.xlabel("iteration")
                    plt.ylabel("fitness")
                    plt.legend()
                    plt.tight_layout()
                    pname = (
                        f"ga_fitness_p{int(point_rate*100)}"
                        f"_s{int(shrink_rate*100)}"
                        f"_g{int(grow_rate*100)}.png")
                    plt.savefig(pname)
                    plt.close()

                    # close the simulation to ensure the next combination
                    # starts with a fresh physics instance
                    p.disconnect(sim.physicsClientId)

                    all_records.extend(records)

        all_df = pd.DataFrame(all_records)

        plt.figure()
        for point_rate in MUTATION_RATES:
            for shrink_rate in MUTATION_RATES:
                for grow_rate in MUTATION_RATES:
                    subset = all_df[(all_df["point_rate"] == point_rate) &
                                    (all_df["shrink_rate"] == shrink_rate) &
                                    (all_df["grow_rate"] == grow_rate)]
                    plt.plot(subset["iteration"], subset["fittest"],
                             label=(f"p{point_rate}_s{shrink_rate}_g{grow_rate}"))
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ga_fittest_all_rates.png")
        plt.close()

        plt.figure()
        for point_rate in MUTATION_RATES:
            for shrink_rate in MUTATION_RATES:
                for grow_rate in MUTATION_RATES:
                    subset = all_df[(all_df["point_rate"] == point_rate) &
                                    (all_df["shrink_rate"] == shrink_rate) &
                                    (all_df["grow_rate"] == grow_rate)]
                    plt.plot(subset["iteration"], subset["mean_fitness"],
                             label=(f"p{point_rate}_s{shrink_rate}_g{grow_rate}"))
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ga_mean_all_rates.png")
        plt.close()

        self.assertTrue(len(all_records) > 0)


if __name__ == "__main__":
    unittest.main()
