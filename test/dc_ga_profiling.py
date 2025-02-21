import os
import numpy as np
from dgamod import *
#from numba import njit, prange, jit
import csv
import pygad
import sys
import time
import configparser
from multiprocessing import Pool
from joblib import Parallel, delayed

# profiling libraries
import cProfile
import pstats
import time
from memory_profiler import memory_usage, profile
from statistics import mean, stdev
import pandas as pd


# Parallel processing, threads
num_threads = sys.argv[2]

os.environ["NUMBA_NUM_THREADS"] = num_threads  # Linux/macOS



# get parameters from config file
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, str(sys.argv[1]))
config = configparser.ConfigParser()
config.read(initfile)


# system parameters
n = config.getint("system_parameters", "n")
dt = config.getfloat("system_parameters", "dt")
b = config.getfloat("system_parameters", "b")

# speed_fraction = config.getfloat(
# "system_parameters", "speed_fraction"
# )  # fraction of qsl speed if loc based fitness
# max_optimization_time = config.getint("system_parameters", "max_optimization_time")

# generates actions and associated propagators
acciones = actions_zhang(b, n)  ## acciones zhang
props = gen_props(acciones, n, dt)

# genetic algorithm parameters
num_generations = config.getint("ga_initialization", "num_generations")
num_genes = config.getint("ga_initialization", "num_genes")
sol_per_pop = config.getint("ga_initialization", "sol_per_pop")
fidelity_tolerance = config.getfloat("ga_initialization", "fidelity_tolerance")
saturation = config.getint("ga_initialization", "saturation")
reward_decay = config.getfloat("ga_initialization", "reward_decay")

# crossover and parent selection
num_parents_mating = config.getint("parent_selection", "num_parents_mating")
parent_selection_type = config.get("parent_selection", "parent_selection_type")
keep_elitism = config.getint("parent_selection", "keep_elitism")
crossover_type = config.get("crossover", "crossover_type")
crossover_probability = config.getfloat("crossover", "crossover_probability")

# mutation
mutation_probability = config.getfloat("mutation", "mutation_probability")
mutation_num_genes = config.getint("mutation", "mutation_num_genes")

# saving data details
dirname = config.get("saving", "directory")
n_samples = 1 #config.getint("saving", "n_samples")
filename = dirname + "/nvsmaxfid.dat"


gene_space = np.arange(0, np.shape(acciones)[0], 1)
gene_type = int

stop_criteria = ["saturate_" + str(saturation)]  # , 'reach_'+str(fidelity_tolerance)]


# call construction functions
on_generation = generation_func_constructor(
    generation_func, [props, fidelity_tolerance]
)

fidelity_args = [
    props,
    fidelity_tolerance,
    reward_decay,
    False,
]  # [dt,props,speed_fraction, max_optimization_time]#,fidelity_tolerance,reward_decay]

fitness_func = fitness_func_constructor(reward_based_fitness_gpu, fidelity_args)

# def fitness_func(ga_instance, action_sequence, action_index) -> float:
#     return reward_based_fitness_gpu(action_sequence, props, fidelity_tolerance, reward_decay, False)


mutation_type = "swap"

# ----------------------------------------------------------


def target_program():
    # with open(filename, "a") as f:

    #for i in range(n_samples): no sampleo (no interesa maximizar)

    initial_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_elitism=keep_elitism,
        gene_space=gene_space,
        gene_type=gene_type,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        mutation_type=mutation_type,
        #on_generation=on_generation,
        mutation_num_genes=mutation_num_genes,
        stop_criteria=stop_criteria,
        save_solutions=False,
        fitness_batch_size=sol_per_pop,
        #parallel_processing=["thread", int(num_threads)]

    )
    with Pool(int(num_threads)) as p:
        initial_instance.run()


def profile_memory():
    """Profile memory usage of the target program."""
    mem_usage = memory_usage(target_program, interval=0.001, retval=False)
    return {
        "max_memory": max(mem_usage),
        "min_memory": min(mem_usage),
        "mean_memory": mean(mem_usage),
        "memory_increments": [mem_usage[i+1]-mem_usage[i] for i in range(len(mem_usage)-1)],
    }


def profile_tot_time():
    """Profile total time usage of the target program."""
    start_time = time.time()
    target_program()
    end_time = time.time()
    return {
        "total_time": end_time - start_time,
    }

def profile_by_function():
    
    profiler = cProfile.Profile()
    profiler.enable()
    target_program()
    profiler.disable()
    profiler.dump_stats(dirname + "/ppfunction_results.prof")

    with open(dirname + "/ppfunction_results.txt", "w") as f:
        stats = pstats.Stats(dirname + "/ppfunction_results.prof", stream=f)
        stats.sort_stats("tottime")
        stats.print_stats()
    
    return 'profile by function'


def main():
    num_runs = 50 # Number of profiling runs
    results = []

    # Run profiling multiple times with different parameters
    for i in range(num_runs):
        print(f'run {i}')
        time_stats = profile_tot_time()
        memory_stats = profile_memory()
        
        # Store the results with additional parameters in the list
        results.append({
            "chain_length": n,
            "pop_size": sol_per_pop,
            "total_time": time_stats["total_time"],
            "min_memory": memory_stats["min_memory"],
            "max_memory": memory_stats["max_memory"],
            "max_memory_increment": max(memory_stats["memory_increments"]),  # Record the max memory usage
        })

    # Convert results into a DataFrame for easier analysis and comparison
    df = pd.DataFrame(results)

    # Calculate mean and standard deviation for each parameter setting
    stats_summary = df.groupby("chain_length").agg(
        total_time_mean=("total_time", "mean"),
        total_time_std=("total_time", "std"),
        max_memory_increment_mean=("max_memory_increment", "mean"),
        max_memory_increment_std=("max_memory_increment", "std"),
        min_memory_mean=("min_memory", "mean"),
        min_memory_std=("min_memory", "std"),
        max_memory_mean=("max_memory", "mean"),
        max_memory_std=("max_memory", "std")
    ).reset_index()

    profile_by_function()

    # Print the summary of stats (means and standard deviations)
    print("\nSummary of Stats (Mean and Standard Deviation):")
    print(stats_summary)

    # Save the DataFrame to a CSV file for later retrieval
    output_file = f"{dirname}/profiling_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Optionally, you can also save the summary stats DataFrame to another file
    summary_file = f"{dirname}/profiling_summary.csv"
    stats_summary.to_csv(summary_file, index=False)
    print(f"Summary stats saved to {summary_file}")



if __name__ == "__main__":
    main()
