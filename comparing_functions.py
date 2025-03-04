import numpy as np
from dgamod import *
import statistics
import time
import pandas as pd

results = []

# -----------------------------------------------------------------
# Test the optimized and original versions of the fitness function
# -----------------------------------------------------------------

dimensions = np.arange(10, 100, 5)  # [2**n for n in range(3, 8)]
iterations = 1024

for n in dimensions:
    print(f"Testing for n = {n}")
    times_vectorized = []
    times_original = []
    times_gpu = []

    # generate propagators
    actions = actions_zhang(100, n)
    props = gen_props(actions, n, 0.15)

    # Measure time the vectorized version
    for iteration in range(iterations):
        action_sequence = np.random.randint(low=0, high=16, size=5 * n)

        start = time.time()
        fitness = reward_based_fitness_vectorized(
            action_sequence, props, tolerance=0.95, reward_decay=0.95
        )
        end = time.time()
        times_vectorized.append(end - start)

    mean_time_vec = statistics.mean(times_vectorized)
    std_time_vec = statistics.stdev(times_vectorized)

    results.append(
        {
            "chain_length": n,
            "fitness": "vectorized",
            "mean_time": mean_time_vec,
            "std": std_time_vec,
        }
    )

    # Measure time for the original version
    for iteration in range(iterations):
        action_sequence = np.random.randint(low=0, high=16, size=5 * n)

        start = time.time()
        fitness = reward_based_fitness(
            action_sequence, props, tolerance=0.95, reward_decay=0.95
        )
        end = time.time()
        times_original.append(end - start)

    mean_time_og = statistics.mean(times_original)
    std_time_og = statistics.stdev(times_original)

    results.append(
        {
            "chain_length": n,
            "fitness": "original",
            "mean_time": mean_time_og,
            "std": std_time_og,
        }
    )

    # Measure time for the GPU version (only after the first 3 iterations)
    # Performs batch calculation

    for iteration in range(10):
        action_sequences = np.random.randint(low=0, high=16, size=[iterations, 5 * n])

        T.cuda.synchronize()  # Ensure all GPU operations finish before timing

        start = time.time()
        fitness = reward_based_fitness_gpu(
            action_sequences, props, tolerance=0.95, reward_decay=0.95
        )
        T.cuda.synchronize()  # Ensure all GPU operations finish before timing

        end = time.time()
        if iteration > 3:
            times_gpu.append(end - start)

    mean_time_gpu = statistics.mean(times_gpu) / iterations
    std_time_gpu = statistics.stdev(times_gpu) / iterations
    results.append(
        {
            "chain_length": n,
            "fitness": "gpu",
            "mean_time": mean_time_gpu,
            "std": std_time_gpu,
        }
    )

df = pd.DataFrame(results)
output_file = f"comparing_functions_for_2n.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
