import numpy as np
import scipy.linalg as la
import cmath as cm
import csv
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os
from dgamod import *
import statistics
from numba import njit, prange
import time
import pandas as pd


def refined_cns(state, action_index, props):
    # Retrieve the matrix corresponding to the action index
    p = props[action_index]

    # Perform matrix-vector multiplication directly
    next_state = p @ state

    # Return the result as a flat 1D array
    return next_state.ravel()


def cns(state, action_index, props):
    state = np.transpose(np.mat(state))
    p = props[action_index]
    next_state = p * state
    next_state = np.asarray(np.transpose(next_state))
    next_state = np.squeeze(next_state)

    return next_state


def reward_based_fitness_original(
    action_sequence, props, tolerance, reward_decay, test_normalization=True
):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0

    for action in action_sequence:
        i += 1
        state = cns(state, action, props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 100 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 2500

        fitness = fitness + reward * (reward_decay**i)

    return fitness


def reward_based_fitness_optimized(
    action_sequence, props, tolerance, reward_decay, test_normalization=True
):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0

    for action in action_sequence:
        i += 1
        state = refined_cns(state, action, props)
        # Compute fidelity directly as a real value
        fid = (
            np.abs(state[n - 1]) ** 2
        )  # Equivalent to `real(state * conjugate(state))`

        # Optimize reward computation
        if fid <= 0.8:
            reward = 10 * fid
        elif fid <= 1 - tolerance:  # No need for double comparison `0.8 <= fid`
            reward = 100 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 2500

        # Update fitness with decay applied
        fitness += reward * (reward_decay**i)

    return fitness


def generate_states(initial_state, action_sequence, props):
    """Generate a matrix where each row is the state at a given step."""
    num_elements = len(initial_state)
    steps = len(action_sequence)
    states = np.zeros((steps + 1, num_elements), dtype=initial_state.dtype)
    states[0] = initial_state  # Set the initial state

    # Sequentially calculate states
    for i in range(1, steps):
        states[i] = refined_cns(states[i - 1], action_sequence[i], props)

    return states


def reward_based_fitness_vectorized(
    action_sequence, props, tolerance, reward_decay, test_normalization=True
):
    # compute states
    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0

    states = generate_states(state, action_sequence, props)

    fitness = calculate_reward(states, tolerance, reward_decay)

    return fitness


def calculate_reward(states, tolerance, reward_decay):
    # Compute fidelity for all states
    fid = np.abs(states[:, n - 1]) ** 2  # Shape: (num_states,)

    # Compute rewards based on conditions
    rewards = np.zeros_like(fid)
    rewards[fid <= 0.8] = 10 * fid[fid <= 0.8]
    rewards[(fid > 0.8) & (fid <= 1 - tolerance)] = 100 / (
        1 + np.exp(10 * (1 - tolerance - fid[(fid > 0.8) & (fid <= 1 - tolerance)]))
    )
    rewards[fid > 1 - tolerance] = 2500

    # Compute fitness with decay
    decay_factors = reward_decay ** np.arange(len(fid))  # Precompute decay factors
    fitness = np.sum(rewards * decay_factors)

    return fitness


results = []
# -----------------------------------------------------------------
# Test the optimized and original versions of the fitness function
# -----------------------------------------------------------------

dimensions = np.arange(8, 64, 8)

iterations = 1000

for n in dimensions:

    times_vectorized = []
    times_optimized = []
    times_original = []

    actions = actions_zhang(100, n)
    props = gen_props(actions, n, 0.15)

    for iteration in range(iterations):
        action_sequence = np.random.randint(low=0, high=16, size=5 * n)

        start = time.time()
        fitness = reward_based_fitness_optimized(
            action_sequence, props, tolerance=0.95, reward_decay=0.95
        )
        end = time.time()
        times_optimized.append(end - start)

    mean_time_opt = statistics.mean(times_optimized)
    std_time_opt = statistics.stdev(times_optimized)

    results.append(
        {
            "chain_length": n,
            "fitness": "optimized",
            "mean_time": mean_time_opt,
            "std": std_time_opt,
        }
    )

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

    for iteration in range(iterations):
        action_sequence = np.random.randint(low=0, high=16, size=5 * n)

        start = time.time()
        fitness = reward_based_fitness_original(
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

df = pd.DataFrame(results)
output_file = f"comparing_functions.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
