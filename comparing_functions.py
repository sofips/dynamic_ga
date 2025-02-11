import numpy as np
import scipy.linalg as la
import cmath as cm
import csv
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os
from dgamod import *
import statistics
#from numba import njit, prange
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


def reward_based_fitness_gpu(action_sequences, props, tolerance, reward_decay):
    device = 'cuda'
    
    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props, dtype=T.complex64, device=device, requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros((num_sequences, steps, chain_length), dtype=T.complex64, device=device)
    states[:, 0, 0] = 1.0  # Initial condition

    # Compute states using batched matrix multiplication
    for i in range(1, steps):
        states[:, i, :] = T.bmm(props[action_sequences[:, i]], states[:, i-1, :].unsqueeze(-1)).squeeze(-1)

    # Compute fidelity
    fid = states[:, :, -1].abs() ** 2  # Take absolute squared of last column

    # Compute rewards in parallel
    rewards = T.zeros_like(fid, device=device)
    rewards[fid <= 0.8] = 10 * fid[fid <= 0.8]
    
    mask = (fid > 0.8) & (fid <= 1 - tolerance)
    rewards[mask] = 100 / (1 + T.exp(10 * (1 - tolerance - fid[mask])))

    rewards[fid > 1 - tolerance] = 2500

    # Apply decay and sum fitness
    decay_factors = reward_decay ** T.arange(steps, device=device).unsqueeze(0)  # Shape: (1, steps)
    fitness = T.sum(rewards * decay_factors, dim=1)  # Sum over steps

    return fitness.cpu().numpy()  # Convert once at the end


results = []
# -----------------------------------------------------------------
# Test the optimized and original versions of the fitness function
# -----------------------------------------------------------------

dimensions = [16,32,64]

iterations = 1024

for n in dimensions:

    times_vectorized = []
    times_optimized = []
    times_original = []
    times_gpu = []

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

    for iteration in range(10):
        action_sequences = np.random.randint(low=0, high=16, size=[iterations,5*n])
        
        T.cuda.synchronize()  # Ensure all GPU operations finish before timing

        start = time.time()
        fitness = reward_based_fitness_gpu(
            action_sequences, props, tolerance=0.95, reward_decay=0.95
        )
        T.cuda.synchronize()  # Ensure all GPU operations finish before timing

        end = time.time()
        if iteration > 3:        
            times_gpu.append(end - start)

    mean_time_gpu = statistics.mean(times_gpu)/iterations
    std_time_gpu = statistics.stdev(times_gpu)/iterations  
    results.append(
        {
            "chain_length": n,
            "fitness": "gpu",
            "mean_time": mean_time_gpu,
            "std": std_time_gpu,
        }
    )

df = pd.DataFrame(results)
output_file = f"comparing_functions.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
