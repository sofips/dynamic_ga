import numpy as np
import scipy.linalg as la
import csv

# import matplotlib.pyplot as plt
from scipy.linalg import expm
import os

# from numba import njit
import torch as T

np.complex_ = np.complex128
np.mat = np.asmatrix


def gen_props(actions, n, dt, test=True):
    """
    Generate propagators for a set of action matrices.

    Parameters:
    actions (numpy.ndarray): A 3D array of shape (n_actions, n, n) containing the action matrices.
    n (int): The dimension of the action matrices equal to the chain length.
    dt (float): The time step for propagation.
    test (bool, optional): If True, perform a test to check the correctness of the propagation. Default is True.

    Returns:
    numpy.ndarray: A 3D array of shape (n_actions, n, n) containing the propagators.

    Notes:
    - The function first diagonalizes each action matrix to obtain eigenvalues and eigenvectors.
    - It then constructs the propagators using the matrix exponential.
    - If `test` is True, it checks if the eigenstates are properly propagated and prints the result.
    """

    n_actions = actions.shape[0]
    comp_i = complex(0, 1)
    props = np.zeros((n_actions, n, n), dtype=np.complex_)

    for i in range(0, n_actions):  # propagator building
        props[i, :, :] = expm(-1j * actions[i] * dt)

    if test:

        en = np.zeros((n_actions, n), dtype=np.complex_)
        bases = np.zeros((n_actions, n, n), dtype=np.complex_)

        for j in range(0, n_actions):  # diagonalization of action matrices
            en[j, :], bases[j, :, :] = la.eig(actions[j, :, :])

        correct_propagation = True

        for a in np.arange(0, n_actions):
            for j in np.arange(0, n):
                errores = (
                    calculate_next_state(
                        bases[a, :, j], a, props, check_normalization=True
                    )
                    - np.exp(-comp_i * dt * en[a, j]) * bases[a, :, j]
                )
                et = np.sum(errores)
                if la.norm(et) > 1e-8:
                    print(
                        "Propagation Error: Eigenstates are not being properly propagated"
                    )
                    correct_propagation = False
                    quit()

        if correct_propagation:
            print("Eigenstate Propagation: correct")
    return props


def fidelity(action_sequence, props, return_time=False, test_normalization=True):
    """
    Calculate the fidelity resulting of a given pulse sequence. The state is initialized to /10...0>

    Parameters:
    action_sequence (list or array-like): A sequence of actions to be applied to the initial state.
    props (ndarray): A 3D array where props[action] is the propagation matrix corresponding to that action.
    return_time (bool, optional): If True, return the time step at which the maximum fidelity is achieved. Default is False.
    test_normalization (bool, optional): If True, test the normalization of the final state. Default is True.

    Returns:
    float: The maximum fidelity achieved.
    tuple: If return_time is True, returns a tuple (max_fid, imax) where max_fid is the maximum fidelity and imax is the time step at which it is achieved.
    """

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    max_fid = 0.0
    imax = 0
    i = 0

    for action in action_sequence:

        i += 1
        state = np.matmul(props[action, :, :], state)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid > max_fid:
            imax = i
            max_fid = fid

    if test_normalization:

        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization test failed. Norm of final state: ", la.norm(state))
            quit()

    if return_time:
        return max_fid, imax

    return max_fid


def calculate_next_state(state, action_index, props, check_normalization=True):
    """
    Calculate the next state by applying the propagator associated to an action.

    Args:
        state (np.ndarray): The current state represented as a numpy array.
        action_index (int): The index of the action to be applied.
        props (list or np.ndarray): The propagator corresponding to that action.

    Returns:
        np.ndarray: The next state after applying the action.

    Raises:
        SystemExit: If the normalization of the next state fails.
    """

    state = np.transpose(np.mat(state))
    p = props[action_index]
    next_state = p * state
    next_state = np.asarray(np.transpose(next_state))
    next_state = np.squeeze(next_state)

    if check_normalization:
        if abs(la.norm(next_state) - 1.0) > 1e-8:
            print("Normalization failed. Norm of state: ", la.norm(state))
            quit()

    return next_state


def reward_based_fitness(
    action_sequence, props, tolerance, reward_decay, test_normalization=True
):
    """
    Calculate the fitness of an action sequence based on the reward assigned by the RL
    algorithm of the referenced work. For every action, fidelity (probability of finding the
    excitation in the last site) is calculated. A proportional reward is assigned and this value is
    acumulated over time with a decay factor, making solutions with higher fidelity in shorter times
    the fittest. Follows the rewards model of Zhang 2018.

    Parameters:
    action_sequence (list): A list of actions to be performed.
    props (ndarray): A numpy array containing propagators to evolve the state.
    tolerance (float): A tolerance value for determining reward thresholds (min. fidelity).
    reward_decay (float): A decay factor for the reward over time.
    test_normalization (bool): A flag to test if the state normalization is maintained.

    Returns:
    fitness (float): The calculated fitness value.
    """

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0

    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props, check_normalization=False)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 100 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 2500

        fitness = fitness + reward * (reward_decay**i)

        # check state normalization
    if test_normalization:
        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization failed. Norm of final state:", la.norm(state))
            quit()

    return fitness


def reward_based_fitness_late(
    action_sequence,
    props,
    initial_state,
    initial_step,
    tolerance=0.05,
    reward_decay=0.95,
    test_normalization=True,
):

    n = np.shape(props)[1]
    state = initial_state
    i = 0
    fitness = 0.0

    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props, check_normalization=False)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 100 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 2500

        fitness = fitness + reward * (reward_decay**i)

        # check state normalization

    if test_normalization:
        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization failed. Norm of final state:", la.norm(state))
            quit()

    return fitness


def localization_based(
    action_sequence, dt, props, speed_fraction, max_opt_time, test_normalization=True
):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    fitness = 0.0
    fidelity_evolution = np.asarray([])
    loc_evolution = np.sum(
        np.asarray(
            [
                np.real(state[j] * np.conjugate(state[j])) * (j + 1)
                for j in range(0, n - 1)
            ]
        )
    )

    i = 0
    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props, check_normalization=False)
        site_localization = np.sum(
            np.asarray(
                [
                    np.real(state[j] * np.conjugate(state[j])) * (j + 1)
                    for j in range(0, n - 1)
                ]
            )
        )
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))
        fidelity_evolution = np.append(fidelity_evolution, fid)
        loc_evolution = np.append(loc_evolution, site_localization)

    max_time = np.argmax(fidelity_evolution)
    speed = speed_fraction * 2 * n / (n - 1)

    if max_opt_time == 0:
        max_opt_time = max_time

    i = 0
    for fid in fidelity_evolution[0 : max_opt_time + 1]:

        reward = 1 / np.abs(loc_evolution[i] - speed * dt * i) ** 2
        fitness = fitness + fid * reward
        i += 1
    # fitness = np.max(fidelity_evolution)*(1+fitness-max_time)

    if test_normalization:
        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization failed. Norm of final state:", la.norm(state))
            quit()

    return n**2 * fitness * np.max(fidelity_evolution) / max_time


def calc_ipr(state):
    nh = np.shape(state)[0]
    ipr = 0

    for i in range(nh):
        ipr += np.real(state[i] * np.conjugate(state[i])) ** 2

    return 1 / ipr


def state_fidelity(state):
    n = np.shape(state)[0]
    fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))
    return fid


def ipr_based(action_sequence, dt, props, test_normalization=True):
    """
    Evaluate the fitness of a given action sequence based on comparison with the natural
    evolution of the system. The function calculates where the excitation is localized
    (on average) and compares with the natural propagation of a "wave" of a speed
    proportional to the natural speed by a factor given by the speed_fraction
    parameter.

    Parameters:
    action_sequence (list): A sequence of actions to be applied to the state.
    dt (float): Time step for the evolution.
    props (ndarray): Properties of the system.
    speed_fraction (float): Fraction of the speed to be considered.
    max_opt_time (int): Maximum optimization time.
    test_normalization (bool, optional): Flag to test normalization of the state. Default is True.

    Returns:
    float: The fitness value of the given action sequence.
    """

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    fitness = 0.0
    fidelity_evolution = np.asarray(state_fidelity(state))
    ipr_evolution = np.asarray(calc_ipr(state))

    i = 0
    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props, check_normalization=False)
        ipr = calc_ipr(state)
        fid = state_fidelity(state)
        fidelity_evolution = np.append(fidelity_evolution, fid)
        ipr_evolution = np.append(ipr_evolution, ipr)

    max_time = np.argmax(fidelity_evolution)

    i = 0
    for fid, ipr in zip(fidelity_evolution, ipr_evolution):

        fitness = fitness + (fid / ipr) * 0.95**i
        i += 1

    return fitness  # n**2 * fitness * np.max(fidelity_evolution) / max_time


def ipr_based2(action_sequence, dt, props, test_normalization=True):
    """
    Evaluate the fitness of a given action sequence based on comparison with the natural
    evolution of the system. The function calculates where the excitation is localized
    (on average) and compares with the natural propagation of a "wave" of a speed
    proportional to the natural speed by a factor given by the speed_fraction
    parameter.

    Parameters:
    action_sequence (list): A sequence of actions to be applied to the state.
    dt (float): Time step for the evolution.
    props (ndarray): Properties of the system.
    speed_fraction (float): Fraction of the speed to be considered.
    max_opt_time (int): Maximum optimization time.
    test_normalization (bool, optional): Flag to test normalization of the state. Default is True.

    Returns:
    float: The fitness value of the given action sequence.
    """

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    fitness = 0.0
    fidelity_evolution = np.asarray(state_fidelity(state))
    ipr_evolution = np.asarray(calc_ipr(state))

    i = 0
    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props, check_normalization=False)
        ipr = calc_ipr(state)
        fid = state_fidelity(state)
        fidelity_evolution = np.append(fidelity_evolution, fid)
        ipr_evolution = np.append(ipr_evolution, ipr)

    max_time = np.argmax(fidelity_evolution)

    i = 0
    for fid, ipr in zip(fidelity_evolution, ipr_evolution):
        alpha = 0.3
        beta = 1 - alpha
        fitness = fitness + fid * (alpha + beta / ipr)
        i += 1

    return n**2 * fitness / max_time


def fitness_func_constructor(fid_function, arguments):
    """
    Parameters:
        - fidelity function(can be either fidelity or en_fidelity)
        - arguments: arguments of fidelity functions
    Return:
        - lambda function: the fitness function as required by PyGAD
    """
    fitness = lambda vec: fid_function(vec, *arguments)

    return lambda ga_instance, solution, solution_idx: fitness(solution)


def generation_print(ga):

    solution, solution_fitness, solution_idx = ga.best_solution()

    print("Generation", ga.generations_completed)
    print("Solution: ", solution, "Fitness: ", solution_fitness)


def generation_func(ga, props, tol):
    """
    Function to be ran on every generation of the genetic algorithm.
    Prints relevant information on the best solution,
    and determines whether to stop the algorithm based on fidelity.

    Args:
        ga (GeneticAlgorithm): An instance of the genetic algorithm.
        props (dict): Propagators being used to calculate fidelity from action sequence.
        tol (float): The tolerance level for the fidelity to determine if the algorithm should stop.

    Returns:
        str: Returns "stop" if the fidelity of the best solution is greater than or equal to (1 - tol).
    """

    solution, solution_fitness, solution_idx = ga.best_solution()

    fid, time = fidelity(solution, props, return_time=True)

    print("Generation", ga.generations_completed)
    print(
        "Solution: ",
        solution,
        "Fidelity: ",
        fid,
        "Time: ",
        time,
        "Fitness: ",
        solution_fitness,
    )

    if fid >= 1 - tol:
        return "stop"


def generation_func_constructor(gen_function, arguments):
    """
    Parameters:
        - generation function
        - arguments: arguments of generation function
    Return:
        - lambda function: the mutation function as required by PyGAD
    """

    on_gen = lambda ga_instance: gen_function(ga_instance, *arguments)

    return lambda ga_instance: on_gen(ga_instance)


def actions_to_file(solution, filename, condition):
    """
    Parameters:
        - solution: best solution obtained
        - filename
        - condition: write or append

    Return:
        - saves best action sequence in file = filename
    """
    with open(filename, condition) as f1:

        writer = csv.writer(f1, delimiter=" ")
        solution = np.asarray(solution)
        for i in range(len(solution)):
            row = [solution[i]]
            writer.writerow(row)

    return True


def time_evolution(solution, props, nh, graph=False, filename=False):
    """
    Parameters:
       - solution: action sequence
       - graph: if False skips plotting graph, else enter figure name
       - file: save time evolution in a file (enter file name)
    Return:
       - array of fidelity evolution
    """

    state = np.zeros(nh, dtype=np.complex_)
    state[0] = 1.0
    nat_evolution = []

    state = np.zeros(nh, dtype=np.complex_)
    state[0] = 1.0
    fid_evolution = []

    for action in solution:

        state = calculate_next_state(state, action, props, check_normalization=False)
        fid = np.real(state[nh - 1] * np.conjugate(state[nh - 1]))
        fid_evolution = np.append(fid_evolution, fid)

        if abs(la.norm(state) - 1.0) > 1e-8:
            raise ValueError(f"Normalization failed. Norm of state: {la.norm(state)}")

    tsteps = np.shape(fid_evolution)[0] + 1

    if graph:

        axs = plt.figure(figsize=(11, 5))
        max_fid = np.max(fid_evolution)
        max_action = np.argmax(fid_evolution)

        plt.plot(np.arange(1, tsteps, 1) * 0.15, fid_evolution, "-o", label="Evol. Fid")
        plt.plot(
            np.arange(0, 200, 1) * 0.15, nat_evolution, "-o", label="sin forzamiento"
        )

        plt.grid()
        plt.title(
            " Fidelity evolution. Max. = {} on time step = {}".format(
                max_fid, max_action
            )
        )
        plt.xlabel("t")
        plt.ylabel("|f|**2")
        plt.legend()
        plt.savefig(graph)

    fid_evolution_array = np.asarray(fid_evolution)

    if filename:
        np.savetxt(filename)

    return fid_evolution_array


# ---------------------------------------------------------------------------
#
# ACTIONS FROM THE REFERENCED WORK
#
# ---------------------------------------------------------------------------


def diagonals_zhang(bmax, i, nh):
    """
    Construction of diagonals associated to referenced work. The first and last three sites
    can be controlled.

    Parameters:
    bmax (float): Control field value.
    i (int): The index determining which diagonal elements to set to 1.
    nh (int): The length of the spin chain, corresponding to the action
    matrices size.

    Returns:
    numpy.ndarray: A diagonal vector of length `nh` with specific elements set to `bmax` based on the index `i`,
    corresponding to the 16 action matrices.
    """

    b = np.full(nh, 0)

    if i == 1:
        b[0] = 1

    elif i == 2:

        b[1] = 1

    elif i == 3:

        b[0] = 1
        b[1] = 1

    elif i == 4:
        b[2] = 1  # correccion

    elif i == 5:
        b[0] = 1
        b[2] = 1

    elif i == 6:
        b[1] = 1
        b[2] = 1

    elif i == 7:
        b[0] = 1
        b[1] = 1
        b[2] = 1

    elif i == 8:
        b[nh - 3] = 1

    elif i == 9:
        b[nh - 2] = 1

    elif i == 10:
        b[nh - 3] = 1
        b[nh - 2] = 1

    elif i == 11:
        b[nh - 1] = 1

    elif i == 12:
        b[nh - 3] = 1
        b[nh - 1] = 1

    elif i == 13:
        b[nh - 2] = 1
        b[nh - 1] = 1

    elif i == 14:
        b[nh - 3] = 1
        b[nh - 2] = 1
        b[nh - 1] = 1

    elif i == 15:
        b[:] = 1
    else:
        b = np.full(nh, 0.0)  # correccion

    b = bmax * b

    return b


def actions_paper2(bmax, nh):

    actions = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales_paper2(bmax, i, nh)

        J = 1

        for k in range(0, nh - 1):
            actions[i, k, k + 1] = J
            actions[i, k + 1, k] = actions[i, k, k + 1]

        for k in range(0, nh):

            actions[i, k, k] = b[k]

    return actions


# ---------------------------------------------------------------------------
#
# SETS OF ACTIONS DEFINED TO RUN ON ONE SITE ONLY
#
# ---------------------------------------------------------------------------


def one_field_actions(bmax, nh):

    action_matrices = np.zeros((nh + 1, nh, nh))
    J = 1.0

    for i in range(0, nh):

        for k in range(0, nh - 1):
            action_matrices[i + 1, k, k + 1] = J
            action_matrices[i + 1, k + 1, k] = action_matrices[i + 1, k, k + 1]

        action_matrices[i + 1, i, i] = bmax

    for k in range(0, nh - 1):
        action_matrices[0, k, k + 1] = J
        action_matrices[0, k + 1, k] = action_matrices[0, k, k + 1]

    return action_matrices


def one_field_actions_extra(bmax, nh):
    """
    i = [0,n-1] : Acciones por sitio
    i = n : Campos apagados
    i = n+1 : Campo negativo primer sitio
    i = n+2 : Campo negativo último sitio
    """
    action_matrices = np.zeros((nh + 3, nh, nh))
    J = 1.0

    for i in range(0, nh):

        for k in range(0, nh - 1):
            action_matrices[i, k, k + 1] = J
            action_matrices[i, k + 1, k] = action_matrices[i, k, k + 1]

        action_matrices[i, i, i] = bmax

    # campos apagados (i=nh)
    for k in range(0, nh - 1):
        action_matrices[nh, k, k + 1] = J
        action_matrices[nh, k + 1, k] = action_matrices[nh, k, k + 1]

    # campo primer sitio
    for k in range(0, nh - 1):
        action_matrices[nh + 1, k, k + 1] = J
        action_matrices[nh + 1, k + 1, k] = action_matrices[i, k, k + 1]

    action_matrices[nh + 1, 0, 0] = -bmax

    # campo ultimo sitio
    for k in range(0, nh - 1):
        action_matrices[nh + 2, k, k + 1] = J
        action_matrices[nh + 2, k + 1, k] = action_matrices[i, k, k + 1]

    action_matrices[nh + 2, nh - 1, nh - 1] = -bmax

    return action_matrices


def one_field_actions_weak(bmax, nh):

    action_matrices = np.zeros((nh + 1, nh, nh))
    J = np.ones(nh)
    J[0] = 0.5
    J[nh - 1] = 0.5

    for i in range(0, nh):

        for k in range(0, nh - 1):
            action_matrices[i + 1, k, k + 1] = J[k]
            action_matrices[i + 1, k + 1, k] = action_matrices[i + 1, k, k + 1]

        action_matrices[i + 1, i, i] = bmax

    for k in range(0, nh - 1):
        action_matrices[0, k, k + 1] = J[k]
        action_matrices[0, k + 1, k] = action_matrices[0, k, k + 1]

    return action_matrices


def refined_cns(state, action_index, props):
    # Retrieve the matrix corresponding to the action index
    p = props[action_index]

    # Perform matrix-vector multiplication directly
    next_state = p @ state

    # Return the result as a flat 1D array
    return next_state.ravel()


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
    n = np.shape(states)[1]
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


def reward_based_fitness_gpu(
    action_sequences, props, tolerance, reward_decay, test_normalization=False
):
    device = "cuda"

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props, dtype=T.complex64, device=device, requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length), dtype=T.complex64, device=device
    )
    states[:, 0, 0] = 1.0  # Initial condition

    # Compute states using batched matrix multiplication
    for i in range(0, steps):
        states[:, i + 1, :] = T.bmm(
            props[action_sequences[:, i]], states[:, i, :].unsqueeze(-1)
        ).squeeze(-1)

    # Compute fidelity
    fid = states[:, :, -1].abs() ** 2  # Take absolute squared of last column

    # Compute rewards in parallel
    rewards = T.zeros_like(fid, device=device)
    rewards[fid <= 0.8] = 10 * fid[fid <= 0.8]

    mask = (fid > 0.8) & (fid <= 1 - tolerance)
    rewards[mask] = 100 / (1 + T.exp(10 * (1 - tolerance - fid[mask])))

    rewards[fid > 1 - tolerance] = 2500

    # Apply decay and sum fitness
    decay_factors = reward_decay ** T.arange(steps + 1, device=device).unsqueeze(
        0
    )  # Shape: (1, steps)
    fitness = T.sum(rewards * decay_factors, dim=1)  # Sum over steps

    return fitness.cpu().numpy()  # Convert once at the end
