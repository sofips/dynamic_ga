import numpy as np
import scipy.linalg as la
import cmath as cm
import csv
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os


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
                    calculate_next_state(bases[a, :, j], a, props)
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


def reward_based_fitness(action_sequence, props, tolerance, reward_decay, test_normalization: True):
    """
    Calculate the fitness of an action sequence based on the reward assigned by the RL 
    algorithm of the referenced work. For every action, fidelity (probability of finding the
    excitation in the last site) is calculated. A proportional reward is assigned and this value is
    acumulated over time with a decay factor, making solutions with higher fidelity in shorter times 
    the fittest.

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
        state = calculate_next_state(state, action, props)
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
):

    n = np.shape(props)[1]

    state = initial_state
    i = 0
    fitness = 0.0

    for action in action_sequence[initial_step:]:
        i += 1
        # state = np.matmul(props[action, :, :], state)
        state = calculate_next_state(state, action, props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        # uso los valores que usan ellos (sin multiplicar por 10!!!!!!!!!!!!)

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 100 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 2500

        fitness = fitness + reward * (reward_decay**i)

        # check state normalization

        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization failed!!!!", la.norm(state))
            quit()

    return fitness


def reward_based_fitness_up_to_max(
    action_sequence, props, tolerance=0.05, reward_decay=0.95
):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0
    fidelity_evolution = np.asarray([])

    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))
        fidelity_evolution = np.append(fidelity_evolution, fid)

    # max_fid = np.max(fidelity_evolution)
    max_time = np.argmax(fidelity_evolution)

    for fid in fidelity_evolution[0 : max_time + 1]:

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 1000 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 25000

        fitness = fitness + reward * (reward_decay**i)

        # check state normalization

        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization failed!!!!", la.norm(state))
            quit()

    return fitness


def non_acumulative(action_sequence, props, tolerance=0.01, reward_decay=0.95):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0
    fidelity_evolution = np.asarray([])

    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))
        fidelity_evolution = np.append(fidelity_evolution, fid)

    max_fid = np.max(fidelity_evolution)
    max_time = np.argmax(fidelity_evolution)

    reward = 1000 / (1 + np.exp(10 * (1 - tolerance - max_fid)))

    b = 0.8
    a = 1 - b
    fitness = reward * (a + b / max_time**2)

    # check state normalization

    if abs(la.norm(state) - 1.0) > 1e-8:
        print("Normalization failed!!!!", la.norm(state))
        quit()

    return fitness


def reward_based_with_differences(
    action_sequence, props, tolerance=0.05, reward_decay=0.95
):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0
    fidelity_evolution = np.asarray([])
    differences = np.asarray([0])

    for action in action_sequence:
        i += 1
        state = calculate_next_state(state, action, props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))
        fidelity_evolution = np.append(fidelity_evolution, fid)
        if i >= 1:
            differences = np.append(
                differences, (fid**2 - fidelity_evolution[i - 2] ** 2)
            )

    max_time = np.argmax(fidelity_evolution)

    i = 0

    for fid in fidelity_evolution[0 : max_time + 1]:

        i += 1

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 0.95:
            reward = 1000 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 25000 * fid

        fitness = (
            fitness + reward * (reward_decay**i) + differences[i]
        )  # /np.mean(differences)
        # check state normalization

        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization failed!!!!", la.norm(state))
            quit()

        b = 0.5

        a = 1 - b

    return fitness


def localization_based(action_sequence, props, speed_fraction, max_opt_time):

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
    for action in action_sequence:  # en cada accion
        i += 1
        state = calculate_next_state(state, action, props)
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

        reward = 1 / np.abs(loc_evolution[i] - speed * 0.15 * i) ** 2
        fitness = fitness + fid * reward
        i += 1
    # fitness = np.max(fidelity_evolution)*(1+fitness-max_time)
    return n**2 * fitness * np.max(fidelity_evolution) / max_time


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


def generation_func(ga, props, tol, directory, histogram=True):

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

    if histogram and (
        ga.generations_completed == 1 or ga.generations_completed % 5 == 0
    ):
        population_histogram(ga, directory, props)

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

        state = np.matmul(props[action, :, :], state)
        # fid = np.real(state[nh-1])**2+np.imag(state[nh-1])**2

        fid = np.real(state[nh - 1] * np.conjugate(state[nh - 1]))
        fid_evolution = np.append(fid_evolution, fid)

        if abs(la.norm(state) - 1.0) > 1e-8:
            print("FALLO EN LA NORMALIZACION", la.norm(state))

        # else:
        # print('NORMALIZACION OK: ',la.norm(state))

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
            " Evolucion fidelidad, max = {}, accion = {}".format(max_fid, max_action)
        )
        plt.xlabel("t")
        plt.ylabel("|f|**2")
        plt.legend()
        plt.savefig(graph)

    fid_evolution_array = np.asarray(fid_evolution)

    if filename:
        np.savetxt(filename)

    return fid_evolution_array


################
# acciones zhang#
################


def diagonales_paper(bmax, i, nh):

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

        b = np.full(nh, -1)
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

        b[0] = 1
        b[1] = 1
        b[2] = 1
        b[nh - 3] = 1
        b[nh - 2] = 1
        b[nh - 1] = 1

    else:
        b = np.full(nh, 0.0)  # correccion

    b = bmax * b

    return b


def actions_paper(bmax, nh):

    actions = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales_paper(bmax, i, nh)

        J = 1  # [-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh - 1):
            actions[i, k, k + 1] = J
            actions[i, k + 1, k] = actions[i, k, k + 1]

        for k in range(0, nh):

            actions[i, k, k] = b[k]

    return actions


def diagonales_paper2(bmax, i, nh):

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

        J = 1  # [-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh - 1):
            actions[i, k, k + 1] = J
            actions[i, k + 1, k] = actions[i, k, k + 1]

        for k in range(0, nh):

            actions[i, k, k] = b[k]

    return actions


def calculate_next_state(state, action_index, props):

    state = np.transpose(np.mat(state))
    p = props[action_index]
    next_state = p * state
    next_state = np.asarray(np.transpose(next_state))
    next_state = np.squeeze(next_state)

    if abs(la.norm(next_state) - 1.0) > 1e-8:
        print("Normalization failed!!! ", la.norm(next_state))
        quit()

    return next_state


def population_histogram(ga, directory, props):
    """
    For a given instance of genetic algorithm, creates a directory
    called hist_frames and plots histograms of population's fidelity
    distribution together with the action distribution.

    Parameters:
    - ga: genetic algorithm instance (See PyGAD documentation)
    - directory: to save frames
    - props: propagators to calculate fidelity in transmission
    """
    figure, axs = plt.subplots(2, 1, figsize=(12, 4))
    nbins = 100
    plt.subplots_adjust(wspace=0.2, hspace=0.7)
    # creates directory if it doesnt exist

    dirname = directory + "/hist_frames"
    isExist = os.path.exists(dirname)

    if not isExist:
        os.mkdir(dirname)

    # access population
    population = ga.population
    population_fidelity = []
    # access n. of generations completed
    ng = ga.generations_completed

    for i in range(0, ga.pop_size[0]):
        action_sequence = population[i, :]
        individual_fidelity = fidelity(action_sequence, props)
        population_fidelity.append(individual_fidelity)

    # array of fidelities in population

    population_fidelity = np.asarray(population_fidelity)

    ax = axs[0]

    # plot histogram of fidelity distribution
    hist, bins, c = ax.hist(
        population_fidelity,
        bins=nbins,
        range=[0, 1],
        edgecolor="black",
        color="#DDFFDD",
    )

    # configure yticks to show percentage of total pop. number
    max_value = int(np.max(hist))
    y = np.linspace(int(0), max_value, 10, dtype=int)
    ax.set_yticks(y)
    ax.set_yticklabels(y * 100 / ga.pop_size[0])

    x = [0]
    x = x + [i / 10 for i in np.arange(0, 10, 1)]
    ax.set_xticks(x)

    # set grid, title and labels
    plt.grid()
    plt.title("Population distribution for gen. number " + str(ng).zfill(3))
    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Population percentage")

    population = population.flatten()
    # --------------------------------------------------
    # array of action distributions
    # ---------------------------------------------------
    ax = axs[1]

    # plot histogram of fidelity distribution
    hist, bins, c = ax.hist(
        population, bins=nbins, range=[0, 17], edgecolor="black", color="#DDFFDD"
    )

    # configure yticks to show percentage of total pop. number
    max_value = int(np.max(hist))
    y = np.linspace(int(0), max_value, 10, dtype=int)
    ax.set_yticks(y)
    ax.set_yticklabels(y * 100 / ga.pop_size[0] / ga.pop_size[1])

    x = np.arange(0, 16, 1)
    ax.set_xticks(x)

    # set grid, title and labels
    plt.grid()
    plt.title("Action distribution for gen. number " + str(ng).zfill(3))
    ax.set_xlabel("Action")
    ax.set_ylabel("Gene percentage")
    plt.tight_layout()
    # save to file
    filename = dirname + "/hist_frame" + str(ng).zfill(3) + ".png"
    plt.savefig(filename)
    plt.close()
    # ga.plot_genes(graph_type = 'histogram', save_dir = dirname + "/gene_dist" + str(ng).zfill(3), solutions = 'all')


def new_diagonals(bmax, i, nh):
    """
    Función para definir la diagonal de las matrices. Usa el mismo offset que el paper de Zhang (como
    si movieramos el 0 de energía) para que 0 -> campo apagado, +-1-> campo encendido en una u otra dirección.

    Parámetros:
        bmax: magnitud del campo magnético
        i: índice de acción
        n: dimensión de las matrices
    """

    b = np.full(nh, 0)

    if i == 1:
        b[0] = 1

    elif i == 2:
        b[1] = 1

    elif i == 3:
        b[0] = -1

    elif i == 4:
        b[2] = 1  # correccion

    elif i == 5:
        b[1] = -1

    elif i == 6:
        b[2] = -1

    elif i == 7:

        b[0] = 1
        b[1] = 1
        b[2] = 1

    elif i == 8:
        b[nh - 3] = 1

    elif i == 9:
        b[nh - 2] = 1

    elif i == 10:
        b[nh - 3] = -1

    elif i == 11:
        b[nh - 1] = 1

    elif i == 12:
        b[nh - 2] = -1

    elif i == 13:
        b[nh - 1] = -1

    elif i == 14:
        b[nh - 3] = 1
        b[nh - 2] = 1
        b[nh - 1] = 1

    elif i == 15:
        b[:] = 1

    else:
        b[:] = -1  # no tiene acción que no haga nada

    b = bmax * b

    return b


def new_actions(bmax, nh):

    actions = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = new_diagonals(bmax, i, nh)
        J = 1

        for k in range(0, nh - 1):
            actions[i, k, k + 1] = J
            actions[i, k + 1, k] = actions[i, k, k + 1]

        for k in range(0, nh):

            actions[i, k, k] = b[k]

    return actions


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
