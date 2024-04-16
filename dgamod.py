import numpy as np
import scipy.linalg as la
import cmath as cm
import csv
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os




def delta(k, n):

    if k == n:
        d = 1.0
    else:
        d = 0.0

    return d


def diagonales(bmax, i, nh):

    if i == 1:
        b = np.full(nh, -1)
        b[0] = 1

    elif i == 2:
        b = np.full(nh, -1)

        b[1] = 1

    elif i == 3:

        b = np.full(nh, -2.0)

        b[0] = 0.0
        b[1] = 0.0

    elif i == 4:

        b = np.full(nh, -1)

        b[2] = 1  # correccion

    elif i == 5:

        b = np.full(nh, -2.0)

        b[0] = 0.0
        b[2] = 0.0

    elif i == 6:

        b = np.full(nh, -2.0)  # correccion

        b[1] = 0.0
        b[2] = 0.0

    elif i == 7:

        b = np.full(nh, -3)

        b[0] = -1
        b[1] = -1
        b[2] = -1

    elif i == 8:
        b = np.full(nh, -1)
        b[nh - 3] = 1

    elif i == 9:

        b = np.full(nh, -1)
        b[nh - 2] = 1

    elif i == 10:

        b = np.full(nh, -2.0)
        b[nh - 3] = 0.0
        b[nh - 2] = 0.0

    elif i == 11:

        b = np.full(nh, -1)
        b[nh - 1] = 1

    elif i == 12:
        b = np.full(nh, -2.0)

        b[nh - 3] = 0.0
        b[nh - 1] = 0.0

    elif i == 13:

        b = np.full(nh, -2.0)

        b[nh - 2] = 0.0  # correccion
        b[nh - 1] = 0.0  # correccion

    elif i == 14:

        b = np.full(nh, -3)

        b[nh - 3] = -1
        b[nh - 2] = -1
        b[nh - 1] = -1

    elif i == 15:

        b = np.full(nh, -2.0)
    else:
        b = np.full(nh, 0.0)  # correccion

    b = bmax * b

    return b


def actions(bmax, nh):

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales(bmax, i, nh)

        J = -0.5  # [-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh - 1):
            mat_acc[i, k, k + 1] = J
            mat_acc[i, k + 1, k] = mat_acc[i, k, k + 1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc


def gen_props(actions, n, b, dt):

    mat_acc = actions
    comp_i = complex(0, 1)
    en = np.zeros((16, n), dtype=np.complex_)
    bases = np.zeros((16, n, n), dtype=np.complex_)
    propagadores = np.zeros((16, n, n), dtype=np.complex_)
    desc_esp = np.zeros((16, n, n), dtype=np.complex_)

    for j in range(0, 16):  # para cada matriz de accion

        en[j, :], bases[j, :, :] = la.eig(mat_acc[j, :, :])

    #     for k in range(0, n):
    #         p = np.outer(bases[j, :, k], bases[j, :, k])

    #         propagadores[j, :, :] = (
    #             propagadores[j, :, :] + cm.exp(-comp_i * dt * en[j, k]) * p
    #         )

    #         desc_esp[j, :, :] = desc_esp[j, :, :] + p * en[j, k]

    # # check de descomposición espectral
    # check_de = True

    # for k in np.arange(0, 16):
    #     for i in np.arange(0, n):
    #         for j in np.arange(0, n):

    #             if mat_acc[k, i, j] - desc_esp[k, i, j] > 1e-6:
    #                 print("error desc. esp")
    #                 check_de = False

    # if check_de:
    #     print("Descomposicion espectral: correcta")

    for i in range(0, 16):  # para cada matriz de accion
        propagadores[i, :, :] =  expm(-1j*actions[i]*dt)

    check_prop = True

    for a in np.arange(0, 16):
        for j in np.arange(0, n):
            errores = calculate_next_state( bases[a, :, j],a,propagadores)-np.exp(-comp_i * dt * en[a, j]) * bases[a, :, j]
            # errores = (
            #     np.matmul(propagadores[a, :, :], bases[a, :, j])
            #     - np.exp(-comp_i * dt * en[a, j]) * bases[a, :, j]
            # )
            et = np.sum(errores)
            if la.norm(et) > 1e-8:
                print("error en propagacion")
                check_prop = False

    if check_prop:
        print("Propagacion de autoestados: correcta")

    return propagadores


def fidelity(action_sequence, props, return_time=False):
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

    if abs(la.norm(state) - 1.0) > 1e-8:
        print("FALLO EN LA NORMALIZACION", la.norm(state))

    if return_time:
        return max_fid, imax

    return max_fid

def reward_based_fitness(action_sequence, props):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0
    tolerance = 0.05

    for action in action_sequence:
        i += 1
        #state = np.matmul(props[action, :, :], state)
        state  = calculate_next_state(state,action,props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 1000 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 25000

        fitness = fitness + reward * (0.95**i)

        # check state normalization

        if abs(la.norm(state) - 1.0) > 1E-8:
            print("Normalization failed!!!!", la.norm(state))
            quit()

    return reward

def reward_based_fitness_up_to_max(action_sequence, props,tolerance=0.05, reward_decay=0.95):

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    i = 0
    fitness = 0.0
    fidelity_evolution = np.asarray([])

    for action in action_sequence:
        i += 1
        state  = calculate_next_state(state,action,props)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))
        fidelity_evolution = np.append(fidelity_evolution,fid)

    #max_fid = np.max(fidelity_evolution)
    max_time = np.argmax(fidelity_evolution)

    for fid in fidelity_evolution[0:max_time+1]:

        if fid <= 0.8:
            reward = 10 * fid
        elif 0.8 <= fid <= 1 - tolerance:
            reward = 1000 / (1 + np.exp(10 * (1 - tolerance - fid)))
        else:
            reward = 25000

        fitness = fitness + reward * (reward_decay**i)

        # check state normalization

        if abs(la.norm(state) - 1.0) > 1E-8:
            print("Normalization failed!!!!", la.norm(state))
            quit()

    return reward


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


def generation_func(ga, props, tol, directory, histogram = True):

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
        population_histogram(ga, directory,props)

    if fid >= 1-tol:
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


def time_evolution(solution, propagadores, nh, graph=False, filename=False):
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

        state = np.matmul(propagadores[action, :, :], state)
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

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales_paper(bmax, i, nh)

        J = 1  # [-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh - 1):
            mat_acc[i, k, k + 1] = J
            mat_acc[i, k + 1, k] = mat_acc[i, k, k + 1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc

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

        #b = np.full(nh, -1)
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

        # b[0] = 1
        # b[1] = 1
        # b[2] = 1
        # b[nh - 3] = 1
        # b[nh - 2] = 1
        # b[nh - 1] = 1
        b[:] = 1
    else:
        b = np.full(nh, 0.0)  # correccion

    b = bmax * b

    return b


def actions_paper2(bmax, nh):

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales_paper2(bmax, i, nh)

        J = 1  # [-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh - 1):
            mat_acc[i, k, k + 1] = J
            mat_acc[i, k + 1, k] = mat_acc[i, k, k + 1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc


def calculate_next_state(state, action_index, props):

    state = np.transpose(np.mat(state))
    p = props[action_index] 
    next_state = p*state
    next_state = np.asarray(np.transpose(next_state))
    next_state = np.squeeze(next_state)

    if abs(la.norm(next_state) - 1.) > 1E-8:
        print('Normalization failed!!! ', la.norm(state))
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
    figure, axs = plt.subplots(2,1,figsize=(12, 4))
    nbins = 100
    plt.subplots_adjust(wspace = 0.2, hspace=0.7) 
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

    for i in range(0,ga.pop_size[0]):
        action_sequence = population[i,:]
        individual_fidelity = fidelity(action_sequence,props)
        population_fidelity.append(individual_fidelity)

    # array of fidelities in population

    population_fidelity = np.asarray(population_fidelity)

    ax = axs[0]

    # plot histogram of fidelity distribution
    hist, bins, c = ax.hist(
        population_fidelity, bins=nbins, range=[0, 1], edgecolor="black", color="#DDFFDD"
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
    #---------------------------------------------------
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
    #ga.plot_genes(graph_type = 'histogram', save_dir = dirname + "/gene_dist" + str(ng).zfill(3), solutions = 'all')

def new_diagonals(bmax, i, nh):
    
    b = np.full(nh, 0)

    if i == 1:
        b[0] = 1

    elif i == 2:

        b[1] = 1

    elif i == 3:
        b[0] = -1
        # b[0] = 1
        # b[1] = 1

    elif i == 4:

        b[2] = 1  # correccion

    elif i == 5:
        b[1] = -1
        # b[0] = 1
        # b[2] = 1

    elif i == 6:
        b[2] = -1
        # b[1] = 1
        # b[2] = 1

    elif i == 7:

        b[0] = 1
        b[1] = 1
        b[2] = 1

    elif i == 8:

        b[nh - 3] = 1

    elif i == 9:

        #b = np.full(nh, -1)
        b[nh - 2] = 1

    elif i == 10:
        b[nh - 3] = -1
        # b[nh - 3] = 1
        # b[nh - 2] = 1

    elif i == 11:

        b[nh - 1] = 1

    elif i == 12:

        b[nh - 2] = -1
        # b[nh - 3] = 1
        # b[nh - 1] = 1

    elif i == 13:

        b[nh - 1] = -1

        # b[nh - 2] = 1
        # b[nh - 1] = 1

    elif i == 14:

        b[nh - 3] = 1
        b[nh - 2] = 1
        b[nh - 1] = 1

    elif i == 15:

        # b[0] = 1
        # b[1] = 1
        # b[2] = 1
        # b[nh - 3] = 1
        # b[nh - 2] = 1
        # b[nh - 1] = 1
        b[:] = 1
    else:
        # b = np.full(nh, 0.0)  # correccion
        b[:] = -1

    b = bmax * b

    return b

def new_actions(bmax, nh):

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = new_diagonals(bmax, i, nh)

        J = 1  # [-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh - 1):
            mat_acc[i, k, k + 1] = J
            mat_acc[i, k + 1, k] = mat_acc[i, k, k + 1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc