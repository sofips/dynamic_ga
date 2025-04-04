"""
Generates configuration files for different runs of 
genetic algorithm implementation, using configparser
library. Creates directory and saves script in it.
"""

import configparser
import os
import sys

# Create instance of ConfigParser
config = configparser.ConfigParser()

# system parameters

n = int(sys.argv[1])  # number of spins in the system
dt = 0.15  # length of temporal steps
b = 100  # magnetic field strength
speed_fraction = 1  # fraction of qsl speed
#max_optimization_time = 0 #int(n//2/0.15)

# genetic algorithm parameters

num_generations = 1000
num_genes = 5 * n
sol_per_pop = 4096
fidelity_tolerance = 0.05
reward_decay = 0.95  # time decay to achieve faster transmission
saturation = 20

# crossover and parent selection
num_parents_mating = sol_per_pop // 10
parent_selection_type = "sss"
keep_elitism = sol_per_pop // 10
crossover_type = "uniform"
crossover_probability = 0.8

# other mutation parameters

mutation_probability = 0.99
mutation_num_genes = n

# execution and results saving
directory = sys.argv[2]
n_samples = 10


config["system_parameters"] = {
    "n": str(n),
    "dt": str(dt),
    "b": str(b),
    "speed_fraction": str(speed_fraction),
    #"max_optimization_time": str(max_optimization_time),
}

config["ga_initialization"] = {
    "num_generations": str(num_generations),
    "num_genes": str(num_genes),
    "sol_per_pop": str(sol_per_pop),
    "fidelity_tolerance": str(fidelity_tolerance),
    "saturation": str(saturation),
    "reward_decay": str(reward_decay),
}


config["parent_selection"] = {
    "num_parents_mating": str(num_parents_mating),
    "parent_selection_type": parent_selection_type,
    "keep_elitism": str(keep_elitism),
}

config["crossover"] = {
    "crossover_type": crossover_type,
    "crossover_probability": str(crossover_probability),
}


config["mutation"] = {
    "mutation_probability": str(mutation_probability),
    "mutation_num_genes": str(mutation_num_genes),
}


config["saving"] = {
    "directory": directory,
    "n_samples": str(n_samples),
}
script = "dc_ga.py"

isExist = os.path.exists(directory)
if not isExist:
    os.mkdir(directory)
else:
    print("Warning: Directory already existed")

src = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.join(src, script)
mod_name = os.path.join(src, "dgamod.py")

cmd = f'cp "{script_name}" "{directory}"'
os.system(cmd)
cmd = f'cp "{mod_name}" "{directory}"'
os.system(cmd)
config_name = directory + "/" + "ga" + directory + ".ini"

with open(config_name, "w") as configfile:
    config.write(configfile)

script_name = directory + "/" + script
config_name = "ga" + directory + ".ini"

cmd = f'python3 "{script_name}" "{config_name}"'
os.system(cmd)
