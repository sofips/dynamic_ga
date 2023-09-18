'''
Generates configuration files for different runs of 
genetic algorithm implementation, using configparser
library. Creates directory and saves script in it.
'''

import configparser
import os
import sys

# Create instance of ConfigParser
config = configparser.ConfigParser()


# system parameters

n =  7
dt = 0.15
b = 0.9  #for j_fidelity

config['system_parameters'] = {'n': str(n),
                               'dt': str(dt),
                               'b': str(b) }


# genetic algorithm parameters

num_generations = 2000
num_genes = 50
sol_per_pop = 1000
fidelity_tolerance = 0.95
saturation = 20


config['ga_initialization'] = {'num_generations': str(num_generations),
                               'num_genes': str(num_genes),
                               'sol_per_pop': str(sol_per_pop),
                               'fidelity_tolerance': str(fidelity_tolerance),
                               'saturation': str(saturation)}

# crossover and parent selection
num_parents_mating = sol_per_pop//5
parent_selection_type = 'sss'
keep_elitism = sol_per_pop//10
crossover_type = 'uniform'
crossover_probability = 0.6

config['parent_selection'] = {'num_parents_mating': str(num_parents_mating),
                              'parent_selection_type': parent_selection_type,
                              'keep_elitism': str(keep_elitism)}

config['crossover'] = {'crossover_type': crossover_type,
                       'crossover_probability': str(crossover_probability)}


# other mutation parameters

mutation_probability = 0.99
mutation_num_genes = 1

config['mutation'] = {'mutation_probability': str(mutation_probability),
                      'mutation_num_genes': str(mutation_num_genes)}


# execution and results saving
directory = 'test'
n_samples = 2

config['saving'] = {'directory': directory,
                    'n_samples': str(n_samples)}
script = 'dc_ga.py'

isExist = os.path.exists(directory)
if not isExist:
    os.mkdir(directory)
else:
    print('Warning: Directory already existed')

src = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.join(src, script)
mod_name = os.path.join(src, 'gmod.py')

cmd = f'cp "{script_name}" "{directory}"'
os.system(cmd)
cmd = f'cp "{mod_name}" "{directory}"'
os.system(cmd)
config_name = directory + '/'+ 'ga'+directory+'.ini'

with open(config_name, 'w') as configfile:
    config.write(configfile)

cmd = f'python3 "{script_name}" "{config_name}"'
os.system(cmd)



