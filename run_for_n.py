import os
import numpy as np

dimensions = np.arange(6,20,1)

for dim in dimensions:
    dirname = 'n{}_si'.format(dim)
    cmd = 'nice -20 python3 exp_gen.py {} {} >out{}_si.out &'.format(dim, dirname, dim)
    os.system(cmd)
    os.system('bg')
    os.system('disown')
