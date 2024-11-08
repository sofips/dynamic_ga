import os
import numpy as np

dimensions = np.arange(10,20,1)

for dim in dimensions:
    dirname = 'n{}_one_action_per_site_acc'.format(dim)
    cmd = 'nice -20 python3 exp_gen.py {} {} >out{}_oaps_acc.out &'.format(dim, dirname, dim)
    os.system(cmd)
    os.system('bg')
    os.system('disown')
