import numpy as np
import os

for i in [1]:
    dirname = 'n13_stats_lbv11_parallel_{}'.format(i)
    cmd = 'nice -20 python3 exp_gen.py 13 {} >> out_{} &'.format(dirname,i)
    os.system(cmd)
    os.system('bg')
    os.system('disown')


