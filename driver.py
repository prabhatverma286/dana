import json
import subprocess
import sys
import numpy as np
import default_params

env = "CartPole-v0"
params = default_params.cartpole

for g in np.arange(0.99, 0.89, -0.01):
    params['gamma'] = g
    subprocess.run([sys.executable, 'train.py', '--env_id', env, '--identifier', 'gamma_' + str(g), '--json_arguments', json.dumps(params)])
