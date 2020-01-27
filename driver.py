import json
import subprocess
import sys
import numpy as np
import default_params

env = "CartPole-v0"
params = default_params.cartpole

for ep in np.arange(0.1, 0.3, step=0.02).round(decimals=4):
    params['exploration_final_eps'] = ep
    subprocess.run([sys.executable, 'train.py', '--env_id', env, '--identifier', 'exploration_final_epsilon' + str(ep), '--json_arguments', json.dumps(params)])
