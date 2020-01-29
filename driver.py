import json
import subprocess
import sys
import numpy as np
import default_params

env = "CartPole-v0"
params = default_params.cartpole

for b in range(20000, 70000, 5000):
    params['buffer_size'] = b
    subprocess.run([sys.executable, 'train.py', '--env_id', env, '--identifier', 'buffer_size_' + str(b), '--json_arguments', json.dumps(params)])
