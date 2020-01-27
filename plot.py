import itertools
import json

import matplotlib.pyplot as plt
COLORS = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

with open("exploration_final_eps_summary.json", "r") as f:
    contents = f.read()
    results_dict = json.loads(contents)
    y_range = [len([1 for v in value if v >= 199]) for _, value in results_dict.items()]
    x_range = [(key.strip("exploration_final_epsilon")[:5]) for key, _ in results_dict.items()]
    lists = sorted(zip(*[x_range, y_range]))
    x_range, y_range = list(zip(*lists))

    i = 0
    plt.bar(x_range, y_range)
    plt.xticks(rotation=90)

plt.show()
