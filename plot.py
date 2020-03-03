import itertools
import json

import matplotlib.pyplot as plt
# COLORS = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'pink',
#         'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
#         'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']
#
# with open("rewards_gamma.json", "r") as f:
#     contents = f.read()
#     results_dict = json.loads(contents)
#     y_range = [len([1 for v in value if v >= 199]) for _, value in results_dict.items()]
#     x_range = [f"{key[6:]:.4}" for key, _ in results_dict.items()]
#     # lists = sorted(zip(*[x_range, y_range]))
#     # x_range, y_range = list(zip(*lists))
#
#     i = 0
#     plt.bar(x_range, y_range)
#     plt.xticks(rotation=90)
#
# plt.show()

with open("chip_performance.json", "r") as f:
    performance = json.loads(f.read())

plt.plot(range(1, 13), performance["run_1"]["performance_1000"], marker='o', linestyle='dotted', markersize=2)
plt.plot(range(1, 13), performance["run_2"]["performance_1000"], color='green', marker='o', linestyle='dotted', markersize=2)
plt.plot(range(1, 13), performance["run_3"]["performance_1000"], color='darkblue', marker='o', linestyle='dotted', markersize=2)
plt.plot(range(1, 13), performance["run_4"]["performance_1000"], color='red', marker='o', linestyle='dotted', markersize=2)
plt.plot(range(1, 13), performance["run_5"]["performance_1000"], color='darkgreen', marker='o', linestyle='dotted', markersize=2)

plt.xlabel("Number of threads")
plt.ylabel("Number of executions in one second")
plt.show()
