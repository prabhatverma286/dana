import itertools
import json

import matplotlib.pyplot as plt
# COLORS = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'pink',
#         'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
#         'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']
#
# with open("resources\\rewards_train_freq.json", "r") as f:
#     contents = f.read()
#     results_dict = json.loads(contents)
#     y_range = [len([1 for v in value if v >= 199]) for _, value in results_dict.items()]
#     x_range = [key[11:] for key, _ in results_dict.items()]
#     # lists = sorted(zip(*[x_range, y_range]))
#     # x_range, y_range = list(zip(*lists))
#
#     i = 0
#     plt.plot(x_range, y_range, marker='o')
#     plt.xticks(rotation=90)

# plt.show()

with open("pong_rewards_dueling.json", "r") as f:
    dueling = json.loads(f.read())["pong"]

with open("pong_rewards_simple.json", "r") as f:
    double = json.loads(f.read())["pong"]

with open("pong_rewards_vanilla.json", "r") as f:
    vanilla = json.loads(f.read())["pong"]


# dueling = dueling[0:len(dueling):5]
# double = double[0:len(double):5]
# vanilla = vanilla[0:len(vanilla):5]


avg_dueling = []

for i in range(0, len(dueling), 5):
    avg_dueling.append((dueling[i] + dueling[i+1] + dueling[i+2] + dueling[i+3] + dueling[i+4])/5)

avg_double = []

for i in range(0, len(double), 5):
    avg_double.append((double[i] + double[i+1] + double[i+2] + double[i+3] + double[i+4])/5)

avg_vanilla = []

for i in range(0, len(vanilla) - 5, 5):
    avg_vanilla.append((vanilla[i] + vanilla[i+1] + vanilla[i+2] + vanilla[i+3] + vanilla[i+4])/5)


double = avg_double
dueling = avg_dueling
vanilla = avg_vanilla

plt.plot(range(len(dueling)), dueling, label="Dueling Double")
plt.plot(range(len(double)), double, label="Double")
plt.plot(range(len(vanilla)), vanilla, label="Vanilla")


# plt.xlabel("Train Frequency")
# plt.ylabel("Number of episodes with rewards >= 199")
plt.legend()

plt.show()
