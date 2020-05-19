# helper file mainly used to plot results
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
# with open("breakout_evaluation_1.json", "r") as f:
#     breakout = json.loads(f.read())

# with open("resources\\sonic_episode_rewards_1.json", "r") as f:
#     pdd_dqn = json.loads(f.read())["sonic"]

with open("resources\\cartpole_tuned.json", "r") as f:
    dueling = json.loads(f.read())["cartpole"]

with open("resources\\cartpole_default.json", "r") as f:
    double = json.loads(f.read())["cartpole"]
#
# with open("resources\\cartpole_vanilla.json", "r") as f:
#     vanilla = json.loads(f.read())["cartpole"]
# #
dueling = dueling[:-3]
double = double[:-1]
# vanilla = vanilla[:4995]
# dueling = dueling[0:len(dueling):5]
# double = double[0:len(double):5]
# vanilla = vanilla[0:len(vanilla):5]


avg_dueling = []
#
for i in range(0, len(dueling), 5):
    avg_dueling.append((dueling[i] + dueling[i+1] + dueling[i+2] + dueling[i+3] + dueling[i+4])/5)
#
# # # avg_breakout = []
# # for i in range(0, len(dueling), 5):
# #     avg_dueling.append((dueling[i] + dueling[i+1] + dueling[i+2] + dueling[i+3] + dueling[i+4])/5)
# # #
avg_double = []

for i in range(0, len(double), 5):
    avg_double.append((double[i] + double[i+1] + double[i+2] + double[i+3] + double[i+4])/5)
#
# avg_vanilla = []
#
# for i in range(0, len(vanilla) - 5, 5):
#     avg_vanilla.append((vanilla[i] + vanilla[i+1] + vanilla[i+2] + vanilla[i+3] + vanilla[i+4])/5)


# double = avg_double[0:300]
# dueling = avg_dueling[0:300]
# vanilla = avg_vanilla[0:300]

# double = double[0:500]
# dueling = dueling[0:500]
# vanilla = vanilla[0:500]
#
plt.plot(range(len(avg_dueling)), avg_dueling, label="Tuned")
plt.plot(range(len(avg_double)), avg_double, label="Default")
# plt.plot(range(len(avg_vanilla)), avg_vanilla, label="Vanilla")

# max(breakout)
# plt.hist([breakout])
# plt.hist(avg_double, label="Double", stacked=True)
# plt.hist(avg_vanilla, label="Vanilla", stacked=True)

# breakout.sort()
# plt.hist(breakout, bins=30)
# print('average dueling = ' + str(sum(dueling)/len(dueling)))
# print('average double = ' + str(sum(double)/len(double)))
# print('average vanilla = ' + str(sum(vanilla)/len(vanilla)))
#
print('tuned  = ' + str(len([x for x in dueling if x >= 199])))
print('double = ' + str(len([x for x in double if x >= 199])))
# print('vanilla = ' + str(len([x for x in vanilla if x >= 199])))
# print('greater than 432 = ' + str(len([x for x in breakout if x >= 432])))
# print('max = ' + str(max(breakout)))
# print('min = ' + str(min(breakout)))

# plt.xlabel("Train Frequency")
# plt.ylabel("Number of episodes with rewards >= 199")
plt.legend()
plt.ylabel("Average reward over 5 episodes")
plt.xlabel("Five episode steps")
plt.show()
