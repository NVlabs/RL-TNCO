import pickle

import matplotlib.pyplot as plt
import numpy as np

filename = '../data/TN_dataset_nun_eqs_100_num_node_20_mean_conn_3.p'
with open(filename, 'rb') as f:
	x = pickle.load(f)

labels = list()
average_times = list()
average_costs = list()
std_times = list()
std_costs = list()
for baseline, results in x[1].items():
	labels.append(baseline)
	times = list()
	costs = list()
	for ii in range(len(results)):
		costs.append(float(results[ii][0]))
		times.append(float(results[ii][1]))
	average_costs.append(sum(costs) / len(costs))
	average_times.append(sum(times) / len(times))
	std_times = np.std(np.array(times))
	std_costs = np.std(np.array(costs))
fig = plt.figure()
plt.bar(labels, average_costs, color='maroon',
        width=0.4)
plt.title('costs')
plt.tight_layout()
# plt.show()
plt.savefig('costs.png')
fig = plt.figure()
plt.bar(labels, average_times, color='maroon',
        width=0.4)
plt.title('times')
plt.yscale('log')
plt.tight_layout()
plt.savefig('times.png')

# plt.show()
