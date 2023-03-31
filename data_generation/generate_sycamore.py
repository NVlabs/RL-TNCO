import pickle

from generate_data import solve_eq, getSystemInfo
from generate_quantum_tensor_networks import QuantumTensorNetwork

layout = 'sycamore'

tn = QuantumTensorNetwork(layout=layout)

circuit_depth = 10

tn.generate_random_quantum_circuit(circuit='ABCDCDAB', circuit_depth=circuit_depth)

# Get the einsum equation

eq = tn.get_equation()
input_tensors, output_tensor = str.split(eq, '->')
input_tensors = str.split(input_tensors, ',')
# construct shapes:
shapes = []
for tensor in input_tensors:
	shapes.append(tuple([2] * len(tensor)))
print('{} tensors'.format(len(input_tensors)))

# Draw the tensor network
# plt.figure()
# tn.draw()
# plt.show()
# contraction_cost, total_time, info, path = solve_eq(eq=eq,shapes=shapes,baseline='oe_greedy')
# simplify
tn = tn.simplify()
eq = tn.get_equation()
input_tensors, output_tensor = str.split(eq, '->')
input_tensors = str.split(input_tensors, ',')
shapes = []
for tensor in input_tensors:
	shapes.append(tuple([2] * len(tensor)))
print('{} tensors'.format(len(input_tensors)))

tensors = list(set(input_tensors))
size_dict = dict()
for t in tensors:
	chars = list(set(t))
	for c in chars:
		size_dict[c] = 2

# tn.draw()
# plt.show()
contraction_cost, total_time, info, path = solve_eq(eq=eq, shapes=shapes, baseline='oe_greedy')

# save

eq_list = list()
baseline_list = ['oe_greedy']

solution_dict = dict()
for baseline in baseline_list:
	solution_dict[baseline] = list()

eq_list.append((eq, shapes, size_dict))
for baseline in baseline_list:
	contraction_cost, solver_time, info, path = solve_eq(eq, shapes, baseline=baseline)
	solution_dict[baseline].append((contraction_cost, solver_time, info, path))
sys_info = getSystemInfo()
pickle.dump((eq_list, solution_dict, sys_info),
            open("simplified_sycamore_53_d_{}.p".format(circuit_depth),
                 "wb"))
