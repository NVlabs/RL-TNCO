import matplotlib as mpl
import networkx as nx
import quimb as qu
import quimb.tensor as qtn
import cotengra as ctg
from random import randrange

mpl.rcParams['figure.dpi'] = 200


class QuantumTensorNetwork:
	def __init__(self, layout='sycamore', Lx=None, Ly=None):
		"""parameters:
        num_qubits: Number of qubits. For example 53 for Sycamore layout.
        layout: 'sycamore', 'rectangular'
        """
		if layout == 'sycamore':
			self.N = 53  # Number of qubits, 53 for Sycamore
		elif layout == 'rectangular':
			self.Lx, self.Ly = Lx, Ly
			self.N = Lx * Ly  # Number of qubits for rectangular lattice

		self.m = 1  # circuit_depth
		self.TN = None
		self.TN_to_plot = None
		self.contr = None
		self.layout = layout
		self.gate_tags = []
		self.site_tags = []
		self.round_tags = []

	def contract(self, methods=None, backend='jax', sliced=False, max_repeats=128, progbar=True, minimize='flops',
	             score_compression=0.5, visualize_tree=True, plot_cost=False):
		# bit_str='0' * (self.N)

		if methods is None:
			methods = ['greedy', 'kahypar']
		if not sliced:
			# psi_proj = qtn.MPS_computational_state(bit_str)
			tn = self.TN.psi  # & psi_proj
			# output_inds = []
			# cast to single precision
			# tn.full_simplify_(output_inds=output_inds)
			tn.astype_('complex64')
			opt = ctg.HyperOptimizer(methods=methods, max_repeats=max_repeats, progbar=progbar, minimize=minimize,
			                         score_compression=score_compression)
			self.contr = tn.contract(all, optimize=opt, get='path-info')

			tn.contract(all, optimize=opt.path, backend=backend)
		if visualize_tree:
			tree = opt.get_tree()
			tree.plot_ring(node_scale=1 / 4, edge_scale=1 / 2)
		if plot_cost:
			opt.plot_trials()

	def simplify(self):
		# bit_str='0' * (self.N)
		# psi_proj = qtn.MPS_computational_state(bit_str)
		tn = self.TN.psi  # & psi_proj
		output_inds = []
		# cast to single precision
		tn.full_simplify_(output_inds=output_inds)
		self.TN = tn
		return self.TN

	def generate_random_peps(self, circuit_depth=1, bond_dim=4, seed=455):
		# Note: self.N must have a root
		self.m = circuit_depth
		self.TN = qtn.PEPS.rand(Lx=self.Lx, Ly=self.Ly, bond_dim=bond_dim)
		if self.m > 1:
			self.TN = self.TN & qtn.PEPS.rand(Lx=self.Lx, Ly=self.Ly, bond_dim=bond_dim)
		self.TN_to_plot = self.TN

	def generate_random_mera(self, circuit_depth=1):
		self.m = circuit_depth
		self.TN = qtn.MERA.rand_invar(self.N)
		if self.m > 1:
			self.TN = self.TN & qtn.MERA.rand_invar(self.N)
		self.TN_to_plot = self.TN

	def generate_random_qaoa(self, circuit_depth=10, reg=3, seed=745):
		"""parameters:
        circuit_depth: Circuit depth of QAOA, an even number equal to twice
        the p parameter in QAOA algorithm, where 2p is the circuit depth.
        reg: Graph regularity

        """
		p = int(circuit_depth / 2)
		self.m = circuit_depth

		G = nx.random_regular_graph(reg, self.N, seed=seed)
		terms = {(i, j): 1 for i, j in G.edges}
		# initialize gamma, beta params
		gammas = qu.randn(p)
		betas = qu.randn(p)
		self.TN = qtn.circ_qaoa(terms, p, gammas, betas)
		self.TN_to_plot = self.TN.get_uni()
		self.gate_tags = ['PSI0', 'H', 'RZZ', 'RX']

	def generate_random_quantum_circuit(self, circuit='ABCDCDAB', circuit_depth=10, load=True, seed=0, elided=0,
	                                    swap_trick=False):
		"""
        parameters:
        circuit_depth: current files simulated for Sycamore 10, 12, 20

        methods:
        'ABCDCDAB' generates/loads circuits based on arXiv:2005.06787v1
        'cz' generates/loads circuits based on Nature, 574, 505–510 (2019) using CZ gates
        'iswap' generates/loads circuits based on Nature, 574, 505–510 (2019) using iSWAP gates

        """
		# Note: self.N and self.m must be set according to .qsim circuit data file
		self.m = circuit_depth
		if circuit == 'ABCDCDAB':
			file = f'../data/sycamore/circuit_n{self.N}_m{self.m}_s{seed}_e{elided}_pABCDCDAB.qsim'

			if swap_trick:
				gate_opts = {'contract': 'swap-split-gate', 'max_bond': 2}
			else:
				gate_opts = {}

			self.TN = qtn.Circuit.from_qasm_file(file, gate_opts=gate_opts)
			self.TN_to_plot = self.TN.get_uni()

		elif circuit == 'cz':
			dim_str = str(self.Lx) + 'x' + str(self.Ly)
			self.TN = qtn.Circuit.from_qasm_file(
				'rectangular/cz_v2/' + dim_str + '/inst_' + dim_str + '_' + str(circuit_depth) + '_' + str(
					randrange(10)) + '.txt', tags='PSI_i')

			self.TN_to_plot = self.TN.get_uni()
			self.gate_tags = ['PSI_i', 'H', 'CZ', 'T', 'X_1/2', 'Y_1/2',
			                  'PSI_f']  # self.site_tags = [c_tn.site_tag(i) for i in range(c_tn.nsites)]  #
			# self.round_tags = ['PSI_i'] + ["ROUND_{}".format(i) for i in range(circuit_depth)]
		elif circuit == 'iswap':
			dim_str = str(self.Lx) + 'x' + str(self.Ly)
			self.TN = qtn.Circuit.from_qasm_file(
				'rectangular/is_v1/' + dim_str + '/inst_' + dim_str + '_' + str(circuit_depth) + '_' + str(
					randrange(10)) + '.txt', tags='PSI_i')

			self.TN_to_plot = self.TN.get_uni()
			self.gate_tags = ['PSI_i', 'H', 'CZ', 'T', 'X_1/2', 'Y_1/2',
			                  'PSI_f']  # self.site_tags = [c_tn.site_tag(i) for i in range(c_tn.nsites)]  #
			# self.round_tags = ['PSI_i'] + ["ROUND_{}".format(i) for i in range(circuit_depth)]

	def get_equation(self):
		return self.TN_to_plot.get_equation()

	def time_evolve(self):
		pass

	def get_TN(self):
		if self.TN is not None:
			return self.TN
		else:
			print("Tensor network not generated.")

	def draw(self):
		self.TN.psi.draw(color=self.gate_tags)
