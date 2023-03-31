import opt_einsum as oe

from agents.vec_env import make_vec_env
from env.tensorNetworkEnv import TNLearningEnv


# opt_einsum optimizer
class RLGN_optimizer(oe.paths.PathOptimizer):
	def __init__(self, config, model, visualize=None, fixed_eqs_set_size=None, debug=False):
		self.data = None
		self.model = model
		self.env = make_vec_env(
			env_id=lambda: TNLearningEnv(config, fixed_eqs_set_size=config['eval']['eval_episodes'],
			                             evaluation_env=True),
			n_envs=1, seed=config['eval']['eval_seed'])

	def set_eq_and_source(self, source_eq, source_shapes):
		# there are two different equivalent representation that should not be mixed
		# eq + shapes and inputs + size dict
		# eq: 'do,alj,efmgc,kl,k,cngif,do,na,mihjb,heb->'
		# shapes: a list of tuples: [(8, 6), (9, 6, 3), (8, 4, 2, 2, 2),...]

		self.data = dict()
		self.data['eq'] = source_eq
		self.data['shapes'] = source_shapes

	def __call__(self, inputs, output, size_dict, memory_limit=None):
		# there are two different equivalent representation that should not be mixed. The first part aligns the two
		# eq + shapes and inputs + size dict
		# eq: 'do,alj,efmgc,kl,k,cngif,do,na,mihjb,heb->'
		# shapes: a list of tuples: [(8, 6), (9, 6, 3), (8, 4, 2, 2, 2),...]

		# inputs: a list of sets : [{'d', 'o'}, {'a', 'j', 'l'},
		# size_dict: a dictionary of sizes: [{'d': 8, 'o': 6, 'a': 9,,...]
		output = tuple(output)
		tensors = [tuple(x) for x in inputs]
		shapes = []
		for t in tensors:
			shapes.append(tuple(size_dict[x] for x in t))
		eq = [''.join(t) for t in tensors]
		eq = ','.join(eq) + '->' + ''.join(output)
		self.data = {'eq': eq,
		             'shapes': shapes,
		             'tensors': tensors,
		             'size_dict': size_dict}
		self.env.envs[0].env.data = self.data
		done = False
		observations = self.env.reset()
		while not done:
			episode_history = self.env.envs[0].env.episode_history
			actions, model_info = self.model.predict(observations, deterministic=True)
			observations, rewards, done, infos = self.env.step(actions)
		return episode_history
