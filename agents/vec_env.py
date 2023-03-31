import multiprocessing as mp
import os
from typing import Any, Callable, List, Optional, Dict, Union, Type

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnv, CloudpickleWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


def _worker(
		remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
	# Import here to avoid a circular import
	from stable_baselines3.common.env_util import is_wrapped

	parent_remote.close()
	env = env_fn_wrapper.var()
	while True:
		try:
			cmd, data = remote.recv()
			if cmd == 'step':
				observation, reward, done, info = env.step(data)
				if done:
					# save final observation where user can get it, then reset
					info['terminal_observation'] = observation
					observation = env.reset()
				remote.send((observation, reward, done, info))
			elif cmd == 'seed':
				remote.send(env.seed(data))
			elif cmd == 'reset':
				observation = env.reset()
				remote.send(observation)
			elif cmd == "render":
				remote.send(env.render(data))
			elif cmd == "close":
				env.close()
				remote.close()
				break
			elif cmd == "get_spaces":
				remote.send((env.observation_space, env.action_space))
			elif cmd == "env_method":
				method = getattr(env, data[0])
				remote.send(method(*data[1], **data[2]))
			elif cmd == "get_attr":
				remote.send(getattr(env.env, data))
			elif cmd == "set_attr":
				remote.send(setattr(env.env, data[0], data[1]))
			elif cmd == "is_wrapped":
				remote.send(is_wrapped(env, data))
			else:
				raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
		except EOFError:
			break


class SubprocVecEnvModified(SubprocVecEnv):
	"""
	Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
	process, allowing significant speed up when the environment is computationally complex.

	For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
	number of logical cores on your CPU.

	.. warning::

		Only 'forkserver' and 'spawn' start methods are thread-safe,
		which is important when TensorFlow sessions or other non thread-safe
		libraries are used in the parent (see issue #217). However, compared to
		'fork' they incur a random_TNs start-up cost and have restrictions on
		global variables. With those methods, users must wrap the code in an
		``if __name__ == "__main__":`` block.
		For more information, see the multiprocessing documentation.

	:param env_fns: Environments to run in subprocesses
	:param start_method: method used to start the subprocesses.
		   Must be one of the methods returned by multiprocessing.get_all_start_methods().
		   Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
	"""

	def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
		self.waiting = False
		self.closed = False
		n_envs = len(env_fns)

		if start_method is None:
			# Fork is not a thread safe method (see issue #217)
			# but is more user friendly (does not require to wrap the code in
			# a `if __name__ == "__main__":`)
			forkserver_available = "forkserver" in mp.get_all_start_methods()
			start_method = "forkserver" if forkserver_available else "spawn"
		ctx = mp.get_context(start_method)

		self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
		self.processes = []
		for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
			args = (work_remote, remote, CloudpickleWrapper(env_fn))
			# daemon=True: if the main process crashes, we should not cause things to hang
			process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
			process.start()
			self.processes.append(process)
			work_remote.close()

		self.remotes[0].send(("get_spaces", None))
		observation_space, action_space = self.remotes[0].recv()
		VecEnv.__init__(self, len(env_fns), observation_space, action_space)


class DummyVecEnvModified(DummyVecEnv):
	"""
	Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
	Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
	as the overhead of multiprocess or multithread outweighs the environment computation time.
	This can also be used for RL methods that
	require a vectorized environment, but that you want a single environments to train with.

	:param env_fns: a list of functions
		that return environments to vectorize
	"""

	def __init__(self, env_fns: List[Callable[[], gym.Env]]):
		super().__init__(env_fns=env_fns)

	# def update_scale(self, scale) -> None:
	#     for env in self.envs:
	#         env.update_scale(scale)
	#
	# def update_best_result(self, solution) -> None:
	#     for env in self.envs:
	#         env.update_best_result(solution)
	#
	# def get_best_result(self) -> None:
	#     return [env.best_result for env in self.envs]

	def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
		"""Return attribute from vectorized environment (see base class)."""
		target_envs = self._get_target_envs(indices)
		return [getattr(env_i.env, attr_name) for env_i in target_envs]

	def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
		"""Set attribute inside vectorized environments (see base class)."""
		target_envs = self._get_target_envs(indices)
		for env_i in target_envs:
			setattr(env_i.env, attr_name, value)


def make_vec_env(
		env_id: Union[str, Callable[..., gym.Env]],
		n_envs: int = 1,
		seed: Optional[int] = None,
		start_index: int = 0,
		monitor_dir: Optional[str] = None,
		wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
		env_kwargs: Optional[Dict[str, Any]] = None,
		vec_env_cls: Optional[Type[Union[DummyVecEnvModified, SubprocVecEnvModified]]] = None,
		vec_env_kwargs: Optional[Dict[str, Any]] = None,
		monitor_kwargs: Optional[Dict[str, Any]] = None,
		wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
	"""
	Create a wrapped, monitored ``VecEnv``.
	By default it uses a ``DummyVecEnv`` which is usually faster
	than a ``SubprocVecEnv``.

	:param env_id: either the env ID, the env class or a callable returning an env
	:param n_envs: the number of environments you wish to have in parallel
	:param seed: the initial seed for the random number generator
	:param start_index: start rank index
	:param monitor_dir: Path to a folder where the monitor files will be saved.
		If None, no file will be written, however, the env will still be wrapped
		in a Monitor wrapper to provide additional information about training.
	:param wrapper_class: Additional wrapper to use on the environment.
		This can also be a function with single argument that wraps the environment in many things.
		Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
		if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
		See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
	:param env_kwargs: Optional keyword argument to pass to the env constructor
	:param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
	:param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
	:param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
	:param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
	:return: The wrapped environment
	"""
	env_kwargs = {} if env_kwargs is None else env_kwargs
	vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
	monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
	wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

	def make_env(rank):
		def _init():
			if isinstance(env_id, str):
				env = gym.make(env_id, **env_kwargs)
			else:
				env = env_id(**env_kwargs)
			if seed is not None:
				env.seed(seed + rank)
				env.action_space.seed(seed + rank)
			# Wrap the env in a Monitor wrapper
			# to have additional training information
			monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
			# Create the monitor folder if needed
			if monitor_path is not None:
				os.makedirs(monitor_dir, exist_ok=True)
			env = Monitor(env, filename=monitor_path, **monitor_kwargs)
			# Optionally, wrap the environment with the provided wrapper
			if wrapper_class is not None:
				env = wrapper_class(env, **wrapper_kwargs)
			return env

		return _init

	# No custom VecEnv is passed
	if vec_env_cls is None:
		# Default: use a DummyVecEnv
		vec_env_cls = DummyVecEnvModified

	return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
