def read_data_file(config):
	eval_file = config['network'].get('test_files', None)
	train_file = config['network'].get('train_files', None)
	return train_file, eval_file
