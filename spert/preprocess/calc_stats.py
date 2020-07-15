import json
import argparse


if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('--input_path', type=str)
	config = arg_parser.parse_args()

	with open(config.input_path, 'r') as f:
		ds = json.load(f)

	for example in ds:
		pass

