
import os
import argparse
import json


def format_models(model_name):
	if split is not None:
		return f'{model_name}-{split}-{seed}-eval'
	else:
		return f'{model_name}-{seed}-eval'


def load_model_predictions(model_path, full_model_name):
	preds = json.load(open(os.path.join(model_path, full_model_name)))
	return preds


def compare_example(label, pred):
	return all(label_value == pred_value for label_value, pred_value in zip(label.values(), pred.values()))


def compare_examples(label, pred):
	if len(label) != len(pred):
		return False
	return all(compare_example(label_example, pred_example) for label_example, pred_example in zip(label, pred))


def compare(label, pred):
	entity_correct = compare_examples(label['entities'], pred['entities'])
	rel_correct = compare_examples(label['relations'], pred['relations'])
	return entity_correct and rel_correct


def compare_results(label_list, correct_pred_list, incorrect_pred_list, correct_model_list, incorrect_model_list):
	results = []
	for label, correct_preds, incorrect_preds in zip(label_list, zip(*correct_pred_list), zip(*incorrect_pred_list)):
		match = True
		result = {
			'label': label
		}
		for c_pred, c_name in zip(correct_preds, correct_model_list):
			match = match and compare(label, c_pred)
			result[c_name] = c_pred
			del c_pred['tokens']
		for i_pred, i_name in zip(incorrect_preds, incorrect_model_list):
			match = match and not compare(label, i_pred)
			result[i_name] = i_pred
			del i_pred['tokens']
		if match:
			results.append(result)
	return results


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()

	arg_parser.add_argument(
		'--label_path', type=str, help="Path to labels", default='/users/max/data/corpora/i2b2/2010/json/test.json')
	arg_parser.add_argument('--model_path', type=str, default='/shared/hltdir4/disk1/max/logs/spert/')
	arg_parser.add_argument('--output_path', type=str, default='results/results.json')
	arg_parser.add_argument('--seed', type=str, default=0)
	arg_parser.add_argument('--split', type=str, default=None)
	arg_parser.add_argument(
		'--correct_model_list', type=str, help="List of model which get sentence right.", default='i2b2-biobert')
	arg_parser.add_argument(
		'--incorrect_model_list', type=str, help="List of models which get sentence wrong.", default='i2b2-bert-base')

	args = arg_parser.parse_args()

	label_path = args.label_path
	model_path = args.model_path
	output_path = args.output_path
	seed = args.seed
	split = args.split
	correct_model_list = [format_models(name) for name in args.correct_model_list.split(',')]
	incorrect_model_list = [format_models(name) for name in args.incorrect_model_list.split(',')]

	labels = json.load(open(label_path))
	correct_preds = [load_model_predictions(model_path, full_model_name) for full_model_name in correct_model_list]
	incorrect_preds = [load_model_predictions(model_path, full_model_name) for full_model_name in incorrect_model_list]

	results = compare_results(labels, correct_preds, incorrect_preds, correct_model_list, incorrect_model_list)

	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)

