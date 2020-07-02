
from pathlib import Path
import collections
import json
from tqdm import tqdm
import spacy

from spert.preprocess.text import Sentence, Segment, Relation, Entity


def fix_types(entity_type):
	if entity_type == 'CHEMICAL':
		return 'Chemical'
	elif entity_type == 'GENE-Y':
		return 'Gene'
	elif entity_type == 'GENE-N':
		return 'Gene'
	else:
		raise NotImplementedError(f'Unknown entity type: {entity_type}')


def read_sentences(path: Path):
	abs_path = path / 'abstracts.tsv'
	entity_path = path / 'entities.tsv'
	gs_path = path / 'gold_standard.tsv'
	# TODO clean up and generalize
	entities = collections.defaultdict(list)
	entity_dict = {}
	with entity_path.open('r') as f:
		for line in f:
			segments = line.strip().split('\t')
			seg_id = segments[0]
			entity = Entity(
				entity_id=segments[1],
				entity_type=fix_types(segments[2]),
				start=int(segments[3]),
				end=int(segments[4]),
			)
			entity_dict[segments[1]] = entity
			entities[seg_id].append(entity)

	relations = collections.defaultdict(list)
	with gs_path.open('r') as f:
		for line in f:
			segments = line.strip().split('\t')
			seg_id = segments[0]
			relation = Relation(
				head=entity_dict[segments[2][segments[2].find(':') + 1:]],
				tail=entity_dict[segments[3][segments[3].find(':') + 1:]],
				rel_type=segments[1]
			)
			relations[seg_id].append(relation)

	nlp = spacy.load('en_core_sci_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
	nlp.add_pipe(nlp.create_pipe('sentencizer'))

	sentences = []
	with abs_path.open('r') as f:
		for line in tqdm(f):
			segments = line.strip().split('\t')
			text = segments[1] + ' ' + segments[2]
			seg = nlp(text)
			seg_id = segments[0]
			doc = Segment(
				seg_id=seg_id,
				entities=entities[seg_id],
				relations=relations[seg_id],
				sentences=seg.sents
			)
			sentences.extend(doc.sentences)

	return sentences


if __name__ == '__main__':
	inputs_path = Path('/users/max/data/corpora/ChemProt/original')
	outputs_path = Path('/users/max/data/corpora/ChemProt/json')
	splits = ['train', 'dev', 'test']
	if not outputs_path.exists():
		outputs_path.mkdir()

	for split in splits:
		split_input_path = inputs_path / split
		split_output_path = (outputs_path / split).with_suffix('.json')

		print(f'Reading split {split_input_path}...')
		sentences = read_sentences(split_input_path)
		split_dict = [s.to_dict() for s in sentences]
		stats = collections.defaultdict(int)
		for sentence in sentences:
			stats['entities'] += len(sentence.entities)
			stats['relations'] += len(sentence.relations)

		for stat, count in stats.items():
			print(f'{stat}: {count}')

		with split_output_path.open('w') as fp:
			json.dump(
				split_dict,
				fp,
				indent=2
			)

