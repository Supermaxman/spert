
from pathlib import Path
import collections
import json
import itertools
from tqdm import tqdm

import spacy

nlp = spacy.load('en_core_sci_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def find_entities(start, end, entities):
	es = []
	for e in entities:
		if start <= e['start'] and e['end'] <= end:
			es.append(e)
	return es


def fix_types(entity_type):
	if entity_type == 'CHEMICAL':
		return 'Chemical'
	elif entity_type == 'GENE-Y':
		return 'Gene'
	elif entity_type == 'GENE-N':
		return 'Gene'
	else:
		raise NotImplementedError(f'Unknown entity type: {entity_type}')


def tokenize_text(doc_id, text, entities, relations):
	sentences = []
	doc = nlp(text)
	for s_idx, sent in enumerate(doc.sents):
		sent_entities = find_entities(sent.start_char, sent.end_char, entities)
		sent_tokens = [token.text for token in sent]
		for entity_pos, entity in enumerate(sent_entities):
			entity['sent_id'] = entity_pos

		sent_relations = set()
		# TODO consider other orderings
		for head, tail in itertools.product(sent_entities, sent_entities):
			for rel_label in relations[head['id']][tail['id']]:
				sent_relations.add((head['sent_id'], tail['sent_id'], rel_label))
		s = Sentence(
			f'D{doc_id}S{s_idx}',
			sent_tokens,
			sent_entities,
			sent_relations
		)
		sentences.append(s)

	return sentences


class Split:
	def __init__(self, path: Path):
		self.sentences = []
		abs_path = path / 'abstracts.tsv'
		entity_path = path / 'entities.tsv'
		gs_path = path / 'gold_standard.tsv'
		# TODO clean up and generalize
		entities = collections.defaultdict(list)
		with entity_path.open('r') as f:
			for line in f:
				segments = line.strip().split('\t')
				entities[segments[0]].append(
					{
						'doc_id': segments[0],
						'start': int(segments[3]),
						'end': int(segments[4]),
						'type': fix_types(segments[2]),
						'id': segments[1],
						'text': segments[5]
					}
				)

		#[doc_id, arg1, arg2, list of labels]
		relations = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
		with gs_path.open('r') as f:
			for line in f:
				segments = line.strip().split('\t')
				doc_id = segments[0]
				label = segments[1]
				arg1 = segments[2][segments[2].find(':') + 1:]
				arg2 = segments[3][segments[3].find(':') + 1:]
				relations[doc_id][arg1][arg2].append(label)

		self.sentences = []
		with abs_path.open('r') as f:
			for line in tqdm(f):
				segments = line.strip().split('\t')
				text = segments[1] + ' ' + segments[2]
				doc_id = segments[0]
				sents = tokenize_text(doc_id, text, entities[doc_id], relations[doc_id])
				self.sentences.extend(sents)

	def to_dict(self):
		return [s.to_dict() for s in self.sentences]


class Sentence:
	def __init__(self, s_id, tokens, entities, relations):
		self.s_id = s_id
		self.tokens = tokens
		self.entities = entities
		self.relations = relations

	def to_dict(self):
		entities = []
		for entity in self.entities:
			entity_dict = {
				'type': entity['type'],
				'start': entity['start'],
				'end': entity['end']
			}
			entities.append(entity_dict)
		relations = []
		for relation in self.relations:
			relation_dict = {
				'type': relation[2]['type'],
				'head': relation[0],
				'tail': relation[1]
			}
			relations.append(relation_dict)
		sent_dict = {
			'tokens': self.tokens,
			'entities': entities,
			'relations': relations,
			'orig_id': self.s_id
		}
		return sent_dict


if __name__ == '__main__':
	inputs_path = Path('/users/max/data/corpora/ChemProt/original')
	outputs_path = Path('/users/max/data/corpora/ChemProt/json')
	splits = ['train', 'dev', 'test']
	if not outputs_path.exists():
		outputs_path.mkdir()

	for split in splits:
		split_input_path = inputs_path / split
		split_output_path = (outputs_path / split).with_suffix('.json')
		s = Split(split_input_path)
		split_dict = s.to_dict()
		with split_output_path.open('w') as fp:
			json.dump(
				split_dict,
				fp,
				indent=2
			)

