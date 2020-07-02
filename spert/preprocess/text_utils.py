

def find_entities(start, end, entities):
	es = []
	for e in entities:
		if start <= e.start and e.end <= end:
			es.append(e)
	return es


def find_relations(entities, relations):
	entity_id_set = set([e.entity_id for e in entities])
	kept_relations = []
	# TODO probably should print skipped cross-sentence relations
	for relation in relations:
		if relation.head.entity_id in entity_id_set and relation.tail.entity_id in entity_id_set:
			kept_relations.append(relation)
	return kept_relations


class Sentence:
	def __init__(self, sent_id, tokens, entities, relations):
		self.sent_id = sent_id
		self.entities = entities
		self.relations = relations

		self.tokens = []
		for entity_pos, entity in enumerate(self.entities):
			entity.sent_pos = entity_pos
			entity.tokens = []

		start_entity_idx = 0
		for token in tokens:
			start = token.idx
			length = len(token.text)
			entity_idx = start_entity_idx
			while entity_idx < len(self.entities):
				current_entity = self.entities[entity_idx]
				# if span is past token then move to next token and start checking from start_span_idx again forward.
				if current_entity.start >= start + length:
					break
				# if span end is before token then stop checking span since all further tokens cannot be in span due to
				# ordering of word piece tokens by start
				elif current_entity.end <= start:
					start_entity_idx += 1
					entity_idx += 1
					continue
				# there must be some overlap between the current span and the current token
				else:
					current_entity.tokens.append(len(self.tokens))
					entity_idx += 1
			self.tokens.append(token.text)

		for entity in self.entities:
			entity.sent_start = entity.tokens[0]
			entity.sent_end = entity.tokens[-1] + 1

	def to_dict(self):
		entities = []
		for entity in self.entities:
			entity_dict = {
				'type': entity.entity_type,
				'start': entity.sent_start,
				'end': entity.sent_end
			}
			entities.append(entity_dict)
		relations = []
		for relation in self.relations:
			relation_dict = {
				'type': relation.rel_type,
				'head': relation.rel_type.head.sent_pos,
				'tail': relation.rel_type.tail.sent_pos
			}
			relations.append(relation_dict)
		sent_dict = {
			'tokens': self.tokens,
			'entities': entities,
			'relations': relations,
			'orig_id': self.sent_id
		}
		return sent_dict


class Segment:
	def __init__(self, seg_id, entities, relations, sentences):
		self.seg_id = seg_id
		self.entities = entities
		self.relations = relations
		self.sentences = []
		for s_idx, sent in enumerate(sentences):
			sent_entities = sorted(find_entities(sent.start_char, sent.end_char, self.entities), key=lambda x: x.start)
			sent_relations = find_relations(sent_entities, self.relations)

			sentence = Sentence(
				sent_id=f'D{seg_id}S{s_idx}',
				tokens=sent,
				entities=sent_entities,
				relations=sent_relations
			)

			self.sentences.append(sentence)


class Relation:
	def __init__(self, head, tail, rel_type):
		self.head = head
		self.tail = tail
		self.rel_type = rel_type


class Entity:
	def __init__(self, entity_id, entity_type, start, end):
		self.entity_id = entity_id
		self.entity_type = entity_type
		self.start = start
		self.end = end
