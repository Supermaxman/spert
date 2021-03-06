from pathlib import Path
import collections
import json
from tqdm import tqdm


from text_utils import Sentence, Relation, Entity


def fix_types(entity_type):
  if entity_type == 'problem':
    return 'Problem'
  elif entity_type == 'test':
    return 'Test'
  elif entity_type == 'treatment':
    return 'Treatment'
  # filler for relation parsing
  elif entity_type == 'dummy':
    return 'dummy'
  else:
    raise NotImplementedError(f'Unknown entity type: {entity_type}')


def create_entity(line):
  span_info, type_info, attr_info = line.strip().split('||')
  fields = span_info.split(' ')
  start, end = fields[-2], fields[-1]
  start = start.split(':')
  end = end.split(':')
  sent_id = int(start[0]) - 1
  start = int(start[1])
  end = int(end[1]) + 1
  entity_type = type_info.strip()[3:-1]
  assertion = attr_info.strip()[3:-1]

  entity = Entity(
    entity_id=hash((sent_id, start, end)),
    entity_type=fix_types(entity_type),
    start=start,
    end=end,
  )
  entity.assertion = assertion

  return entity, sent_id


def read_sentences(path: Path):
  name = path.name.replace('.txt', '')
  ann_path = path.with_suffix('.ann')

  entities = collections.defaultdict(list)
  entity_dict = {}
  with ann_path.open('r') as fp:
    lines = fp.readlines()
    entity_lines = [line for line in lines if line.startswith('T')]
    rel_lines = [line for line in lines if line.startswith('R')]
    for line in entity_lines:
      entity, sent_id = create_entity(line)
      if entity.entity_id in entity_dict:
        continue
      entities[sent_id].append(entity)
      entity_dict[entity.entity_id] = entity

  relations = collections.defaultdict(list)
  if rel_path.exists():
    with rel_path.open('r') as fp:
      for line in fp:
        arg1, type_info, arg2 = line.split('||')
        head, head_sent_id = create_entity(arg1 + '||t="dummy"||a="dummy"')
        tail, tail_sent_id = create_entity(arg2 + '||t="dummy"||a="dummy"')
        head = entity_dict[head.entity_id]
        tail = entity_dict[tail.entity_id]
        rel_type = type_info[3:-1]
        if 'INV$' in rel_type:
          rel_type = rel_type.replace('INV$', '')
          head, tail = tail, head
          head_sent_id, tail_sent_id = tail_sent_id, head_sent_id
        relation = Relation(
          head=head,
          tail=tail,
          rel_type=rel_type
        )
        # TODO assert head_sent_id == tail_sent_id or skip / log cross-sentence rels
        relations[head_sent_id].append(relation)

  sentences = []
  with path.open('r') as fp:
    for sent_id, line in enumerate(fp):
      tokens = []
      for w in line.strip().split(' '):
        if len(w.strip()) > 0:
          tokens.append(Token(w, len(tokens)))
      sent_entities = sorted(entities[sent_id], key=lambda x: x.start)
      sent_rels = sorted(relations[sent_id], key=lambda x: x.head.start)

      sentence = Sentence(
        sent_id=f'D{name}S{sent_id}',
        tokens=tokens,
        entities=sent_entities,
        relations=sent_rels
      )
      sentences.append(sentence)

  return sentences


class Token:
  def __init__(self, text, idx):
    self.text = text
    self.idx = idx

  def __len__(self):
    return 1


if __name__ == '__main__':
  inputs_path = Path('/users/max/data/corpora/n2c2/2018/track2')
  outputs_path = Path('/users/max/data/corpora/i2b2/2010/json')
  # TODO construct dev set
  splits = ['train', 'test']
  if not outputs_path.exists():
    outputs_path.mkdir()

  for split in splits:
    split_input_path = inputs_path / split
    split_output_path = (outputs_path / split).with_suffix('.json')

    print(f'Reading split {split_input_path}...')
    sentences = []
    for doc_path in tqdm(split_input_path.glob('**/*.txt')):
      doc_sentences = read_sentences(doc_path)
      sentences.extend(doc_sentences)
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
