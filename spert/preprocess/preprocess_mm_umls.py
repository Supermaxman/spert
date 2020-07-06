from pathlib import Path
import collections
import json
from tqdm import tqdm
import spacy
import itertools

from . import text_utils
from . import umls_reader
from . import umls


def read_sentences(path: Path, umls_rel_lookup):
  name = path.name.replace('.txt', '')
  ann_path = path.with_suffix('.ann')

  entities = []
  entity_dict = {}
  umls_entity_dict = {}
  with ann_path.open('r') as fp:
    lines = [text.strip() for text in fp]
    entity_lines = [text for text in lines if text.startswith('T')]
    umls_lines = [text for text in lines if text.startswith('#')]
    for line in entity_lines:
      # T<entity_id> \TAB Concept \SPACE <start> \SPACE <end> \TAB <text>
      entity_id, entity_info, _ = line.strip().split('\t')
      entity_id = entity_id[1:]
      # all entities will have entity type "Concept" for now
      entity_type, start, end = entity_info.split()
      entity = text_utils.Entity(
        entity_id=entity_id,
        entity_type=entity_type,
        start=int(start),
        end=int(end),
      )
      entity_dict[entity_id] = entity
      entities.append(entity)
    for line in umls_lines:
      # #<entity_id> \TAB AnnotatorNotes \SPACE T<entity_id> \TAB <CUI>--<UMLS_TYPES_CSV_LIST>
      _, entity_id, umls_info = line.strip().split('\t')
      entity_id = entity_id.split()[1][1:]
      umls_cui, umls_types = umls_info.split('--')
      umls_types = umls_types.split(',')
      entity = entity_dict[entity_id]
      entity.umls_cui = umls_cui
      entity.umls_types = umls_types
      umls_entity_dict[entity.umls_cui] = entity

  nlp = spacy.load('en_core_sci_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
  nlp.add_pipe(nlp.create_pipe('sentencizer'))

  sentences = []
  with path.open('r') as f:
    text = f.read().strip()
    seg = nlp(text)
    doc = text_utils.Segment(
      seg_id=name,
      entities=entities,
      relations=[],
      sentences=seg.sents
    )
    sentences.extend(doc.sentences)

  for sentence in sentences:
    for head, tail in itertools.product(sentence.entities, sentence.entities):
      # ignore reflexive relations,
      # but allow reflexive cui relations, just not same mention
      if head != tail:
        rel_types = umls_rel_lookup[(head.umls_cui, tail.umls_cui)]
        if len(rel_types) > 0:
          for rel_type in rel_types:
            relation = text_utils.Relation(
              head=head,
              tail=tail,
              rel_type=rel_type
            )
            sentence.relations.append(relation)

  return sentences


def read_umls_rel_lookup(path):
  def umls_rel_filter(x):
    # remove reflexive relations
    if x.cui2 == x.cui1:
      return False
    # TODO consider more filters:
    # # ignore siblings, CHD is enough to infer
    # if x.rel == 'SIB':
    #   return False
    # # ignore PAR, CHD is symmetric
    # if x.rel == 'PAR':
    #   return False
    # # ignore RO with no relA, not descriptive
    # if x.rel == 'RO' and x.rela == '':
    #   return False
    # # symmetric with AQ
    # if x.rel == 'QB':
    #   return False
    # # too vague
    # if x.rel == 'RB':
    #   return False
    return True

  lookup = collections.defaultdict(list)
  rel_iter = umls_reader.read_umls(
    path,
    umls.UmlsRelation,
    umls_filter=umls_rel_filter
  )
  for rel in rel_iter:
    # TODO consider types including rela for more detailed relation types
    lookup[(rel.cui2, rel.cui1)].append(rel.rel)
  return lookup


if __name__ == '__main__':
  inputs_path = Path('/users/max/data/corpora/medmentions/MedMentions/full/data/brat/')
  outputs_path = Path('/users/max/data/corpora/medmentions/MedMentions/full/data/json')
  umls_path = Path('/users/max/data/ontologies/umls_2019/2019AA-full/2019AA/META/MRREL.RRF')

  print('Reading umls rels...')
  umls_rel_lookup = read_umls_rel_lookup(umls_path)

  splits = ['train', 'dev', 'test']
  if not outputs_path.exists():
    outputs_path.mkdir()

  for split in splits:
    split_input_path = inputs_path / split
    split_output_path = (outputs_path / split).with_suffix('.json')

    print(f'Reading split {split_input_path}...')
    sentences = []
    for doc_path in tqdm(split_input_path.glob('**/*.txt')):
      doc_sentences = read_sentences(doc_path, umls_rel_lookup)
      sentences.extend(doc_sentences)
    split_dict = [s.to_dict() for s in sentences]
    stats = collections.defaultdict(int)
    for sentence in sentences:
      stats['entities'] += len(sentence.entities)
      for entity in sentence.entities:
        stats[entity.entity_type] += 1
      stats['relations'] += len(sentence.relations)
      for relation in sentence.relations:
        stats[relation.rel_type] += 1

    for stat, count in stats.items():
      print(f'{stat}: {count}')

    with split_output_path.open('w') as fp:
      json.dump(
        split_dict,
        fp,
        indent=2
      )
