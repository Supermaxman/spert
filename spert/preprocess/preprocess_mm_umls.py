from pathlib import Path
import collections
import json
from tqdm import tqdm
import spacy
import itertools

from . import text_utils
from . import umls_reader
from . import umls


def read_sentences(path: Path, nlp, umls_rel_lookup):
  sentences = []
  with path.open('r') as fp:
    doc_id = None
    title = None
    text = None
    entities = []
    for line in fp:
      line = line.strip()
      if not line:
        seg = nlp(f'{title}. {text}')
        doc = text_utils.Segment(
          seg_id=doc_id,
          entities=entities,
          relations=[],
          sentences=seg.sents
        )
        sentences.extend(doc.sentences)
        doc_id = None
        title = None
        text = None
        entities = []
      else:
        if title is None:
          # doc_id, _, title
          segs = line.split('|')
          doc_id = segs[0]
          title = '|'.join(segs[2:])
          title = title.strip()
        elif text is None:
          segs = line.split('|')
          doc_id = segs[0]
          # might be | in text so just in case we rejoin
          text = '|'.join(segs[2:])

          text = text.strip()
        else:
          doc_id, start, end, _, umls_types, umls_cui = line.split('\t')
          start = int(start)
          end = int(end)
          # if this entity is not from the title then we need to add 2 to
          # start and end char positions since we added a period
          # to concatenate the title to the document.
          if start >= len(title):
            start += 1
            end += 1
          umls_type = umls_types.split(',')[0]
          umls_cui = umls_cui.replace('UMLS:', '')
          entity = text_utils.Entity(
            entity_id=len(entities),
            entity_type=umls_type,
            start=int(start),
            end=int(end),
          )
          entity.umls_cui = umls_cui
          entity.umls_types = umls_types
          entities.append(entity)

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


def read_umls_rel_lookup(path, keep_rels):
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
    # too few
    if x.rel == 'SY' or x.rel == 'AQ' or x.rel == 'QB':
      return False
    if x.rel not in keep_rels:
      return False
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
  inputs_path = Path('/users/max/data/corpora/medmentions/MedMentions/full/data/')
  outputs_path = Path('/users/max/data/corpora/medmentions/MedMentions/full/data/json2')

  umls_path = Path('/users/max/data/ontologies/umls_2019/2019AA-full/2019AA/META/MRREL.RRF')

  nlp = spacy.load('en_core_sci_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
  nlp.add_pipe(nlp.create_pipe('sentencizer'))

  print('Reading umls rels...')
  with (outputs_path / 'types.json').open('r') as f:
    type_info = json.load(f)
    keep_rels = set(type_info['relations'].keys())
  print(f'Keep rel types: {keep_rels}')
  umls_rel_lookup = read_umls_rel_lookup(umls_path, keep_rels)

  splits = ['train', 'dev', 'test']
  split_files = {
    'train': 'corpus_pubtator_pmids_trng.txt',
    'dev': 'corpus_pubtator_pmids_dev.txt',
    'test': 'corpus_pubtator_pmids_test.txt'
  }
  if not outputs_path.exists():
    outputs_path.mkdir()

  print('Reading full dataset...')
  all_sentences = read_sentences(inputs_path / 'corpus_pubtator.txt', nlp, umls_rel_lookup)

  for split in splits:
    split_input_path = inputs_path / split_files[split]
    with split_input_path.open('r') as f:
      split_ids = set([x.strip() for x in f])

    split_output_path = (outputs_path / split).with_suffix('.json')

    print(f'Reading split {split}...')
    sentences = [sentence for sentence in all_sentences if sentence.sent_id.split('S')[0][1:] in split_ids]
    split_dict = [s.to_dict() for s in sentences]
    stats = collections.Counter()
    for sentence in sentences:
      stats['entities'] += len(sentence.entities)
      for entity in sentence.entities:
        stats[entity.entity_type] += 1
      stats['relations'] += len(sentence.relations)
      for relation in sentence.relations:
        stats[relation.rel_type] += 1

    for stat, count in stats.most_common():
      print(f'{stat}: {count}')

    with split_output_path.open('w') as fp:
      json.dump(
        split_dict,
        fp,
        indent=2
      )
