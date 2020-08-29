import random

import torch

from spert import util


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # positive entities
    pos_entity_types = doc.entity_span_types
    pos_entity_sizes = doc.entity_span_sizes
    pos_entity_masks = [create_entity_mask(*span, context_size) for span in doc.entity_spans]

    pos_rel_spans = list(doc.rel_spans.keys())
    pos_rels = [(doc.entity_spans[s1], doc.entity_spans[s2]) for s1, s2 in pos_rel_spans]
    pos_rel_types = doc.rel_span_types

    # positive relations
    pos_rel_masks = [create_rel_mask(*rel, context_size) for rel in pos_rel_spans]

    # negative entities
    neg_entity_spans = doc.neg_entity_spans
    neg_entity_sizes = doc.neg_entity_span_sizes

    num_neg_entity_samples = min(len(neg_entity_spans), neg_entity_count)
    # sample negative entities
    neg_entity_samples = random.sample(
        list(zip(neg_entity_spans, neg_entity_sizes)),
        num_neg_entity_samples
    )
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = doc.neg_rel_spans

    num_neg_rel_samples = min(len(neg_rel_spans), neg_rel_count)
    # sample negative relations
    neg_rel_spans = random.sample(
        neg_rel_spans,
        num_neg_rel_samples
    )

    neg_rels = [(doc.entity_spans[s1], doc.entity_spans[s2]) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*span, context_size) for span in neg_rel_spans]
    neg_rel_types = [0] * len(neg_rel_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels

    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)
    rel_types_onehot = rel_types_onehot[:, 1:]  # all zeros for 'none' relation

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types_onehot,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = doc.spans
    entity_masks = [create_entity_mask(span.span_start, span.span_end, context_size) for span in doc.spans]
    entity_sizes = doc.span_sizes

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
