import torch
from torch.utils.data import TensorDataset
import numpy as np
from random import sample
import random
import json
import hashlib

entail_label_map = {'entail':1, 'nonentail':0}

def getDomainName(name):
    if name=="auto_commute":
        return "auto  commute"
    elif name=="credit_cards":
        return "credit cards"
    elif name=="kitchen_dining":
        return "kitchen  dining"
    elif name=="small_talk":
        return "small talk"
    elif ' ' not in name:
        return name
    else:
        raise NotImplementedError("Not supported domain name %s"%(name))

def splitName(dom):
    domList = []
    for name in dom.split(','):
        domList.append(getDomainName(name))
    return domList

def makeTrainExamples(data:list, tokenizer, label=None, mode='unlabel'):
    """
    unlabel: simply pad data and then convert into tensor
    multi-class: pad data and compose tensor dataset with labels
    entailment-utt: compose entailment examples with other utterances
    entailment-lab: compose entailment examples with labels
    """
    if mode != "unlabel":
        assert label is not None, f"Label is provided for the required setting {mode}"
        if mode == "multi-class":
            examples = tokenizer.pad(data, padding='longest', return_tensors='pt')
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            examples = TensorDataset(label,
                                     examples['input_ids'],
                                     examples['token_type_ids'],
                                     examples['attention_mask'])
        elif mode == "entailment-utt":
            examples = []
            entail_label = []
            label = np.array(label)
            # each datum compose one entailment example and one non-entailment example
            for d, l in zip(data, label):
                # entail
                ind = np.random.choice(np.where(label==l)[0], 1)[0]
                d_ = data[ind]
                e = {}
                e['input_ids'] = d['input_ids'] + d_['input_ids'][1:]
                e['token_type_ids'] = d['token_type_ids'] + [1]*(len(d_['token_type_ids'])-1)
                e['attention_mask'] = d['attention_mask'] + [1]*(len(d_['attention_mask'])-1)
                examples.append(e)
                entail_label.append(entail_label_map['entail'])
                # nonentail
                ind = np.random.choice(np.where(label!=l)[0], 1)[0]
                d_ = data[ind]
                e = {}
                e['input_ids'] = d['input_ids'] + d_['input_ids'][1:]
                e['token_type_ids'] = d['token_type_ids'] + [1]*(len(d_['token_type_ids'])-1)
                e['attention_mask'] = d['attention_mask'] + [1]*(len(d_['attention_mask'])-1)
                examples.append(e)
                entail_label.append(entail_label_map['nonentail'])
            examples = tokenizer.pad(examples, padding='longest', return_tensors='pt')
            entail_label = torch.tensor(entail_label)
            examples = TensorDataset(entail_label,
                                     examples['input_ids'],
                                     examples['token_type_ids'],
                                     examples['attention_mask'])
        elif mode == "entailment-lab":
            examples = []
            entail_label = []
            lab_set = set(label)
            # each datum compose one entailment example and one non-entailment example
            for d, l in zip(data, label):
                # entail
                e = {}
                e['input_ids'] = d['input_ids'] + tokenizer.encode(l)[1:]
                e['token_type_ids'] = d['token_type_ids'] + [1]*(len(e['input_ids'])-len(d['token_type_ids']))
                e['attention_mask'] = d['attention_mask'] + [1]*(len(e['input_ids'])-len(d['attention_mask']))
                examples.append(e)
                entail_label.append(entail_label_map['entail'])
                # nonentail
                nl = sample(lab_set - set([l]), 1)[0]
                e = {}
                e['input_ids'] = d['input_ids'] + tokenizer.encode(nl)[1:]
                e['token_type_ids'] = d['token_type_ids'] + [1]*(len(e['input_ids'])-len(d['token_type_ids']))
                e['attention_mask'] = d['attention_mask'] + [1]*(len(e['input_ids'])-len(d['attention_mask']))
                examples.append(e)
                entail_label.append(entail_label_map['nonentail'])
            examples = tokenizer.pad(examples, padding='longest', return_tensors='pt')
            entail_label = torch.tensor(entail_label)
            examples = TensorDataset(entail_label,
                                     examples['input_ids'],
                                     examples['token_type_ids'],
                                     examples['attention_mask'])
        else:
            raise ValueError(f"Undefined setting {mode}")
    else:
        examples = tokenizer.pad(data, padding='longest', return_tensors='pt')
        examples = TensorDataset(examples['input_ids'],
                                 examples['token_type_ids'],
                                 examples['attention_mask'])
    return examples

def makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer, mode='multi-class'):
    """
    multi-class: simply pad data
    entailment-utt: compose entailment examples with other utterances
    entailment-lab: compose entailment examples with labels
    """
    if mode == "multi-class":
        supportX = tokenizer.pad(supportX, padding='longest', return_tensors='pt')
        queryX = tokenizer.pad(queryX, padding='longest', return_tensors='pt')
    elif mode == "entailment-utt":
        sX, sY = [], []
        for i, (x1, y1) in enumerate(zip(supportX, supportY)):
            for j, (x2, y2) in enumerate(zip(supportX, supportY)):
                if i < j:
                    e = {}
                    e['input_ids'] = x1['input_ids'] + x2['input_ids'][1:]
                    e['token_type_ids'] = x1['token_type_ids'] + [1]*(len(x2['input_ids'])-1)
                    e['attention_mask'] = x1['attention_mask'] + [1]*(len(x2['input_ids'])-1)
                    sX.append(e)
                    if y1 == y2:
                        sY.append(entail_label_map['entail'])
                    else:
                        sY.append(entail_label_map['nonentail'])
        qX, qY = [], []
        # https://blog.finxter.com/how-to-remove-duplicates-from-a-python-list-while-preserving-order/
        lab_set = list(dict.fromkeys(supportY))
        for x1, y1 in zip(queryX, queryY):
            for x2, y2 in zip(supportX, supportY):
                e = {}
                e['input_ids'] = x1['input_ids'] + x2['input_ids'][1:]
                e['token_type_ids'] = x1['token_type_ids'] + [1]*(len(x2['input_ids'])-1)
                e['attention_mask'] = x1['attention_mask'] + [1]*(len(x2['input_ids'])-1)
                qX.append(e)
            qY.append(lab_set.index(y1))
        supportX = tokenizer.pad(sX, padding='longest', return_tensors='pt')
        supportY = sY
        queryX = tokenizer.pad(qX, padding='longest', return_tensors='pt')
        queryY = qY
    elif mode == "entailment-lab":
        sX, sY = [], []
        lab_set = list(set(supportY))
        for x, y in zip(supportX, supportY):
            for lab in lab_set:
                e = {}
                e['input_ids'] = x['input_ids'] + tokenizer.encode(lab)[1:]
                e['token_type_ids'] = x['token_type_ids'] + [1]*(len(e['input_ids'])-len(x['token_type_ids']))
                e['attention_mask'] = x['attention_mask'] + [1]*(len(e['input_ids'])-len(x['attention_mask']))
                sX.append(e)
                if lab == y:
                    sY.append(entail_label_map['entail'])
                else:
                    sY.append(entail_label_map['nonentail'])
        qX, qY = [], []
        for x, y in zip(queryX, queryY):
            for lab in lab_set:
                e = {}
                e['input_ids'] = x['input_ids'] + tokenizer.encode(lab)[1:]
                e['token_type_ids'] = x['token_type_ids'] + [1]*(len(e['input_ids'])-len(x['token_type_ids']))
                e['attention_mask'] = x['attention_mask'] + [1]*(len(e['input_ids'])-len(x['attention_mask']))
                qX.append(e)
            qY.append(lab_set.index(y))
        supportX = tokenizer.pad(sX, padding='longest', return_tensors='pt')
        supportY = sY
        queryX = tokenizer.pad(qX, padding='longest', return_tensors='pt')
        queryY = qY
    return supportX, supportY, queryX, queryY

#https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def getMD5(items):
    dictionary = {}
    dictionary[0] = items
    
    strCode = json.dumps(dictionary, sort_keys=False).encode('utf-8')
    sign = hashlib.md5(strCode).hexdigest()
    return sign
