# Code adapted from HuggingFace repository
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
import re
from collections import defaultdict, Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import spacy
import torch
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                          XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from friss.configs.config import MFC_dataset_directory, config_args, issues_config, issue

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frames_i2w = ["Economic", "Capacity-and-resrouces", "Morality", "Fairness-and-equality",
              "Legality,-constitutionality-and-jurisprudence", "Policy-prescription-and-evaluation",
              "Crime-and-punishment", "Security-and-defense", "Health-and-safety", "Quality-of-life",
              "Cultural-identity", "Public-opinion", "Political", "External-regulation-and-reputation", "Other"]
MAX_EPOCHS = 100

nlp = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")
samesex_6_balanced_label_mapping = {10: 0, 11: 1, 12: 2, 3: 3, 5: 4, 13: 5}
spacy_tokenizer = SpacyTokenizer(language="en_core_web_sm", pos_tags=True)


# e.g. tokens = spacy_tokenizer.tokenize(sentence)


def deserialize_from_file(filename="data.json"):
    with open(filename, "rb") as read_file:
        data = json.load(read_file)
        return data


def serialize_to_file(filename="data.json", data=None):
    with open(filename, "w") as write_file:
        json.dump(data, write_file)


def append_to_file(filename="data.json", data=None):
    if os.path.exists(filename):
        with open(filename, "rb") as read_file:
            previous_data = json.load(read_file)
            data = previous_data + data

    with open(filename, "w") as write_file:
        json.dump(data, write_file)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_a_tokens=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_a_tokens = text_a_tokens
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id):
        self.guid = guid,
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MulticlassProcessor(object):
    """Processor for the binary data sets"""

    def __init__(self, num_labels, doc_labels):
        self.num_labels = num_labels
        self.doc_labels = doc_labels

    def get_labels(self):
        """See base class."""
        # return [str(i) for i in set(self.doc_labels)]
        return [str(i) for i in range(self.num_labels)]

    def create_examples(self, lines, text_a_tokens_for_all_docs=None):
        """Creates examples for the training and dev sets."""
        if text_a_tokens_for_all_docs:
            assert len(lines) == len(text_a_tokens_for_all_docs)
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[3]
            text_a_tokens = None
            if text_a_tokens_for_all_docs:
                text_a_tokens = text_a_tokens_for_all_docs[i]
            label = int(line[1]) - 1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_a_tokens=text_a_tokens, text_b=None, label=label))
        return examples

    def create_examples_sents(self, text_a_tokens_for_all_docs):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, text_a_tokens_for_one_doc) in enumerate(text_a_tokens_for_all_docs):
            guid = i
            label = int(self.doc_labels[i])
            for text_a_tokens in text_a_tokens_for_one_doc:  # text_a_tokens is for one sentence
                examples.append(
                    InputExample(guid=guid, text_a=None, text_a_tokens=text_a_tokens, text_b=None, label=label))
        return examples


def convert_example_to_feature_roberta_base(example_row, pad_token=0,
                                            sequence_a_segment_id=0, sequence_b_segment_id=1,
                                            cls_token_segment_id=1, pad_token_segment_id=0,
                                            mask_padding_with_zero=True, sep_token_extra=False):
    example, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra = example_row

    tokens_a = example.text_a_tokens

    # the comparision is finished
    guid = example.guid
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if example.label == -1:
        label_id = 0  # For unlabelled data, assume the label_id is 0
    else:
        label_id = example.label

    return InputFeatures(guid=guid,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, sep_token_extra=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples = [(example, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token,
                 cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra) for example in examples]

    process_count = cpu_count() - 2

    with Pool(process_count) as p:
        features = list(
            tqdm(p.imap(convert_example_to_feature_roberta_base, examples, chunksize=500), total=len(examples)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def top_k_accuracy(top_k_best_preds, labels, k=2):
    assert top_k_best_preds.size(1) >= k
    num_correct_top_k_preds = 0
    for i, v in enumerate(top_k_best_preds):
        v = v[:k]
        if labels[i] in v:
            num_correct_top_k_preds += 1

    acc = num_correct_top_k_preds / top_k_best_preds.size(0)
    return acc


def compute_metrics(preds, labels, num_labels, false_pos=None, false_neg=None):
    top_k_best_preds_values, top_k_best_preds_indices = torch.topk(torch.Tensor(preds), 3, dim=1)
    best_preds = top_k_best_preds_indices[:, 0].numpy()

    assert len(best_preds) == len(labels)
    mcc = matthews_corrcoef(labels, best_preds)
    conf_mx = multilabel_confusion_matrix(labels, best_preds, labels=list(range(num_labels)))
    classification_report_result = classification_report(labels, best_preds, digits=6)
    accuracy = accuracy_score(labels, best_preds)
    top_2_accuracy = top_k_accuracy(top_k_best_preds_indices, labels, k=2)
    top_3_accuracy = top_k_accuracy(top_k_best_preds_indices, labels, k=3)
    f1 = f1_score(labels, best_preds, average='macro')

    if false_neg is None:
        # the dict key is the label, and the Counter is how many wrongly predicted for each wrongly predicted class
        false_neg = defaultdict(Counter)
    if false_pos is None:
        false_pos = defaultdict(Counter)
    for i, label in enumerate(labels):
        if label != best_preds[i]:
            false_neg[label][best_preds[i]] += 1
            false_pos[best_preds[i]][label] += 1

    return mcc, conf_mx, classification_report_result, accuracy, f1, false_pos, false_neg, top_2_accuracy, top_3_accuracy


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.counter = Counter()

    def update(self, val, n=1):
        self.val = val
        self.counter[self.val] += 1
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def plot(self, name):
        plt.bar(self.counter.keys(), self.counter.values())
        plt.savefig(name)
        plt.clf()


class TopKMeter(object):
    def __init__(self, k=30, g=0.7):
        self.top_k_keys = list()
        self.top_k_values = list()
        self.top_k_weights = list()
        self.top_k_keys_no_duplicates = list()
        self.top_k_values_no_duplicates = list()
        self.top_k_weights_no_duplicates = list()
        self.k = k
        self.g = g
        self.items_above_the_threshold = defaultdict(list)

    def process_text(self, text):
        text = text.lower().replace("</s>", "").replace("<s>", "")
        return text

    def update_above_threshold(self, weights, keys, values):
        for i in range(len(weights)):
            if weights[i] > self.g:
                processed_key = self.process_text(keys[i])
                self.items_above_the_threshold[processed_key].append((values[i], weights[i]))

    def get_items_above_threshold(self, k=500):
        r = dict()
        i = 0
        for key in sorted(self.items_above_the_threshold, key=lambda x: len(self.items_above_the_threshold[x]),
                          reverse=True):
            if i == k:
                break
            l = self.items_above_the_threshold[key]  # a list of tuples (value, weight), ideally sort it by weight
            l.sort(key=lambda y: y[1], reverse=True)
            r[key] = l
            i += 1
        return r

    def update(self, weights, keys, values, reverse=True):
        self.top_k_keys.extend(keys)
        self.top_k_weights.extend(weights)
        self.top_k_values.extend(values)
        temp_k = list()
        temp_v = list()
        temp_w = list()
        for w, k, v in sorted(zip(self.top_k_weights, self.top_k_keys, self.top_k_values), reverse=reverse):
            temp_w.append(w)
            temp_k.append(k)
            temp_v.append(v)

        self.top_k_keys = temp_k[:min(self.k, len(self.top_k_keys))]
        self.top_k_values = temp_v[: min(self.k, len(self.top_k_keys))]
        self.top_k_weights = temp_w[:min(self.k, len(self.top_k_keys))]

    def update_no_duplicates(self, weights, keys, values):

        self.top_k_keys_no_duplicates.extend(keys)
        self.top_k_values_no_duplicates.extend(values)
        self.top_k_weights_no_duplicates.extend(weights)
        temp_w = list()
        temp_k = list()
        temp_v = list()
        for w, k, v in sorted(
                zip(self.top_k_weights_no_duplicates, self.top_k_keys_no_duplicates, self.top_k_values_no_duplicates),
                reverse=True):
            temp_w.append(w)
            temp_k.append(k)
            temp_v.append(v)

        self.top_k_keys_no_duplicates = list()
        self.top_k_values_no_duplicates = list()
        self.top_k_weights_no_duplicates = list()

        for i, k in enumerate(temp_k):
            if len(self.top_k_keys_no_duplicates) == self.k:
                break
            if k not in self.top_k_keys_no_duplicates:
                self.top_k_keys_no_duplicates.append(k)
                self.top_k_values_no_duplicates.append(temp_v[i])
                self.top_k_weights_no_duplicates.append(temp_w[i])

    def get_topk_values(self):
        return self.top_k_values

    def get_topk_values_no_duplicates(self):
        return self.top_k_values_no_duplicates

    def get_topk_keys_no_duplicates(self):
        return self.top_k_keys_no_duplicates


def get_time_stamp():
    now = datetime.now()
    return now.strftime("%m-%d-%Y-%H:%M:%S")


def get_doc_labels(issue):
    input_file_path = os.path.join(MFC_dataset_directory, issue, "{}.tsv".format(issue))
    with open(input_file_path, mode='r') as ifile:
        tsv_reader = csv.reader(ifile, delimiter='\t')
        frame_labels = [data[1] for data in tsv_reader]
    return frame_labels


def _read_tsv(input_file):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def load_original_data():
    # load dataset and create a split
    dataset = _read_tsv(os.path.join(config_args['data_dir'], str(config_args['issue']) + '.tsv'))
    return dataset


def load_data_with_validation(n_splits):
    # load dataset and create a split
    if os.path.exists(os.path.join(config_args["data_dir"], "fold_train_valid_test_split_indices.json")):
        return deserialize_from_file(os.path.join(config_args["data_dir"], "fold_train_valid_test_split_indices.json"))
    lines = _read_tsv(os.path.join(config_args['data_dir'], str(config_args['issue']) + '.tsv'))

    kf = StratifiedKFold(n_splits=n_splits, random_state=68, shuffle=True)

    y = [int(i[1]) - 1 for i in lines]  # Minus one because MFC labels start from 1
    dataset = dict()
    dataset_indices = dict()
    time = 0
    for train_indices, test_indices in kf.split(lines, y):  # ([0,1,...,1000],[1001,...,9999]) () () ...
        dataset[time] = dict()
        dataset_indices[time] = dict()
        dataset[time]['train'] = [lines[i] for i in train_indices]
        # Split out part of train for validation
        assert train_indices.shape[0] == len(dataset[time]['train'])
        train_y = [int(i[1]) for i in dataset[time]['train']]
        X_train, X_valid = train_test_split(dataset[time]['train'], test_size=0.1, random_state=68, stratify=train_y)

        dataset_indices[time]['train_indices'] = [int(i[0]) for i in X_train]

        dataset_indices[time]['valid_indices'] = [int(i[0]) for i in X_valid]

        dataset_indices[time]['test_indices'] = test_indices.tolist()
        time += 1
    serialize_to_file(os.path.join(config_args["data_dir"], "fold_train_valid_test_split_indices.json"),
                      dataset_indices)
    return dataset_indices


def load_data(n_splits):
    # load dataset and create a split
    if os.path.exists(os.path.join(config_args["data_dir"], "fold_train_test_split_indices.json")):
        return deserialize_from_file(os.path.join(config_args["data_dir"], "fold_train_test_split_indices.json"))
    lines = _read_tsv(os.path.join(config_args['data_dir'], str(config_args['issue']) + '.tsv'))

    kf = StratifiedKFold(n_splits=n_splits, random_state=68, shuffle=True)

    y = [int(i[1]) - 1 for i in lines]  # Minus one because MFC labels start from 1

    dataset_indices = dict()
    fold_index = 0
    for train_indices, test_indices in kf.split(lines, y):  # ([0,1,...,1000],[1001,...,9999]) () () ...

        dataset_indices[fold_index] = dict()

        dataset_indices[fold_index]['train_indices'] = train_indices.tolist()

        dataset_indices[fold_index]['test_indices'] = test_indices.tolist()

        assert train_indices.shape[0] == len(dataset_indices[fold_index]['train_indices'])
        fold_index += 1
    serialize_to_file(os.path.join(config_args["data_dir"], "fold_train_test_split_indices.json"), dataset_indices)
    return dataset_indices


def load_bert_features_sents(task, tokenizer, doc_labels, text_a_tokens_for_all_docs=None):
    processor = processors[task](issues_config[config_args['issue']], doc_labels)
    output_mode = config_args['output_mode']
    label_list = processor.get_labels()
    examples = processor.create_examples_sents(text_a_tokens_for_all_docs)
    features = convert_examples_to_features(examples, label_list, config_args['max_seq_length'], tokenizer, output_mode,
                                            cls_token_at_end=bool(config_args['model_type'] in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if config_args['model_type'] in ['xlnet'] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(config_args['model_type'] in ['roberta']),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(config_args['model_type'] in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if config_args['model_type'] in ['xlnet'] else 0)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids


def spacy2bert(bert_tokenizer, tokens):
    """
    Convert a list of tokens to wordpiece tokens and offsets, as well as adding
    BERT CLS and SEP tokens to the begining and end of the sentence.

    A slight oddity with this function is that it also returns the wordpiece offsets
    corresponding to the _start_ of words as well as the end.

    We need both of these offsets (or at least, it's easiest to use both), because we need
    to convert the labels to tags using the end_offsets. However, when we are decoding a
    BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
    because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
    wordpieces (this happens in the case that a word is split into multiple word pieces,
    and then we take the last tag of the word, which might correspond to, e.g, I-V, which
    would not be allowed as it is not preceeded by a B tag).

    For example:

    `annotate` will be bert tokenized as ["anno", "##tate"].
    If this is tagged as [B-V, I-V] as it should be, we need to select the
    _first_ wordpiece label to be the label for the token, because otherwise
    we may end up with invalid tag sequences (we cannot start a new tag with an I).

    # Returns

    wordpieces : List[str]
        The BERT wordpieces from the words in the sentence.
    end_offsets : List[int]
        Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
        results in the end wordpiece of each word being chosen.
    start_offsets : List[int]
        Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
        results in the start wordpiece of each word being chosen.
    """
    word_piece_tokens = list()
    end_offsets = list()
    start_offsets = list()
    cumulative = 0
    for token in tokens:
        token = token.lower()
        word_pieces = bert_tokenizer.wordpiece_tokenizer.tokenize(token)
        start_offsets.append(cumulative + 1)
        cumulative += len(word_pieces)
        end_offsets.append(cumulative)
        word_piece_tokens.extend(word_pieces)
    wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

    return wordpieces, end_offsets, start_offsets


class SrlTreeNode(object):
    def __init__(self, verb, argm_mod=None, argm_neg=None, argm_dir=None):
        # "B-V", "B-ARGM-MOD", "B-ARGM-DIR", "B-ARGM-NEG", 'B-ARG0', 'B-ARG1', "B-ARG2", "B-ARG3", "B-ARG4"
        self.arg0 = None  # self.arg0 and self.arg1 can either be a list of strings, or another SrlTreeNode
        self.arg1 = None
        self.arg2 = None
        self.arg3 = None
        self.arg4 = None
        self.verb = verb
        self.argm_mod = argm_mod
        self.argm_neg = argm_neg
        self.argm_dir = argm_dir


B_sorted_srl_tags = ['B-ARG0', 'B-ARG1', 'B-ARG2', 'B-ARG3', 'B-ARG4', 'B-ARG5', 'B-ARGA', 'B-ARGM-ADJ', 'B-ARGM-ADV',
                     'B-ARGM-CAU', 'B-ARGM-COM', 'B-ARGM-DIR', 'B-ARGM-DIS', 'B-ARGM-EXT', 'B-ARGM-GOL', 'B-ARGM-LOC',
                     'B-ARGM-LVB', 'B-ARGM-MNR', 'B-ARGM-MOD', 'B-ARGM-NEG', 'B-ARGM-PNC', 'B-ARGM-PRD', 'B-ARGM-PRP',
                     'B-ARGM-REC', 'B-ARGM-TMP', 'B-C-ARG0', 'B-C-ARG1', 'B-C-ARG2', 'B-C-ARG3', 'B-C-ARG4',
                     'B-C-ARGM-ADV', 'B-C-ARGM-CAU', 'B-C-ARGM-EXT', 'B-C-ARGM-LOC', 'B-C-ARGM-MNR', 'B-C-ARGM-TMP',
                     'B-R-ARG0', 'B-R-ARG1', 'B-R-ARG2', 'B-R-ARG3', 'B-R-ARGM-ADV', 'B-R-ARGM-CAU', 'B-R-ARGM-COM',
                     'B-R-ARGM-EXT', 'B-R-ARGM-GOL', 'B-R-ARGM-LOC', 'B-R-ARGM-MNR', 'B-R-ARGM-PRP', 'B-R-ARGM-TMP',
                     'B-V']
I_sorted_srl_tags = ['I-ARG0', 'I-ARG1', 'I-ARG2', 'I-ARG3', 'I-ARG4', 'I-ARG5', 'I-ARGM-ADJ', 'I-ARGM-ADV',
                     'I-ARGM-CAU', 'I-ARGM-COM', 'I-ARGM-DIR', 'I-ARGM-DIS', 'I-ARGM-EXT', 'I-ARGM-GOL', 'I-ARGM-LOC',
                     'I-ARGM-MNR', 'I-ARGM-NEG', 'I-ARGM-PNC', 'I-ARGM-PRD', 'I-ARGM-PRP', 'I-ARGM-TMP', 'I-C-ARG0',
                     'I-C-ARG1', 'I-C-ARG2', 'I-C-ARG3', 'I-C-ARG4', 'I-C-ARGM-ADV', 'I-C-ARGM-CAU', 'I-C-ARGM-EXT',
                     'I-C-ARGM-LOC', 'I-C-ARGM-MNR', 'I-C-ARGM-TMP', 'I-R-ARG0', 'I-R-ARG1', 'I-R-ARG2', 'I-R-ARG3',
                     'I-R-ARGM-ADV', 'I-R-ARGM-COM', 'I-R-ARGM-EXT', 'I-R-ARGM-GOL', 'I-R-ARGM-LOC', 'I-R-ARGM-MNR',
                     'I-R-ARGM-TMP']

SORTED_SRL_TAGS = B_sorted_srl_tags + I_sorted_srl_tags + ['O']
RAW_SRL_FILE_PATH = os.path.join(MFC_dataset_directory, '{}/{}_bert_srl.json'.format(issue, issue))
RAW_COREF_FILE_PATH = os.path.join(MFC_dataset_directory, '{}/{}_bert_coref.json'.format(issue, issue))
RAW_COREF_SENTS_FILE_PATH = os.path.join(MFC_dataset_directory, '{}/{}_bert_coref_sents.json'.format(issue, issue))
RAW_SRL_SENTS_FILE_PATH = os.path.join(MFC_dataset_directory, '{}/{}_bert_srl_sents.json'.format(issue, issue))
RAW_SRL_SENTS_UNLABELLED_FILE_PATH = os.path.join(MFC_dataset_directory,
                                                  '{}/{}_bert_srl_sents.json'.format("immigration_unlabelled",
                                                                                     "immigration_unlabelled"))
RAW_SRL_SENTS_UNSHUFFLED_FILE_PATH = os.path.join(MFC_dataset_directory,
                                                  '{}/{}_bert_srl_sents.json'.format("immigration_unshuffled",
                                                                                     "immigration_unshuffled"))
ALL_STOPWORDS = nlp.Defaults.stop_words


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False


def root_texts(texts):
    roots = list()
    # selected_ner_labels = ["EVENT", "FAC", "GPE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]
    for doc in nlp_lg.pipe(texts, disable=['ner']):  # disable=["tagger", "parser", "ner"]

        root_list = list()
        for chunk in doc.noun_chunks:
            root = chunk.text
            root_doc = nlp(root, disable=["tagger", "parser", "ner"])
            root = " ".join([tok.lemma_ for tok in root_doc if not tok.is_stop])
            root = re.sub('[^a-zA-Z ]+', '', root)  # allow only english letters
            root = root.lower().strip()
            root_list.append(root)
        roots.append(root_list)
    return roots


processors = {
    "MFC_multiclass_classification": MulticlassProcessor
}
