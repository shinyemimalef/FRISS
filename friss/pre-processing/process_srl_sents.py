# Step Two of Pipeline

import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from friss.configs.config import MFC_graph_dataset_raw_directory
from friss.utils import common_member, serialize_to_file, deserialize_from_file, \
    RAW_SRL_SENTS_UNLABELLED_FILE_PATH, RAW_SRL_SENTS_UNSHUFFLED_FILE_PATH, get_doc_labels

interested_tags_list = ["B-V", "B-ARGM-MOD", "B-ARGM-DIR", "B-ARGM-NEG", 'B-ARG0', 'B-ARG1', "B-ARG2", "B-ARG3",
                        "B-ARG4"]
real_args_list = ['B-ARG0', 'B-ARG1', "B-ARG2", "B-ARG3", "B-ARG4"]


def get_predicate_span_tags_dict(tags):
    # Initial Structure of span_tags_dict: {"B-V": [(30, 31)], "B-ARG2": [(28,28), (32,40),...]}
    predicate_span_tags_dict = defaultdict(list)
    i = 0
    while i < len(tags):
        if tags[i] in interested_tags_list:
            start_index = i
            i += 1
            while i < len(tags) and tags[i][0] == 'I':
                i += 1
            end_index = i - 1
            predicate_span_tags_dict[tags[start_index]].append((start_index, end_index))
            continue
        else:
            i += 1
    return predicate_span_tags_dict


# Given the srl, words in coref and coref clusters
# Returns a list of items with aligned srl and coref information
# Each item belongs to one document, and is a dictionary with key being srl tags and values tag positions and cluster positions
# Each position is a tuple of starting word index and end word index
def process_srl_sents_for_one_doc(srl_doc):
    doc_words = list()

    doc_span_tags = list()
    for srl_sent in srl_doc:
        sent_span_tags = list()
        for srl_verb in srl_sent['verbs']:
            tags = srl_verb['tags']
            predicate_span_tags_dict = get_predicate_span_tags_dict(tags)

            # Pattern 1: The srl unit should have verb present
            if "B-V" not in predicate_span_tags_dict.keys():
                continue
            # Pattern 2: If there are more than one verb, choose the first one for now.
            if len(predicate_span_tags_dict['B-V']) > 1:
                predicate_span_tags_dict['B-V'] = [predicate_span_tags_dict['B-V'][0]]
            # Pattern 3: The srl unit should have at least one real arg
            if not common_member(list(predicate_span_tags_dict.keys()), real_args_list):
                continue
            # Pattern 4: If there are more than one particular type of arg, choose the one closest to the verb:
            # E.G. [ARG1: UNCLE SAM] [ARGM-MOD: CAN'T] [V: TURN] [ARGM-DIR: BACK] [ARG1: ON LEGAL U.S. IMMIGRANTS] .
            assert len(predicate_span_tags_dict['B-V']) == 1
            appro_verb_pos = int((predicate_span_tags_dict['B-V'][0][0] + predicate_span_tags_dict['B-V'][0][1]) / 2)
            for tag, poss in predicate_span_tags_dict.items():
                if len(poss) > 1:
                    appro_poss = [int((pos[0] + pos[1]) / 2) for pos in poss]
                    appro_diff_from_verb = [abs(appro_pos - appro_verb_pos) for appro_pos in appro_poss]
                    index = np.where(appro_diff_from_verb == np.amin(
                        appro_diff_from_verb))  # The returned index is a tuple (array([0, 2]),)
                    predicate_span_tags_dict[tag] = [predicate_span_tags_dict[tag][index[0][0]]]

            sent_span_tags.append(predicate_span_tags_dict)
        if len(sent_span_tags) > 0:
            doc_span_tags.append(sent_span_tags)
            sent_words = srl_sent['words']
            doc_words.append(sent_words)
    # for each doc
    return doc_words, doc_span_tags


def process_srl_sents_for_one_doc_v2(srl_doc):
    doc_words = list()

    doc_span_tags = list()
    for srl_sent in srl_doc:
        sent_span_tags = list()
        for srl_verb in srl_sent['verbs']:
            tags = srl_verb['tags']
            predicate_span_tags_dict = get_predicate_span_tags_dict(tags)

            # Pattern 1: The srl unit should have verb present
            if "B-V" not in predicate_span_tags_dict.keys():
                continue
            # Pattern 2: If there are more than one verb, choose the first one for now.
            if len(predicate_span_tags_dict['B-V']) > 1:
                predicate_span_tags_dict['B-V'] = [predicate_span_tags_dict['B-V'][0]]
            # Pattern 3: The srl unit should have at least one real arg
            if not common_member(list(predicate_span_tags_dict.keys()), real_args_list):
                continue
            # Pattern 4: If there are more than one particular type of arg, choose the one closest to the verb:
            # E.G. [ARG1: UNCLE SAM] [ARGM-MOD: CAN'T] [V: TURN] [ARGM-DIR: BACK] [ARG1: ON LEGAL U.S. IMMIGRANTS] .
            assert len(predicate_span_tags_dict['B-V']) == 1
            appro_verb_pos = int((predicate_span_tags_dict['B-V'][0][0] + predicate_span_tags_dict['B-V'][0][1]) / 2)
            for tag, poss in predicate_span_tags_dict.items():
                if len(poss) > 1:
                    appro_poss = [int((pos[0] + pos[1]) / 2) for pos in poss]
                    appro_diff_from_verb = [abs(appro_pos - appro_verb_pos) for appro_pos in appro_poss]
                    index = np.where(appro_diff_from_verb == np.amin(
                        appro_diff_from_verb))  # The returned index is a tuple (array([0, 2]),)
                    predicate_span_tags_dict[tag] = [predicate_span_tags_dict[tag][index[0][0]]]

            sent_span_tags.append(predicate_span_tags_dict)

        doc_span_tags.append(sent_span_tags)
        sent_words = srl_sent['words']
        doc_words.append(sent_words)
    # for each doc
    return doc_words, doc_span_tags


def process_srl_sents_for_all_docs(raw_srl_path, labelled=True):
    srl_results = deserialize_from_file(raw_srl_path)
    frame_labels = None
    if labelled:
        frame_labels = get_doc_labels("immigration_unshuffled")
        assert len(frame_labels) == len(srl_results)
        doc_num = len(frame_labels)
        issue = "immigration_unshuffled"
    else:
        doc_num = len(srl_results)
        issue = "immigration_unlabelled"

    mfc_graphs = list()
    # for doc_id in doc_ids:
    for doc_id in tqdm(range(doc_num)):
        srl_doc = srl_results[doc_id]
        doc_words, verb_args_list = process_srl_sents_for_one_doc(srl_doc)
        # Doc words is a list of sentences, each sentence is a list of words
        if labelled:
            mfc_graphs.append(
                [doc_id, issue, int(frame_labels[doc_id]) - 1, doc_words, verb_args_list])
        else:
            mfc_graphs.append(
                [doc_id, issue, -1, doc_words, verb_args_list])

    serialize_to_file(os.path.join(MFC_graph_dataset_raw_directory,
                                   "mfc_graphs_sents_complete_{}_{}.json".format(issue, doc_num)), mfc_graphs)


if __name__ == '__main__':
    process_srl_sents_for_all_docs(RAW_SRL_SENTS_UNSHUFFLED_FILE_PATH, labelled=True)
    process_srl_sents_for_all_docs(RAW_SRL_SENTS_UNLABELLED_FILE_PATH, labelled=False)
