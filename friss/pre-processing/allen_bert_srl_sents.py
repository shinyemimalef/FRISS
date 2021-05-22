# Step One of Pipeline

import argparse
import csv
import os
import re

import torch
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

from friss.utils import nlp_lg, deserialize_from_file, serialize_to_file, AverageMeter

issue = "immigration_unlabeled"
user = ""

home_dir = f'/home/{user}'
datasets_directory = os.path.join(home_dir, 'datasets')
MFC_dataset_directory = os.path.join(datasets_directory, 'MFC-dataset')
print_to_console = False
# Limit the number of BERT TOKENS to 512 words
# because bert can process at most 512 tokens including the two special tokens [CLS] and [SEP]
# at the beginning and the end of the sentence.
NUM_LIMIT_BERT_TOKENS = 512
NUM_LIMIT_TOKENS = 300
SAVING_INTERVAL = 100


def get_predictors():
    print("current device: {}".format(torch.cuda.current_device()))
    srl_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz",
        cuda_device=torch.cuda.current_device())

    return srl_predictor


def run(srl_predictor, tsv_lines,
        srl_output_file, step, partition):
    def _run_srl_predictor(input):
        result = srl_predictor.predict_batch_json(input)
        if print_to_console:
            print("prediction: ", result)
        return result

    # process srl and coref
    srl_results = list()  # This is a list of dictionary, each dictionary refers to one document
    for i in tqdm(range(step * partition, (step + 1) * partition)):
        if i >= len(tsv_lines):
            break

        data = tsv_lines[i]
        assert int(data[0]) == i
        torch.cuda.empty_cache()

        sents_str = [{"sentence": sent_str} for sent_str in data[-1]]
        srl_results.append(_run_srl_predictor(sents_str))

    serialize_to_file(srl_output_file, srl_results)


def get_tsv_lines(input_file):
    with open(input_file, mode='r') as ifile:
        tsv_reader = csv.reader(ifile, delimiter='\t')
        tsv_lines = [data for data in tsv_reader]
        for i in tqdm(range(len(tsv_lines))):
            doc_str = re.sub(' +', ' ', tsv_lines[i][3]).strip()
            doc_str = re.sub("U.S.", "US", doc_str).strip()
            doc_str = re.sub(" /.", ".", doc_str).strip()
            doc_str = re.sub("/.", ".", doc_str).strip()
            doc_tokens = doc_str.split()
            doc_str = " ".join([doc_token[0] + doc_token[1:].lower() for doc_token in doc_tokens])
            spacy_doc = nlp_lg(doc_str, disable=["ner"])
            sents_str = [str(sent).strip() for sent in spacy_doc.sents]
            doc_str = " ".join(sents_str)
            tsv_lines[i][3] = doc_str
            tsv_lines[i].append(sents_str)

    return tsv_lines


def main(step, partition):
    tsv_lines_file = os.path.join(MFC_dataset_directory, "{}/{}_tsv_lines.json".format(issue, issue))
    if os.path.exists(tsv_lines_file):
        tsv_lines = deserialize_from_file(tsv_lines_file)
    else:
        raise FileNotFoundError
    srl_predictor = get_predictors()
    srl_output_file = os.path.join(MFC_dataset_directory,
                                   "{}/{}_bert_srl_sents_{}_{}.json".format(issue, issue, step, partition))
    run(srl_predictor,
        tsv_lines,
        srl_output_file, step, partition)


def combine(step, partition):
    srl_all = list()

    for i in range(step + 1):
        srl_output_file = os.path.join(MFC_dataset_directory,
                                       "{}/{}_bert_srl_sents_{}_{}.json".format(issue, issue, i, partition))

        temp_srl = deserialize_from_file(srl_output_file)
        srl_all.extend(temp_srl)

    serialize_to_file(os.path.join(MFC_dataset_directory, "{}/{}_bert_srl_sents.json".format(issue, issue)), srl_all)


def test():
    all_srl = deserialize_from_file(
        os.path.join(MFC_dataset_directory, "{}/{}_bert_srl_sents.json".format(issue, issue)))
    all_coref = deserialize_from_file(
        os.path.join(MFC_dataset_directory, "{}/{}_bert_coref_sents.json".format(issue, issue)))
    print("len of all_srl: {}".format(len(all_srl)))
    print("len of all_coref: {}".format(len(all_coref)))
    print(all_srl[-1])
    print(all_coref[-1])
    sents_per_doc_avg_meter = AverageMeter()
    sent_len_avg_meter = AverageMeter()
    predicates_per_sent_avg_meter = AverageMeter()

    for doc in all_srl:
        sents_per_doc_avg_meter.update(len(doc))
        for sent in doc:
            sent_len_avg_meter.update(len(sent['words']))
            predicates_per_sent_avg_meter.update(len(sent['verbs']))
    sent_len_avg_meter.plot("sent_len.png")
    sents_per_doc_avg_meter.plot("sents_per_doc.png")
    predicates_per_sent_avg_meter.plot("predicates_per_sent.png")
    print(sent_len_avg_meter.avg)
    print(sents_per_doc_avg_meter.avg)
    print(predicates_per_sent_avg_meter.avg)


def convert_tsv_to_json():
    input_file = os.path.join(MFC_dataset_directory, "{}/{}.tsv".format(issue, issue))
    tsv_lines_file = os.path.join(MFC_dataset_directory, "{}/{}_tsv_lines.json".format(issue, issue))

    tsv_lines = get_tsv_lines(input_file)
    serialize_to_file(tsv_lines_file, tsv_lines)


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(
        description="Parser for allen_bert_Srl_coref_sents ")

    main_arg_parser.add_argument("--step", type=int, required=True, help="step")
    main_arg_parser.add_argument('--partition', type=int, required=True, help="partition")
    shell_args = main_arg_parser.parse_args()

    # Run the following three lines one by one!
    # convert_tsv_to_json()
    # main(getattr(shell_args, 'step'), getattr(shell_args, 'partition'))
    combine(getattr(shell_args, 'step'), getattr(shell_args, 'partition'))
