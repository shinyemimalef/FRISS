# Step Three of the Pipeline

from torch.utils.data import Dataset

from friss.configs.config import *
from friss.utils import *


class MfcSentsDataset(Dataset):
    def __init__(self, root, name):
        self.name = name
        self.root = root
        super().__init__()
        self.input_file_names = [os.path.join(self.root, "raw", input_file_name) for input_file_name in
                                 self.get_input_file_names()]
        self.raw_data = list()
        if 'unshuffled' in self.input_file_names[0]:
            self.num_labelled = 5275
        else:
            self.num_labelled = 5933
        for input_file_name in self.input_file_names:
            self.raw_data.extend(deserialize_from_file(input_file_name))
        self.doc_labels = list()
        for i in range(len(self.raw_data)):
            if i < self.num_labelled:
                self.doc_labels.append(self.raw_data[i][2] - 1)
            else:
                self.doc_labels.append(0)  # for unlabelled data, assume the label is 0 and use mask during loss

        self.roberta_doc_arg_verb_spans_for_all_docs_path = os.path.join(self.root, "processed",
                                                                         "roberta_doc_arg_verb_spans_for_all_docs.json")
        self.dataset_size = len(self.raw_data)
        self.processed_roberta_path = os.path.join(MFC_graph_dataset_processed_directory,
                                                   "mfc_roberta_sents_features.pt")
        if os.path.exists(self.processed_roberta_path):
            self.roberta_features = torch.load(self.processed_roberta_path)
            self.roberta_doc_arg_verb_spans_for_all_docs = deserialize_from_file(
                self.roberta_doc_arg_verb_spans_for_all_docs_path)
        else:
            self.roberta_features = self.preprocess_roberta_features()
            self.roberta_doc_arg_verb_spans_for_all_docs = self.convert_word_span_to_roberta_token_span()

    def get_input_file_names(self):
        return ["mfc_graphs_sents_complete_immigration_5933.json",
                "mfc_graphs_sents_complete_immigration_unlabelled_41966.json"]

    def __len__(self):
        return self.dataset_size

    def preprocess_roberta_features(self):
        text_a_tokens_for_all_docs = list()
        end_offsets_for_all_docs = list()
        start_offsets_for_all_docs = list()
        num_sents_by_doc = list()
        cumulative_doc_start_indices = list()
        cumulative_doc_end_indices = list()
        doc_num_sents_cumulative = 0

        config_class, model_class, tokenizer_class = MODEL_CLASSES[config_args['base_type']]
        tokenizer = tokenizer_class.from_pretrained(config_args['model_name'])

        for i in tqdm(range(self.dataset_size)):
            doc = self.raw_data[i]
            doc_words = doc[-2]
            start_offsets_for_one_doc = list()
            end_offsets_for_one_doc = list()
            text_a_tokens_for_one_doc = list()
            num_sents_by_doc.append(len(doc_words))
            cumulative_doc_start_indices.append(doc_num_sents_cumulative)
            doc_num_sents_cumulative += len(doc_words)
            cumulative_doc_end_indices.append(doc_num_sents_cumulative)
            for sent_words in doc_words:
                bpe_tokens = list()
                bpe_flattened_for_one_sent = list()
                start_offsets_one_sent = list()
                end_offsets_one_sent = list()
                cumulative = 0
                for j, t in enumerate(sent_words):
                    if j != 0:
                        temp_ = tokenizer.tokenize(" ".join(['X', t]))[1:]  # Remove the X token
                    else:
                        temp_ = tokenizer.tokenize(t)
                    bpe_tokens.append(temp_)
                    bpe_flattened_for_one_sent.extend(temp_)
                    start_offsets_one_sent.append(cumulative + 1)  # Pos 0 is reserved for <s>
                    end_offsets_one_sent.append(cumulative + len(temp_))
                    cumulative += (len(temp_))

                start_offsets_for_one_doc.append(start_offsets_one_sent)
                end_offsets_for_one_doc.append(end_offsets_one_sent)
                text_a_tokens_for_one_doc.append(bpe_flattened_for_one_sent)

            text_a_tokens_for_all_docs.append(text_a_tokens_for_one_doc)
            start_offsets_for_all_docs.append(start_offsets_for_one_doc)
            end_offsets_for_all_docs.append(end_offsets_for_one_doc)

        all_input_ids, all_input_masks, all_segment_ids, all_label_ids = load_bert_features_sents(
            config_args['task_name'],
            tokenizer, self.doc_labels, text_a_tokens_for_all_docs=text_a_tokens_for_all_docs)

        roberta_features = {
            "all_input_ids": all_input_ids,
            "all_input_masks": all_input_masks,
            "all_label_ids": all_label_ids,
            "num_sents_by_doc": torch.LongTensor(num_sents_by_doc),
            "start_offsets_for_all_docs": start_offsets_for_all_docs,
            "end_offsets_for_all_docs": end_offsets_for_all_docs,
            "cumulative_doc_start_indices": cumulative_doc_start_indices,
            "cumulative_doc_end_indices": cumulative_doc_end_indices
        }

        torch.save(roberta_features, self.processed_roberta_path)
        print("successful loading bert features!")
        return roberta_features

    def span_convert(self, predicate_arg_verbs_dict, sent_start_offsets, sent_end_offsets):
        roberta_predicate_arg_verbs_dict = dict()
        for k, v in predicate_arg_verbs_dict.items():
            roberta_predicate_arg_verbs_dict[k] = [
                min(sent_start_offsets[v[0][0]], (int(config_args['max_seq_length']) - 1)),
                min(sent_end_offsets[v[0][1]], (int(config_args['max_seq_length']) - 1))]
        return roberta_predicate_arg_verbs_dict

    def convert_word_span_to_roberta_token_span(self):
        roberta_doc_arg_verb_spans_for_all_docs = list()
        for i in tqdm(range(len(self.roberta_features['start_offsets_for_all_docs']))):
            # print("i" + str(i))
            raw_doc = self.raw_data[i]
            doc_arg_verb_spans = raw_doc[4]
            roberta_doc_arg_verb_spans = list()
            for j, sent_arg_verb_spans in enumerate(doc_arg_verb_spans):
                # print("j " + str(j))
                roberta_sent_arg_verb_spans = list()
                for k, predicate_arg_verbs_dict in enumerate(sent_arg_verb_spans):
                    # print("k " + str(k))
                    roberta_sent_arg_verb_spans.append(self.span_convert(predicate_arg_verbs_dict,
                                                                         self.roberta_features[
                                                                             'start_offsets_for_all_docs'][i][j],
                                                                         self.roberta_features[
                                                                             'end_offsets_for_all_docs'][i][j]))

                roberta_doc_arg_verb_spans.append(roberta_sent_arg_verb_spans)
            roberta_doc_arg_verb_spans_for_all_docs.append(roberta_doc_arg_verb_spans)

        serialize_to_file(self.roberta_doc_arg_verb_spans_for_all_docs_path, roberta_doc_arg_verb_spans_for_all_docs)
        return roberta_doc_arg_verb_spans_for_all_docs

    def __getitem__(self, index):  # This is absolute index in terms of the entire dataset

        doc_id = index
        doc_type = 1 if index < self.num_labelled else 0  # 0 means unlabelled and 1 means labelled
        doc_label_id = self.doc_labels[index]
        cumulative_doc_start_index = self.roberta_features['cumulative_doc_start_indices'][index]
        cumulative_doc_end_index = self.roberta_features['cumulative_doc_end_indices'][index]
        num_sents = self.roberta_features['num_sents_by_doc'][index].squeeze().item()
        # Here, we will load the roberta features as group each document
        roberta_input_ids = self.roberta_features['all_input_ids'][cumulative_doc_start_index: cumulative_doc_end_index]
        roberta_input_masks = self.roberta_features['all_input_masks'][
                              cumulative_doc_start_index: cumulative_doc_end_index]
        roberta_doc_arg_verb_spans = self.roberta_doc_arg_verb_spans_for_all_docs[index]
        return roberta_input_ids, roberta_input_masks, roberta_doc_arg_verb_spans, num_sents, doc_label_id, doc_id, doc_type


def collate(batch):
    batch_input_ids = torch.cat([item[0] for item in batch], dim=0)
    batch_input_masks = torch.cat([item[1] for item in batch], dim=0)
    batch_num_sents = torch.LongTensor([item[3] for item in batch])
    batch_cumulative_sents_start_indices_by_doc = list()
    batch_cumulative_sents_end_indices_by_doc = list()
    batch_cumulative = 0
    for num_sents in batch_num_sents:
        batch_cumulative_sents_start_indices_by_doc.append(batch_cumulative)
        batch_cumulative += num_sents.item()
        batch_cumulative_sents_end_indices_by_doc.append(batch_cumulative)
    batch_cumulative_sents_start_indices_by_doc = torch.LongTensor(batch_cumulative_sents_start_indices_by_doc)
    batch_cumulative_sents_end_indices_by_doc = torch.LongTensor(batch_cumulative_sents_end_indices_by_doc)

    batch_labels = torch.LongTensor([item[4] for item in batch])
    batch_doc_ids = torch.LongTensor([item[5] for item in batch])
    batch_doc_types = torch.FloatTensor([item[6] for item in batch])
    batch_roberta_doc_arg_verb_spans = [item[2] for item in batch]
    return batch_input_ids, batch_input_masks, batch_roberta_doc_arg_verb_spans, batch_num_sents, \
           batch_cumulative_sents_start_indices_by_doc, batch_cumulative_sents_end_indices_by_doc, batch_labels, batch_doc_ids, batch_doc_types
