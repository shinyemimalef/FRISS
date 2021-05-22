# this is a pytorch model that combine bert and gcn
from collections import defaultdict
import torch
from torch import nn
from torch.nn import Linear
from torch.nn.functional import softmax, one_hot, log_softmax
from torch.nn.utils.rnn import pad_sequence

from friss.configs.config import config_args
from friss.utils import device


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps / masked_sums)


def init_weights_relu(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


def batch_sent_embeddings_by_doc(sent_embeddings, sent_start_indices_by_doc,
                                 sent_end_indices_by_doc, batch_size, batch_first=True):
    sent_embeddings_by_doc = [sent_embeddings[sent_start_indices_by_doc[i]: sent_end_indices_by_doc[i]] for i in
                              range(batch_size)]

    sent_embeddings_by_doc_padding_mask = [torch.ones(i.size(0)) for i in sent_embeddings_by_doc]
    sent_embeddings_by_doc_padding_mask = pad_sequence(sent_embeddings_by_doc_padding_mask, batch_first=True,
                                                       padding_value=0).to(device)
    sent_embeddings_by_doc = pad_sequence(sent_embeddings_by_doc, batch_first=batch_first)

    return sent_embeddings_by_doc, sent_embeddings_by_doc_padding_mask


# input is input_ids(Roberta token id) and output is actual texts
# input_ids is a list of ids (Integer), number of sentences and number of tokens in each sentences (N*64)
def get_texts_by_tags(input_ids, tokenizer, tags_numbers_by_sent, tags, tags_spans_by_tag):
    input_ids_by_tags = [torch.repeat_interleave(input_ids, torch.LongTensor(tags_numbers_by_sent[i]).to(device), dim=0)
                         for i, _ in enumerate(tags)]
    batch_srl_tags_ids = list()
    batch_srl_tags_texts = list()
    batch_srl_sents_tags_texts = list()

    for i, input_ids_by_tag in enumerate(input_ids_by_tags):
        batch_srl_tag_ids = list()
        batch_srl_tag_texts = list()
        batch_srl_sents_tag_texts = list()

        for j, input_ids in enumerate(input_ids_by_tag):
            input_ids_of_the_tag = torch.index_select(input_ids, 0, torch.arange(start=tags_spans_by_tag[i][j][0],
                                                                                 end=tags_spans_by_tag[i][j][
                                                                                         1] + 1).long().to(device))

            batch_srl_tag_ids.append(input_ids_of_the_tag)
            batch_srl_tag_texts.append(tokenizer.decode(input_ids_of_the_tag))
            batch_srl_sents_tag_texts.append(tokenizer.decode(input_ids).replace("<s>", "").replace("</s>", ""))

        batch_srl_tags_ids.append(batch_srl_tag_ids)
        batch_srl_tags_texts.append(batch_srl_tag_texts)
        batch_srl_sents_tags_texts.append(batch_srl_sents_tag_texts)

    return batch_srl_tags_ids, batch_srl_tags_texts, batch_srl_sents_tags_texts


class GlobalAttentionHeadReluMean(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform_nn = Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config_args["classifier_dropout_prob"])
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, start_indices, end_indices, batch_size, batch_mask=None):
        """"""

        x = self.dropout(x)
        x = self.transform_nn(x)  # batch_size x sequence_length x feature_size
        x = self.activation(x)
        x, mask = batch_sent_embeddings_by_doc(x, start_indices, end_indices, batch_size, batch_first=True)
        if batch_mask is not None:
            x = x[batch_mask]
            mask = mask[batch_mask]

        seq_len = mask.sum(dim=1).float().contiguous().view(-1, 1)
        x = torch.sum(x, dim=1) / seq_len  # batch_size x feature_size

        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits


def get_selected_tag_numbers_by_sent(tags, batch_doc_arg_verb_spans):
    tags_numbers_by_sent_dic = defaultdict(list)
    tags_spans_by_tag_dic = defaultdict(list)

    for doc_arg_verb_spans in batch_doc_arg_verb_spans:
        for sent_arg_verb_spans in doc_arg_verb_spans:
            current_tag_count = {key: 0 for key in tags}
            for predicate_arg_verb_dict in sent_arg_verb_spans:
                for tag in tags:
                    if tag in predicate_arg_verb_dict:
                        tags_spans_by_tag_dic[tag].append(predicate_arg_verb_dict[tag])
                        current_tag_count[tag] += 1

            for tag in tags:
                tags_numbers_by_sent_dic[tag].append(current_tag_count[tag])
    tags_numbers_by_sent = [tags_numbers_by_sent_dic[tag] for tag in tags]
    tags_spans_by_tag = [tags_spans_by_tag_dic[tag] for tag in tags]
    return tags_spans_by_tag, tags_numbers_by_sent


# batch_sents_tokens_output is the roberta output on sentence level,
# batch_doc_arg_verb_spans is the srl results for all the tags,
# num_tokens_per_sent is the number of tokens per sentence
def get_one_srl_tag_embeddings(tokens_embeddings_by_sent, num_tokens_by_sent, tag_spans_by_tag, tag_numbers_by_sent):
    # Repeat-interleave the sentence embeddings to tag embeddings based on the number of tags per sentence
    tag_numbers_by_sent = torch.LongTensor(tag_numbers_by_sent).to(device)
    tag_spans_by_tag = torch.LongTensor(tag_spans_by_tag).to(device)
    tokens_embeddings_by_tag = torch.repeat_interleave(tokens_embeddings_by_sent, tag_numbers_by_sent, dim=0)
    num_tokens_by_tag = torch.repeat_interleave(num_tokens_by_sent, tag_numbers_by_sent, dim=0)
    sent_mean_tokens_embeddings_by_tag = torch.sum(tokens_embeddings_by_tag, dim=1) / num_tokens_by_tag

    tag_embeddings_by_tag = list()
    for i, k in enumerate(tokens_embeddings_by_tag):
        start_index = tag_spans_by_tag[i][0]
        end_index = tag_spans_by_tag[i][1] + 1

        tag_multi_tokens_embedding = torch.index_select(k, 0, torch.arange(start=start_index, end=end_index).long().to(
            device))
        tag_mean_tokens_embedding = torch.mean(tag_multi_tokens_embedding, dim=0)  # or torch.max(temp, dim=0)[0]
        tag_embeddings_by_tag.append(tag_mean_tokens_embedding)
    tag_embeddings_by_tag = torch.stack(tag_embeddings_by_tag, dim=0)

    assert sent_mean_tokens_embeddings_by_tag.size() == tag_embeddings_by_tag.size()

    return sent_mean_tokens_embeddings_by_tag, tag_embeddings_by_tag


def get_selected_srl_units_embeddings(tokens_embeddings_by_sent, batch_doc_arg_verb_spans, num_tokens_by_sent,
                                      tags=['B-V', "B-ARGM-MOD", "B-ARGM-DIR", "B-ARGM-NEG", 'B-ARG0',
                                            'B-ARG1', 'B-ARG2', 'B-ARG3', 'B-ARG4']):
    tags_spans_by_tag, tags_numbers_by_sent = get_selected_tag_numbers_by_sent(tags, batch_doc_arg_verb_spans)
    sent_mean_tokens_embeddings = list()  # for all type of tags
    tags_embeddings = list()  # for all type of tags
    tags_numbers = list()
    for i in range(len(tags)):
        tag_numbers_by_sent = tags_numbers_by_sent[i]
        if sum(tag_numbers_by_sent) > 0:
            tag_spans_by_tag = tags_spans_by_tag[i]
            sent_mean_tokens_embeddings_by_tag, tag_embeddings_by_tag = get_one_srl_tag_embeddings(
                tokens_embeddings_by_sent,
                num_tokens_by_sent,
                tag_spans_by_tag,
                tag_numbers_by_sent)
            sent_mean_tokens_embeddings.append(sent_mean_tokens_embeddings_by_tag)
            tags_embeddings.append(tag_embeddings_by_tag)
            tags_numbers.append(tag_embeddings_by_tag.size(0))
        else:
            tags_numbers.append(0)

    sent_mean_tokens_embeddings = torch.cat(sent_mean_tokens_embeddings, dim=0)
    tags_embeddings = torch.cat(tags_embeddings, dim=0)
    return sent_mean_tokens_embeddings, tags_embeddings, tags_numbers_by_sent, tags_numbers, tags_spans_by_tag


def batch_embeddings_by_doc_all_tags(x_by_tags,
                                     batch_cumulative_sents_start_indices_by_doc,
                                     batch_cumulative_sents_end_indices_by_doc,
                                     tags_numbers_by_sent, tags, batch_size, batch_first=True):
    cumulative_num_tags = {tag: 0 for i, tag in enumerate(tags)}
    tags_start_indices_by_doc = defaultdict(list)  # start_indices are inclusive
    tags_end_indices_by_doc = defaultdict(list)  # end_indices are exclusive
    # This is the number of tags for each tag type for each document {tag_type : [number_of_tags_for_doc]}
    tags_numbers_by_doc = defaultdict(list)
    tags_mask_by_doc = defaultdict(list)

    # i is inclusive and j is exclusive
    for i, j in zip(batch_cumulative_sents_start_indices_by_doc, batch_cumulative_sents_end_indices_by_doc):
        for tag in tags:
            tags_start_indices_by_doc[tag].append(cumulative_num_tags[tag])
        for k in range(i, j):  # k is the sentence index
            for m, tag in enumerate(tags):
                cumulative_num_tags[tag] += tags_numbers_by_sent[m][k]
        for tag in tags:
            tags_end_indices_by_doc[tag].append(cumulative_num_tags[tag])
            tags_numbers_by_doc[tag].append(tags_end_indices_by_doc[tag][-1] - tags_start_indices_by_doc[tag][-1])
            tags_mask_by_doc[tag].append(tags_numbers_by_doc[tag][-1] != 0)

    # this is a list of tag_embeddings_by_doc
    tags_embeddings_by_doc = list()
    pseudo_tags_padding_mask = list()
    for i, tag in enumerate(tags):
        tag_embeddings_by_doc = list()
        # batch_size is the number of documents in the batch
        for j in range(batch_size):
            if tags_start_indices_by_doc[tag][j] < tags_end_indices_by_doc[tag][j]:
                tag_embeddings_by_doc.append(
                    x_by_tags[i][tags_start_indices_by_doc[tag][j]:tags_end_indices_by_doc[tag][j]])
            else:
                tag_embeddings_by_doc.append(torch.zeros(1, x_by_tags[i].size(1)).to(
                    device))

        tags_mask_by_doc[tag] = torch.Tensor(tags_mask_by_doc[tag]).float().contiguous().view(-1, 1).to(device)

        # The pseudo_tag_numbers_by_doc is different from tag_numbers_by_doc when a doc does not contain any tag
        # And in this case, paseudo_tag_numbers considers 1, where tag_numbers would be 0.
        pseudo_tag_numbers_by_doc = torch.Tensor([q.size(0) for q in tag_embeddings_by_doc]).float().contiguous().view(
            -1, 1).to(device)
        pseudo_tag_padding_mask = [torch.ones(int(q[0].item())) for q in pseudo_tag_numbers_by_doc]
        pseudo_tag_padding_mask = pad_sequence(pseudo_tag_padding_mask, batch_first=batch_first,
                                               padding_value=0).to(device)  # doc_num x max_tag_num
        tag_embeddings_by_doc = pad_sequence(tag_embeddings_by_doc, batch_first=batch_first,
                                             padding_value=0)  # doc_num x max_tag_num x features_dim


        tag_embeddings_by_doc = torch.sum(tag_embeddings_by_doc, dim=1) / pseudo_tag_numbers_by_doc
        tags_embeddings_by_doc.append(tag_embeddings_by_doc)

        pseudo_tags_padding_mask.append(pseudo_tag_padding_mask)

    return tags_embeddings_by_doc, pseudo_tags_padding_mask, tags_mask_by_doc


def matrix_to_list_by_length(m, nums):
    m_list = list()
    i = 0
    for n in nums:
        if n > 0:
            m_list.append(m[i:i + n, :])
            i += n
    return m_list


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(
        shape).to(
        device)  # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, t):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.size())
    return softmax(y / t, dim=-1)


def gumbel_logsoftmax_sample(logits, t):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.size())
    return log_softmax(y / t, dim=-1)


def custom_gumbel_softmax(logits, temperature, hard=False, log=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    if log:
        y = gumbel_logsoftmax_sample(logits, temperature)
    else:
        y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard
    return y


def orthogonal_regularization(param, reg=1e-4):
    sym = torch.mm(param, torch.t(param))
    sym -= torch.eye(param.shape[0]).to(device)
    orth_loss = reg * sym.abs().sum()
    return orth_loss


class DictionaryLearningGumbelSoftmaxMultiView(torch.nn.Module):
    def __init__(self, config, v=3, n=1, k=1, c=15):
        super().__init__()
        self.c = c
        self.dictionary_embeddings = [nn.Parameter(torch.Tensor(self.c, config.hidden_size).to(device)) for i
                                      in range(v)]
        for dic_w in self.dictionary_embeddings:
            torch.nn.init.xavier_uniform_(dic_w.data, nn.init.calculate_gain('relu'))
        # This is a linear transformation layer shared between all views
        self.transform_nn = Linear(config.hidden_size * n, config.hidden_size)
        # This is a list of linear projection layers for each view respectively
        self.proj_nn = nn.ModuleList(Linear(config.hidden_size, self.c) for i in range(v))
        self.dropout = nn.Dropout(config_args["dictionary_learning_dropout_prob"])
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = nn.ReLU()
        self.k = k

    def forward(self, x, tags_numbers, temperature=0.5):
        x = self.dropout(x)
        x = self.transform_nn(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.dropout(x)

        x_list = matrix_to_list_by_length(x, tags_numbers)

        j = 0
        logits_list = list()
        for i, l in enumerate(self.proj_nn):
            if tags_numbers[i] > 0:
                logits_list.append(l(x_list[j]))
                j += 1

        dictionary_logits = torch.cat(logits_list, dim=0)
        dictionary_probs = softmax(dictionary_logits, dim=1)
        reverse_dictionary_probs = torch.square(1 - dictionary_probs)
        dictionary_gumbel_weights = custom_gumbel_softmax(dictionary_logits, temperature, hard=False)

        _, pos_inds = torch.topk(dictionary_gumbel_weights, self.k)  # neg_inds dim:  total_number_tags x k
        pos_inds_one_hot = one_hot(pos_inds, num_classes=self.c).sum(dim=1)
        neg_inds = (pos_inds_one_hot == 0).nonzero()[:, 1].reshape(-1, self.c - self.k)
        assert neg_inds.size(0) == dictionary_gumbel_weights.size(0)

        neg_inds_list = matrix_to_list_by_length(neg_inds, tags_numbers)

        neg_samples_list = list()
        j = 0
        for i, dic_embed in enumerate(self.dictionary_embeddings):
            if tags_numbers[i] > 0:
                for instance_index, neg_inds_of_this_index in enumerate(neg_inds_list[j]):
                    neg_samples_list.append(torch.index_select(dic_embed, 0, neg_inds_of_this_index))
                j += 1
        neg_samples_embeddings = torch.stack(neg_samples_list, dim=0)  # neg_samples_dim : total_number_tags x 14 x 768

        reverse_dictionary_probs_list = list()
        for i, neg_ind in enumerate(neg_inds):
            reverse_dictionary_probs_list.append(
                softmax(torch.index_select(reverse_dictionary_probs[i], 0, neg_ind), dim=0))
        reverse_dictionary_probs = torch.stack(reverse_dictionary_probs_list, dim=0)

        dictionary_weights_list = matrix_to_list_by_length(dictionary_gumbel_weights, tags_numbers)

        j = 0
        reconstructed_embeddings_list = list()
        for i, dic_embed in enumerate(self.dictionary_embeddings):
            if tags_numbers[i] > 0:
                reconstructed_embeddings_list.append(torch.chain_matmul(dictionary_weights_list[j], dic_embed))
                j += 1

        recontructed_embeddings = torch.cat(reconstructed_embeddings_list, dim=0)

        return dictionary_gumbel_weights, recontructed_embeddings, dictionary_logits, neg_samples_embeddings, reverse_dictionary_probs

    # This returns the top-k for each row of dictionary based on the dictionary weights
    # texts_by_tags is a list of texts_by_tag which is texts by each tag, and if the tag doesn't have any texts, it would not show up in the list
    def evaluate(self, x, tags_numbers, tags_texts, sents_tags_texts, distance_measure, k=100, temperature=0.5):

        x = self.dropout(x)
        x = self.transform_nn(x)
        x = self.activation(x)

        x = self.dropout(x)

        x_list = matrix_to_list_by_length(x, tags_numbers)

        j = 0
        logits_list = list()
        for i, l in enumerate(self.proj_nn):
            if tags_numbers[i] > 0:
                logits_list.append(l(x_list[j]))
                j += 1

        dictionary_logits = torch.cat(logits_list, dim=0)
        dictionary_probs = softmax(dictionary_logits, dim=1)

        reverse_dictionary_probs = torch.square(1 - dictionary_probs)
        dictionary_gumbel_weights = custom_gumbel_softmax(dictionary_logits, temperature, hard=False)

        _, pos_inds = torch.topk(dictionary_gumbel_weights, self.k)  # neg_inds dim:  total_number_tags x k
        pos_inds_one_hot = one_hot(pos_inds, num_classes=self.c).sum(dim=1)
        neg_inds = (pos_inds_one_hot == 0).nonzero()[:, 1].reshape(-1, self.c - self.k)
        assert neg_inds.size(0) == dictionary_gumbel_weights.size(0)

        neg_inds_list = matrix_to_list_by_length(neg_inds, tags_numbers)
        neg_samples_list = list()
        j = 0
        for i, dic_embed in enumerate(self.dictionary_embeddings):
            if tags_numbers[i] > 0:
                for instance_index, neg_inds_of_this_index in enumerate(neg_inds_list[j]):
                    neg_samples_list.append(torch.index_select(dic_embed, 0, neg_inds_of_this_index))
                j += 1
        neg_samples_embeddings = torch.stack(neg_samples_list, dim=0)  # neg_samples_dim : total_number_tags x 14 x 768

        reverse_dictionary_probs_list = list()
        for i, neg_ind in enumerate(neg_inds):
            reverse_dictionary_probs_list.append(
                softmax(torch.index_select(reverse_dictionary_probs[i], 0, neg_ind), dim=0))
        reverse_dictionary_probs = torch.stack(reverse_dictionary_probs_list, dim=0)

        dictionary_weights_list = matrix_to_list_by_length(dictionary_gumbel_weights, tags_numbers)

        j = 0
        reconstructed_embeddings_list = list()
        for i, dic_embed in enumerate(self.dictionary_embeddings):
            if tags_numbers[i] > 0:
                reconstructed_embeddings_list.append(torch.chain_matmul(dictionary_weights_list[j], dic_embed))
                j += 1

        reconstructed_embeddings = torch.cat(reconstructed_embeddings_list, dim=0)

        # list of tuples with the first element the weights, and the second indices
        # Top K weights and indices for each frame of all the srl tags

        # The metrics can be cosine_similarity, l2_distance, or inner_product
        if distance_measure == "l2":
            top_k_metrics_and_indices_tags = [
                torch.topk(torch.cdist(self.dictionary_embeddings[i], x_by_tag, p=2), min(k, x_by_tag.size(0)),
                           largest=False, sorted=True) for i, x_by_tag in enumerate(x_list)]

        else:
            top_k_metrics_and_indices_tags = [
                torch.topk(torch.einsum('ik,jk->ij', self.dictionary_embeddings[i], x_by_tag), min(k, x_by_tag.size(0)),
                           largest=True, sorted=True) for i, x_by_tag in enumerate(x_list)]

        top_k_tags_list_metric = list()
        for i, top_k_weights_and_indices in enumerate(top_k_metrics_and_indices_tags):
            top_k_indices = top_k_weights_and_indices[1]
            # this list contains a list of top_k_tag_list for each frame
            top_k_list_for_one_tag = list()
            for top_k_indices_for_one_frame in top_k_indices:
                top_k_sents_for_one_frame = [tags_texts[i][j] for j in top_k_indices_for_one_frame]
                top_k_tags_for_one_frame = [sents_tags_texts[i][j] for j in top_k_indices_for_one_frame]
                top_k_list_for_one_tag.append((top_k_sents_for_one_frame, top_k_tags_for_one_frame))
            top_k_tags_list_metric.append(top_k_list_for_one_tag)

        # this list contains a list of top_k_tag_list
        top_k_tags_list = list()
        for i, tags_texts_by_tag in enumerate(tags_texts):
            top_k_list_for_one_tag = list()
            for j in range(self.c):
                top_k_list_for_one_tag.append((tags_texts[i], sents_tags_texts[i]))
            top_k_tags_list.append(top_k_list_for_one_tag)

        return dictionary_gumbel_weights, reconstructed_embeddings, dictionary_logits, neg_samples_embeddings, \
               reverse_dictionary_probs, dictionary_weights_list, top_k_tags_list, top_k_metrics_and_indices_tags, top_k_tags_list_metric
