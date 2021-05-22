from friss.models.model_utils import *
from transformers import BertPreTrainedModel, RobertaModel
import math


class RoBerta_baseline(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = GlobalAttentionHeadReluMean(config)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.classifier.apply(init_weights_relu)

        self.frame_classification_loss_sent = nn.NLLLoss()

    def forward(self, batch, isTrain=True):
        outputs = self.roberta(
            batch[0],
            attention_mask=batch[1],
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )

        roberta_tokens_embeddings_by_sent = outputs[0]

        batch_cumulative_sents_start_indices_by_doc = batch[4]
        batch_cumulative_sents_end_indices_by_doc = batch[5]
        batch_labels = batch[6]
        batch_size = batch_labels.size(0)

        num_tokens_by_sent = batch[1].sum(dim=1).float().contiguous().view(-1, 1)

        sent_mean_tokens_embeddings_by_sent = torch.sum(roberta_tokens_embeddings_by_sent, dim=1) / num_tokens_by_sent
        log_probs_sent = self.classifier(sent_mean_tokens_embeddings_by_sent,
                                         batch_cumulative_sents_start_indices_by_doc,
                                         batch_cumulative_sents_end_indices_by_doc, batch_size)

        log_probs_sent = self.log_softmax(log_probs_sent)
        ce_loss_sent = self.frame_classification_loss_sent(log_probs_sent.contiguous().view(-1, self.num_labels),
                                                           batch_labels)

        loss = ce_loss_sent

        log_probs = log_probs_sent

        outputs = (loss, log_probs, None, 0, 0, 0) + outputs[2:]

        return outputs  # (loss), log_probs, (hidden_states), (attentions)


class FRISS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.tags = config.tags
        self.num_k = config.num_k
        self.alpha = config.alpha
        self.beta = config.beta
        self.dictionary = DictionaryLearningGumbelSoftmaxMultiView(config, v=len(self.tags), n=2, k=self.num_k)
        self.classifier = GlobalAttentionHeadReluMean(config)
        self.dictionary.apply(init_weights_relu)
        self.classifier.apply(init_weights_relu)

        self.margin = config.margin
        self.schedule_r = config.schedule_r
        self.temperature_limit = config.temperature_limit
        self.evaluate_gumbel_temperature = config.evaluate_temperature
        self.frame_classification_loss_sent = nn.CrossEntropyLoss()
        self.frame_classification_loss_dic = nn.CrossEntropyLoss()

        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.gumbel_temperature = 1.0

    def forward(self, batch, isTrain=True):
        outputs = self.roberta(
            batch[0],
            attention_mask=batch[1],
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        # 64 is max num of tokens for each sentence
        roberta_tokens_embeddings_by_sent = outputs[0]  # num_sents x 64 x 768
        batch_roberta_doc_arg_verb_spans = batch[2]
        batch_cumulative_sents_start_indices_by_doc = batch[4]
        batch_cumulative_sents_end_indices_by_doc = batch[5]
        batch_labels = batch[6]
        batch_mask = batch[8].bool()
        global_step = batch[9]
        batch_size = batch_labels.size(0)

        if global_step % 50 == 0:
            self.gumbel_temperature = max(self.temperature_limit, math.exp(self.schedule_r * global_step))

        num_tokens_by_sent = batch[1].sum(dim=1).float().contiguous().view(-1, 1)
        sent_mean_tokens_embeddings_all_tags, tags_embeddings_all_tags, tags_numbers_by_sent, tags_numbers, tags_spans_by_tag = \
            get_selected_srl_units_embeddings(roberta_tokens_embeddings_by_sent, batch_roberta_doc_arg_verb_spans,
                                              num_tokens_by_sent, tags=self.tags)

        available_tags = [tag for i, tag in enumerate(self.tags) if tags_numbers[i] > 0]

        dictionary_learning_input_all_tags = torch.cat([tags_embeddings_all_tags, sent_mean_tokens_embeddings_all_tags],
                                                       dim=1)
        dictionary_weights, recontructed_embeddings, dictionary_logits, negative_frame_embeddings, reverse_dictionary_probs = self.dictionary(
            dictionary_learning_input_all_tags,
            tags_numbers, temperature=self.gumbel_temperature)

        dictionary_logits_by_tag_list = matrix_to_list_by_length(dictionary_logits, tags_numbers)

        dictionary_logits_by_doc_list, pseudo_tags_padding_mask, tags_mask_by_doc = batch_embeddings_by_doc_all_tags(
            dictionary_logits_by_tag_list,
            batch_cumulative_sents_start_indices_by_doc,
            batch_cumulative_sents_end_indices_by_doc,
            tags_numbers_by_sent, available_tags,
            batch_size, batch_first=True)

        dictionary_log_probs_by_doc_list = [dictionary_logits_by_doc_list[i] * tags_mask_by_doc[tag]
                                            for i, tag in enumerate(available_tags)]
        dictionary_log_probs_by_doc = torch.stack(dictionary_log_probs_by_doc_list,
                                                  dim=0)  # num_dictionaries x num_docs x num_frames
        log_probs_from_dictionary_learning = dictionary_log_probs_by_doc.sum(dim=0)[batch_mask]

        sent_mean_tokens_embeddings_by_sent = torch.sum(roberta_tokens_embeddings_by_sent, dim=1) / num_tokens_by_sent

        log_probs_sent = self.classifier(sent_mean_tokens_embeddings_by_sent, batch_cumulative_sents_start_indices_by_doc,
                            batch_cumulative_sents_end_indices_by_doc, batch_size, batch_mask=batch_mask)


        ce_loss_sent = self.frame_classification_loss_sent(
            log_probs_sent.contiguous().view(-1, self.num_labels),
            batch_labels[batch_mask])
        ce_loss_dic = self.frame_classification_loss_dic(
            log_probs_from_dictionary_learning.contiguous().view(-1, self.num_labels),
            batch_labels[batch_mask])

        # Triplet Loss
        negative_tags_embeddings_all_tags = list()
        positive_tags_embeddings_all_tags = list()
        reconstructed_embeddings_all_tags = list()
        tags_embeddings_all_tags_list = matrix_to_list_by_length(tags_embeddings_all_tags, tags_numbers)
        recontructed_embeddings_all_tags_list = matrix_to_list_by_length(recontructed_embeddings, tags_numbers)
        for i, tags_embeddings_one_tag in enumerate(tags_embeddings_all_tags_list):
            tag_size = tags_embeddings_one_tag.size(0)
            positive_tags_embeddings_all_tags.append(tags_embeddings_one_tag.repeat(self.num_labels - self.num_k, 1))
            reconstructed_embeddings_all_tags.append(
                recontructed_embeddings_all_tags_list[i].repeat(self.num_labels - self.num_k, 1))
            neg_tags_embeddings_one_tag = list()
            for j in range(self.num_labels - self.num_k):
                neg_tags_embeddings_one_tag.append(tags_embeddings_one_tag[torch.randperm(tag_size)])

            neg_tags_embeddings_one_tag = torch.cat(neg_tags_embeddings_one_tag, dim=0)
            negative_tags_embeddings_all_tags.append(neg_tags_embeddings_one_tag)

        positive_tags_embeddings_all_tags = torch.cat(positive_tags_embeddings_all_tags, dim=0)
        negative_tags_embeddings_all_tags = torch.cat(negative_tags_embeddings_all_tags, dim=0)
        reconstructed_embeddings_all_tags = torch.cat(reconstructed_embeddings_all_tags, dim=0)
        triplet_loss = self.triplet_loss(reconstructed_embeddings_all_tags, positive_tags_embeddings_all_tags,
                                         negative_tags_embeddings_all_tags)

        loss = ce_loss_sent + self.alpha * ce_loss_dic + self.beta * triplet_loss

        log_probs = log_probs_sent + log_probs_from_dictionary_learning

        outputs = (loss, log_probs, dictionary_log_probs_by_doc_list, ce_loss_sent, ce_loss_dic, triplet_loss) + outputs[2:]

        return outputs  # (loss), log_probs, (hidden_states), (attentions)

    def evaluate(self, tokenizer, batch):
        outputs = self.roberta(
            batch[0],
            attention_mask=batch[1],
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        # 64 is max num of tokens for each sentence
        roberta_tokens_embeddings_by_sent = outputs[0]  # num_sents x 64 x 768
        batch_roberta_doc_arg_verb_spans = batch[2]
        batch_cumulative_sents_start_indices_by_doc = batch[4]
        batch_cumulative_sents_end_indices_by_doc = batch[5]
        batch_labels = batch[6]
        batch_size = batch_labels.size(0)

        num_tokens_by_sent = batch[1].sum(dim=1).float().contiguous().view(-1, 1)
        # tag_numbers shows the number of tags for each tag type in the batch
        sent_mean_tokens_embeddings_all_tags, tags_embeddings_all_tags, tags_numbers_by_sent, tags_numbers, tags_spans_by_tag = \
            get_selected_srl_units_embeddings(roberta_tokens_embeddings_by_sent, batch_roberta_doc_arg_verb_spans,
                                              num_tokens_by_sent, tags=self.tags)

        # filter out tags_numbers_by_sent and tags_spans_by_tag
        available_tags = [tag for i, tag in enumerate(self.tags) if tags_numbers[i] > 0]
        tags_spans_by_tag = [tag_spans for i, tag_spans in enumerate(tags_spans_by_tag) if tags_numbers[i] > 0]
        tags_numbers_by_sent = [tag_num for i, tag_num in enumerate(tags_numbers_by_sent) if
                                tags_numbers[i] > 0]

        batch_srl_tags_ids, batch_srl_tags_texts, batch_srl_sents_tags_texts = get_texts_by_tags(
            batch[0], tokenizer, tags_numbers_by_sent, available_tags, tags_spans_by_tag)

        dictionary_learning_input_all_tags = torch.cat([tags_embeddings_all_tags, sent_mean_tokens_embeddings_all_tags],
                                                       dim=1)

        gumbel_temperature = self.evaluate_gumbel_temperature if self.evaluate_gumbel_temperature > 0 else self.gumbel_temperature

        dictionary_weights, recontructed_embeddings, dictionary_logits, negative_frame_embeddings, reverse_dictionary_probs, top_k_weights_and_indices_tags, top_k_tags_list, top_k_metrics_and_indices_tags, top_k_tags_list_metric = self.dictionary.evaluate(
            dictionary_learning_input_all_tags, tags_numbers, batch_srl_tags_texts, batch_srl_sents_tags_texts, "l2",
            temperature=gumbel_temperature)

        dictionary_logits_by_tag_list = matrix_to_list_by_length(dictionary_logits, tags_numbers)

        dictionary_logits_by_doc_list, pseudo_tags_padding_mask, tags_mask_by_doc = batch_embeddings_by_doc_all_tags(
            dictionary_logits_by_tag_list,
            batch_cumulative_sents_start_indices_by_doc,
            batch_cumulative_sents_end_indices_by_doc,
            tags_numbers_by_sent, available_tags,
            batch_size, batch_first=True)

        dictionary_log_probs_by_doc_list = [dictionary_logits_by_doc_list[i] * tags_mask_by_doc[tag]
                                            for i, tag in enumerate(available_tags)]
        dictionary_log_probs_by_doc = torch.stack(dictionary_log_probs_by_doc_list,
                                                  dim=0)  # num_dictionaries x num_docs x num_frames
        log_probs_from_dictionary_learning = dictionary_log_probs_by_doc.sum(dim=0)

        sent_mean_tokens_embeddings_by_sent = torch.sum(roberta_tokens_embeddings_by_sent, dim=1) / num_tokens_by_sent
        log_probs_sent = self.classifier(sent_mean_tokens_embeddings_by_sent, batch_cumulative_sents_start_indices_by_doc,
                            batch_cumulative_sents_end_indices_by_doc, batch_size)

        # Cross-Entropy Loss
        ce_loss_sent = self.frame_classification_loss_sent(
            log_probs_sent.contiguous().view(-1, self.num_labels),
            batch_labels)
        ce_loss_dic = self.frame_classification_loss_dic(
            log_probs_from_dictionary_learning.contiguous().view(-1, self.num_labels),
            batch_labels)

        # Triplet Loss
        negative_tags_embeddings_all_tags = list()
        positive_tags_embeddings_all_tags = list()
        reconstructed_embeddings_all_tags = list()
        tags_embeddings_all_tags_list = matrix_to_list_by_length(tags_embeddings_all_tags, tags_numbers)
        recontructed_embeddings_all_tags_list = matrix_to_list_by_length(recontructed_embeddings, tags_numbers)
        for i, tags_embeddings_one_tag in enumerate(tags_embeddings_all_tags_list):
            tag_size = tags_embeddings_one_tag.size(0)
            positive_tags_embeddings_all_tags.append(tags_embeddings_one_tag.repeat(self.num_labels - self.num_k, 1))
            reconstructed_embeddings_all_tags.append(
                recontructed_embeddings_all_tags_list[i].repeat(self.num_labels - self.num_k, 1))
            neg_tags_embeddings_one_tag = list()
            for j in range(self.num_labels - self.num_k):
                neg_tags_embeddings_one_tag.append(tags_embeddings_one_tag[torch.randperm(tag_size)])

            neg_tags_embeddings_one_tag = torch.cat(neg_tags_embeddings_one_tag, dim=0)
            negative_tags_embeddings_all_tags.append(neg_tags_embeddings_one_tag)

        positive_tags_embeddings_all_tags = torch.cat(positive_tags_embeddings_all_tags, dim=0)
        negative_tags_embeddings_all_tags = torch.cat(negative_tags_embeddings_all_tags, dim=0)
        reconstructed_embeddings_all_tags = torch.cat(reconstructed_embeddings_all_tags, dim=0)
        triplet_loss = self.triplet_loss(reconstructed_embeddings_all_tags, positive_tags_embeddings_all_tags,
                                         negative_tags_embeddings_all_tags)

        loss = ce_loss_sent + self.alpha * ce_loss_dic + self.beta * triplet_loss

        log_probs = log_probs_sent + log_probs_from_dictionary_learning

        outputs = (loss, log_probs, dictionary_log_probs_by_doc_list,
                   ce_loss_sent, ce_loss_dic, triplet_loss) + outputs[2:]

        return outputs, top_k_weights_and_indices_tags, top_k_tags_list, top_k_metrics_and_indices_tags, top_k_tags_list_metric