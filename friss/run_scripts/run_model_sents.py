from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed
from transformers import RobertaConfig, RobertaTokenizer

from friss.utils import (load_data, serialize_to_file, compute_metrics)
from friss.MfcSentsDataset import collate
from friss.configs.config import issues_config, MFC_mlm_result_directory
from friss.dataset_utils import load_sliced_dataset
from friss.models.sents_model import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

MODEL_CLASSES = {
    'roberta_baseline': (RobertaConfig, RoBerta_baseline, RobertaTokenizer),
    'friss': (RobertaConfig, FRISS, RobertaTokenizer),
}

MODEL_CLASSES_TYPES = {
    'friss': True,
    'roberta_baseline': False
}


def train(model, tokenizer, fold, dataset_splits, tb_writer):
    def set_learning_rates():
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = list()
        if hasattr(model, 'classifier'):
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': config_args["classifier_lr"], 'weight_decay': config_args['weight_decay']})

            optimizer_grouped_parameters.append(
                {'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': config_args["classifier_lr"], 'weight_decay': 0.0})

        if hasattr(model, config_args['base_type']):
            optimizer_grouped_parameters.append({'params': [p for n, p in
                                                            getattr(model, config_args['base_type']).named_parameters()
                                                            if not any(nd in n for nd in no_decay)],
                                                 'lr': config_args["lm_learning_rate"],
                                                 'weight_decay': config_args['weight_decay']})
            optimizer_grouped_parameters.append({'params': [p for n, p in
                                                            getattr(model, config_args['base_type']).named_parameters()
                                                            if any(nd in n for nd in no_decay)],
                                                 'lr': config_args["lm_learning_rate"],
                                                 'weight_decay': 0.0})

        if hasattr(model, 'dictionary'):
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in model.dictionary.named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'lr': config_args["dictionary_lr"], 'weight_decay': config_args['weight_decay']})
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in model.dictionary.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': config_args["dictionary_lr"], 'weight_decay': 0.0})
            optimizer_grouped_parameters.append(
                {'params': model.dictionary.dictionary_embeddings,
                 'lr': config_args["dictionary_lr"], 'weight_decay': config_args['weight_decay']})

        return optimizer_grouped_parameters

    def pre_training_setup():

        train_dataset, sampler = load_sliced_dataset(dataset_splits, fold, 'train',
                                                     batch_size_labeled=config_args['train_batch_size_labeled'],
                                                     batch_size_unlabeled=config_args['train_batch_size_unlabeled'],
                                                     num_unlabelled=config_args['num_unlabelled'])

        train_dataloader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate,
                                      num_workers=config_args['num_workers'])

        t_total = len(train_dataloader) // config_args['gradient_accumulation_steps'] * config_args['num_train_epochs']

        optimizer_grouped_parameters = set_learning_rates()

        optimizer = AdamW(optimizer_grouped_parameters, lr=config_args['classifier_lr'],
                          weight_decay=config_args['weight_decay'],
                          eps=config_args['adam_epsilon'])

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config_args['warmup_steps'],
                                                    num_training_steps=t_total)

        return train_dataset, train_dataloader, optimizer, scheduler, t_total

    def log_before_training():
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", config_args['num_train_epochs'])
        logger.info("  Total train batch size  = %d",
                    config_args['train_batch_size_labeled'] + config_args['train_batch_size_unlabeled'])
        logger.info("  Gradient Accumulation steps = %d", config_args['gradient_accumulation_steps'])
        logger.info("  Total optimization steps = %d", t_total)

    def write_results_to_eval_file(v_measure, mcc, confusion_matrix, class_report, acc, f1, prefix, mode='test'):

        with open(config_args['output_eval_file'], "a+") as writer:
            writer.write(
                "mode-{}-iter-{}-issue-{}-fold-{}-accuracy-{}-f1-{}\n".format(mode, prefix, config_args['issue'], fold,
                                                                              acc, f1))
            writer.write("mcc:\n{}\n".format(mcc))
            writer.write("confusion_matrix:\n{}\n".format(str(confusion_matrix)))
            writer.write("class_report:\n{}\n".format(str(class_report)))
            writer.write("v_measure: {}\n".format(str(v_measure)))
            logger.info(
                "mode-{}-prefix-{}-issue-{}-fold-{}-accuracy-{}-f1-{}\n".format(mode, prefix, config_args['issue'],
                                                                                fold, acc, f1))

    def log_metrics(model, tokenizer, fold, dataset, best_acc, global_step, mode):

        eval_losses_avg, mcc, confusion_matrix, class_report, acc, f1 = validate_test(
            model, tokenizer, fold, dataset, mode, dictionary=MODEL_CLASSES_TYPES[config_args["model_type"]],
            analysis=False)

        # ...log the running loss
        tb_writer.add_scalar('Accuracy/{}'.format(mode), acc, global_step)
        tb_writer.add_scalar('F1/{}'.format(mode), f1, global_step)

        tb_writer.add_scalar(f'Loss/{mode}', eval_losses_avg[0], global_step)
        if MODEL_CLASSES_TYPES[config_args["model_type"]]:
            tb_writer.add_scalar(f'CELossSent/{mode}', eval_losses_avg[1], global_step)
            tb_writer.add_scalar(f'CELossDic/{mode}', eval_losses_avg[2], global_step)
            tb_writer.add_scalar(f'TripletLoss/{mode}', eval_losses_avg[3], global_step)
        for param_group_index, optimizer_param_group in enumerate(optimizer.param_groups):
            tb_writer.add_scalar(f'LearningRate/{param_group_index}', optimizer_param_group['lr'], global_step)

        # Save model checkpoint
        if acc > best_acc:
            logger.info("Saving model checkpoint to %s", config_args['output_dir'])
            torch.save(model, os.path.join(config_args['output_dir'], "model.pt"))
            tokenizer.save_pretrained(config_args['output_dir'])
            config_args['best_global_step'] = global_step
            config_args['best_accuracy'] = acc
            torch.save(config_args, os.path.join(config_args['output_dir'], 'training_args.bin'))
        return eval_losses_avg[0], acc, f1

    print("before pre_Training_setup")
    train_dataset, train_dataloader, optimizer, scheduler, t_total = pre_training_setup()
    model = model.to(device)
    model.train()
    log_before_training()
    previous_best_acc = 0
    macro_f1_associated_with_previous_best_acc = 0
    early_stop = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(config_args['num_train_epochs']), desc="Epoch")
    for _ in train_iterator:  # Iterating over pre set number of epochs
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = [item.to(device) if type(item) == torch.Tensor else item for item in batch]
            batch.append(global_step)
            outputs = model(batch)

            loss = outputs[0]
            if MODEL_CLASSES_TYPES[config_args["model_type"]]:
                ce_loss_sent = outputs[3]
                ce_loss_dic = outputs[4]
                triplet_loss = outputs[5]

            if config_args['gradient_accumulation_steps'] > 1:
                loss = loss / config_args['gradient_accumulation_steps']
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_args['max_grad_norm'])
            tr_loss += loss.item()
            if (step + 1) % config_args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if config_args['logging_steps'] > 0 and global_step % config_args['logging_steps'] == 0:
                    # Get the train accuracy and write it to eval file
                    tb_writer.add_scalar(f'Loss/train', loss, global_step)
                    if MODEL_CLASSES_TYPES[config_args["model_type"]]:
                        tb_writer.add_scalar("CELossSent/train", ce_loss_sent, global_step)
                        tb_writer.add_scalar("CELossDic/train", ce_loss_dic, global_step)
                        tb_writer.add_scalar("TripletLoss/train", triplet_loss, global_step)
                    batch_mask = batch[8].bool()
                    tr_out_label_ids = batch[6][batch_mask].detach().cpu().numpy()
                    tr_logits = outputs[1]
                    tr_preds = tr_logits.detach().cpu().numpy()
                    tr_mcc, tr_confusion_matrix, tr_class_report, tr_acc, tr_f1, tr_false_pos, tr_false_neg, tr_top_2_accuracy, tr_top_3_accuracy = compute_metrics(
                        tr_preds, tr_out_label_ids,
                        config_args["num_labels"])
                    tr_v_measure = -1
                    write_results_to_eval_file(tr_v_measure, tr_mcc, tr_confusion_matrix, tr_class_report, tr_acc,
                                               tr_f1, global_step, mode='train')

                    # Update the previous best for validation
                    torch.cuda.empty_cache()
                    eval_loss, acc, f1 = log_metrics(model, tokenizer, fold, dataset_splits, previous_best_acc,
                                                     global_step, 'test')

                    if previous_best_acc < acc:
                        previous_best_acc = acc
                        macro_f1_associated_with_previous_best_acc = f1
                        early_stop = 0
                    else:
                        early_stop += 1
                        if early_stop == config_args['early_stop']:
                            return eval_loss, previous_best_acc, macro_f1_associated_with_previous_best_acc
                    model.train()

    last_loss, last_acc, last_f1 = log_metrics(model, tokenizer, fold, dataset_splits, previous_best_acc, global_step,
                                               'test')
    if previous_best_acc < last_acc:
        previous_best_acc = last_acc
        macro_f1_associated_with_previous_best_acc = last_f1
    return last_loss, previous_best_acc, macro_f1_associated_with_previous_best_acc


def validate_test(model, tokenizer, fold, dataset_splits, mode, false_pos=None, false_neg=None, dictionary=False,
                  num_instances=300, analysis=False):

    eval_dataset = load_sliced_dataset(dataset_splits, fold, mode)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config_args['eval_batch_size'], collate_fn=collate,
                                 num_workers=config_args['num_workers'])

    eval_losses = list()
    eval_ce_losses_sent = list()
    eval_ce_losses_dic = list()
    eval_triplet_losses = list()
    nb_eval_steps = 0
    preds = None

    out_label_ids = None
    model.eval()

    for i, batch in enumerate(tqdm(eval_dataloader)):
        # torch.cuda.empty_cache()
        batch = [item.to(device) if type(item) == torch.Tensor else item for item in batch]
        batch.append(0)  # this is for pseudo supervised learning global step
        with torch.no_grad():
            torch.cuda.empty_cache()
            if dictionary:
                outputs, dictionary_gumbel_weights_list, top_k_tags_list, top_k_metrics_and_indices_tags, top_k_tags_list_metrics = model.evaluate(
                    tokenizer, batch)
            else:
                outputs = model(batch, isTrain=False)

            tmp_eval_loss, log_probs = outputs[:2]
            eval_losses.append(tmp_eval_loss.cpu().item())

            if dictionary:

                tmp_ce_loss_sent = outputs[3]
                tmp_ce_loss_dic = outputs[4]
                tmp_triplet_loss = outputs[5]
                eval_ce_losses_sent.append(tmp_ce_loss_sent.cpu().item())
                eval_ce_losses_dic.append(tmp_ce_loss_dic.cpu().item())
                eval_triplet_losses.append(tmp_triplet_loss.cpu().item())

        nb_eval_steps += 1
        preds = np.append(preds, log_probs.cpu().numpy(),
                          axis=0) if preds is not None else log_probs.cpu().numpy()
        out_label_ids = np.append(out_label_ids, batch[6].cpu().numpy(),
                                  axis=0) if out_label_ids is not None else batch[6].cpu().numpy()

    if dictionary:
        all_losses = [eval_losses, eval_ce_losses_sent, eval_ce_losses_dic, eval_triplet_losses]
        all_losses_avg = [round(sum(i) / len(i), 3) for i in all_losses]
    else:
        all_losses = [eval_losses]
        all_losses_avg = [round(sum(i) / len(i), 3) for i in all_losses]
    mcc, confusion_matrix, class_report, acc, f1, false_pos, false_neg, top_2_accuracy, top_3_accuracy = compute_metrics(
        preds, out_label_ids, config_args["num_labels"], false_pos=false_pos, false_neg=false_neg)
    return all_losses_avg, mcc, confusion_matrix, class_report, acc, f1


def build_args_parser():
    main_arg_parser = argparse.ArgumentParser(
        description="Parser for FRISS")
    main_arg_parser.add_argument("--num_workers", type=str, default='0',
                                 help="num of workers for dataloader")
    main_arg_parser.add_argument("--issue", type=str, required=False,
                                 help="The issue: either samesex or immigration or others")
    main_arg_parser.add_argument("--model_name", type=str, required=False,
                                 help="specify the models: choose your choice from website :D ")
    main_arg_parser.add_argument("--model_type", type=str, required=False,
                                 help="specify the models: choices are 'bert', 'roberta' and 'xlnet'")
    main_arg_parser.add_argument("--fold", type=str, required=True, help="specify the fold index (0-9)")
    main_arg_parser.add_argument("--run_name", type=str, required=True, help="specify an unique identifier for the run")
    main_arg_parser.add_argument("--num_runs", type=int, default=3, help="specify how many times to run for each fold")
    main_arg_parser.add_argument('--mlm_identifier', type=str, default="11-19-2020-00-54-13",
                                 help="Specify the identifier to mlm")

    main_arg_parser.add_argument('--evaluate_temperature', type=float, default=-1,
                                 help="the temperature used for evalution")
    main_arg_parser.add_argument('--train_batch_size_labeled', type=int, default=8, help="train_batch_size_labeled")
    main_arg_parser.add_argument('--train_batch_size_unlabeled', type=int, default=0, help="train_batch_size_unlabeled")

    main_arg_parser.add_argument('--seed', type=int, default=68, help="random seed")
    main_arg_parser.add_argument("--num_unlabelled", type=str, required=True, help="spcify the num_unlabelld (0-41966)")
    main_arg_parser.add_argument("--tags", nargs="+", default=["B-V"])
    main_arg_parser.add_argument("--start_from", type=str, required=True,
                                 help="specify from which pre-trained model to train")

    main_arg_parser.add_argument('--classifier_lr', type=float, default=5e-4, help="the classifier learning rate")
    main_arg_parser.add_argument('--dictionary_lr', type=float, default=5e-4, help="the dictionary learning rate")
    main_arg_parser.add_argument('--alpha', type=float, default=1.0, help="manual tuning param alpha")
    main_arg_parser.add_argument('--beta', type=float, default=1.0, help="manual tuning param beta")
    main_arg_parser.add_argument('--classifier_dropout_prob', type=float, default=0.2,
                                 help="the classifier dropout rate")
    main_arg_parser.add_argument('--dictionary_learning_dropout_prob', type=float, default=0.2,
                                 help="the dictionary learning dropout rate")
    main_arg_parser.add_argument('--weight_decay', type=float, default=5e-7, help="optimizer weight decay")

    main_arg_parser.add_argument('--margin', type=float, default=3.0, help="the margin for triplet loss")
    main_arg_parser.add_argument('--num_k', type=int, default=1, help="gumbel softmax top-k parameter")
    main_arg_parser.add_argument('--schedule_r', type=float, default=-0.0005,
                                 help="the dictionary learning schedule hyperparameter which determines the temperature t = e^(-r * global_step), the default value results in the temperature reaching 0.606 at 1000 steps, and 0.47 at 1500 ")
    main_arg_parser.add_argument('--temperature_limit', type=float, default=0.5,
                                 help="the minimum temperature for dictionary learning gumbel softmax")

    shell_args = main_arg_parser.parse_args()

    s_dic = dict()
    for k in vars(shell_args):
        if getattr(shell_args, k):
            s_dic[k] = getattr(shell_args, k)
        if k == "train_batch_size_labeled" or k == "train_batch_size_unlabeled":
            s_dic[k] = getattr(shell_args, k)

    config_args.update(s_dic)

    config_args['fold'] = config_args['fold'].replace("'", "")
    config_args['num_unlabelled'] = int(config_args['num_unlabelled'])
    config_args['num_workers'] = int(config_args['num_workers'])
    assert config_args['fold'] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    assert config_args["model_type"] in MODEL_CLASSES
    config_args['num_labels'] = issues_config[config_args['issue']]

    config_args["output_dir"] = os.path.join(config_args["output_dir"],
                                             f"{config_args['run_name']}_{config_args['model_type']}",
                                             config_args['fold'])
    if not os.path.exists(config_args['output_dir']):
        os.makedirs(config_args['output_dir'])

def load_pretrained_model(start_from="raw"):
    assert start_from in ["mlm", "raw"]
    if start_from == "mlm":
        config_class, model_class, tokenizer_class = MODEL_CLASSES[config_args['model_type']]
        config = config_class.from_pretrained(config_args['model_name'], num_labels=config_args['num_labels'],
                                              finetuning_task=config_args['task_name'])
        checkpoint = os.path.join(MFC_mlm_result_directory, config_args["mlm_identifier"])
        tokenizer = tokenizer_class.from_pretrained(config_args['model_name'])
        config.margin = config_args['margin']
        config.schedule_r = config_args['schedule_r']
        config.temperature_limit = config_args['temperature_limit']
        config.evaluate_temperature = config_args['evaluate_temperature']
        config.do_freeze = config_args['do_freeze']
        config.unfrozen_steps = config_args['unfrozen_steps']
        config.frozen_steps = config_args['frozen_steps']
        config.num_k = config_args['num_k']
        config.tags = config_args['tags']
        config.alpha = config_args['alpha']
        config.beta = config_args['beta']
        model = model_class.from_pretrained(checkpoint, config=config)

    else:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[config_args['model_type']]
        config = config_class.from_pretrained(config_args['model_name'], num_labels=config_args['num_labels'],
                                              finetuning_task=config_args['task_name'])
        tokenizer = tokenizer_class.from_pretrained(config_args['model_name'])
        config.margin = config_args['margin']
        config.schedule_r = config_args['schedule_r']
        config.temperature_limit = config_args['temperature_limit']
        config.evaluate_temperature = config_args['evaluate_temperature']
        config.do_freeze = config_args['do_freeze']
        config.unfrozen_steps = config_args['unfrozen_steps']
        config.frozen_steps = config_args['frozen_steps']
        config.num_k = config_args['num_k']
        config.tags = config_args['tags']
        config.alpha = config_args['alpha']
        config.beta = config_args['beta']
        model = model_class.from_pretrained(config_args['model_name'], config=config)

    return model, tokenizer


def load_model_from_path(fold):
    checkpoint_path = os.path.join(config_args['test_output_dir'], str(fold))
    config_class, model_class, tokenizer_class = MODEL_CLASSES[config_args['test_model_type']]
    model = torch.load(os.path.join(checkpoint_path, "model.pt"))
    tokenizer = tokenizer_class.from_pretrained(config_args['model_name'])
    model.to(device)
    test_config_args = torch.load(os.path.join(checkpoint_path, 'training_args.bin'))
    print(test_config_args)
    return model, tokenizer, test_config_args


def main():
    build_args_parser()
    set_seed(config_args['seed'])
    dataset_splits = load_data(config_args['num_folds'])

    # Starts training
    config_args['output_eval_file'] = os.path.join(config_args['output_dir'], "results.txt")
    final_result_file = os.path.join(config_args['output_dir'], "final_result.txt")

    # Delete the output_eval_file if exists
    if os.path.exists(config_args['output_eval_file']):
        os.remove(config_args['output_eval_file'])
    if os.path.exists(final_result_file):
        os.remove(final_result_file)

    accuracies = list()
    f1s = list()

    serialize_to_file(os.path.join(config_args["output_dir"], "run_config.json"), config_args)


    for run_index in range(config_args['num_runs']):

        tb_writer = SummaryWriter(os.path.join(config_args['output_dir'], "runs/{}".format(run_index)))
        model, tokenizer = load_pretrained_model(start_from=config_args['start_from'])
        test_loss, acc, f1 = train(model, tokenizer, config_args['fold'], dataset_splits, tb_writer)
        accuracies.append(acc)
        f1s.append(f1)
        with open(final_result_file, "w") as f:
            f.write("accuracies and f1s after {} runs for fold {}:{}:{} ".format(run_index + 1,
                                                                                 config_args['fold'],
                                                                                 accuracies, f1s))
    logger.info(
        "max accuracy and f1 over {} runs for fold {}:{}:{} ".format(config_args['num_runs'], config_args['fold'],
                                                                     max(accuracies), max(f1s)))



if __name__ == '__main__':
    main()

