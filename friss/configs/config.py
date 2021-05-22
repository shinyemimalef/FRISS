import os

issue = "immigration_all"
user = ""
home_dir = f"/home/{user}"

datasets_directory = os.path.join(home_dir, 'datasets')
MFC_dataset_directory = os.path.join(datasets_directory, "MfcDataset")
MFC_mlm_dataset_directory = os.path.join(datasets_directory, 'MfcMlmDataset')
MFC_graph_dataset_directory = os.path.join(datasets_directory, 'MfcGraphBertDataset/{}'.format(issue))
MFC_graph_dataset_raw_directory = os.path.join(MFC_graph_dataset_directory, 'raw')
MFC_graph_dataset_processed_directory = os.path.join(MFC_graph_dataset_directory, 'processed')

results_dir = os.path.join(home_dir, "results")
MFC_mlm_result_directory = os.path.join(results_dir, "MfcMlmDataset")
MFC_graph_result_directory = os.path.join(results_dir, 'MfcGraphBertDataset/{}'.format(issue))

class_labels_distribution = [414, 210, 76, 155, 957,
                             473, 803, 286, 239, 410,
                             556, 243, 969, 132, 10]

class_labels_dist = [6.98, 3.54, 1.28, 2.61, 16.13, 7.97, 13.53, 4.82, 4.03, 6.91, 9.37, 4.1, 16.33, 2.22, 0.17]

default_class_labels_distribution = [1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1]

config_args = {
    'issue': issue,
    'data_dir': MFC_graph_dataset_directory,
    'base_type': 'roberta',
    'model_type': 'roberta',
    'model_name': 'roberta-base',
    'start_from': 'mlm',  # options: 'raw', 'mlm'
    'mlm_identifier': 'mlm_sents',
    'task_name': 'MFC_multiclass_classification',
    'output_dir': MFC_graph_result_directory,
    'sampler': True,
    'do_train': True,
    'do_eval': True,
    'eval_dictionary': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 64,
    'output_mode': 'classification',
    'train_batch_size': 8,
    'eval_batch_size': 32,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 15,
    'num_folds': 10,
    'weight_decay': 5e-7,
    'dictionary_lr': 5e-4,
    'classifier_lr': 5e-4,

    'classifier_dropout_prob': 0.2,
    'dictionary_learning_dropout_prob': 0.2,
    'lm_learning_rate': 2e-5,  # pretrained_model learning rate

    'adam_epsilon': 1e-8,
    'warmup_steps': 10,
    'max_grad_norm': 1.0,
    'early_stop': 10,
    'logging_steps': 100,
    'evaluate_during_training': True,
    'eval_all_checkpoints': True,
    'overwrite_output_dir': True,
}

issues_config = {
    "immigration": 15,
    "immigration_all": 15
}
