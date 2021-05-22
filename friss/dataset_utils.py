import math
import random
import torch
from torch.utils.data import Subset
from torch.utils.data.sampler import Sampler

from friss.MfcSentsDataset import MfcSentsDataset
from friss.configs.config import config_args

mfc_sents_dataset = MfcSentsDataset(config_args['data_dir'], "immigration_all")

class MySampler(Sampler):

    def __init__(self, fold_indices, batch_size_labeled=8, batch_size_unlabeled=8, shuffle=True):
        self.fold_indices = fold_indices
        self.batch_size_labeled = batch_size_labeled
        self.batch_size_unlabeled = batch_size_unlabeled
        self.num_labelled = mfc_sents_dataset.num_labelled
        self.labeled_indices = [i for i in self.fold_indices if i < self.num_labelled]
        self.unlabeled_indices = [i for i in self.fold_indices if i >= self.num_labelled]
        self.relative_labeled_indices = list(range(0, len(self.labeled_indices)))
        self.relative_unlabeled_indices = list(range(len(self.labeled_indices), len(self.fold_indices)))
        self.num_labelled_subset = len(self.relative_labeled_indices)
        self.num_unlabelled_subset = len(self.relative_unlabeled_indices)

        assert batch_size_labeled > 0
        self.num_batches_labelled = math.ceil(len(self.labeled_indices) / batch_size_labeled)
        if batch_size_unlabeled > 0 and len(self.unlabeled_indices) > 0:

            self.num_batches_unlabelled = math.ceil(len(self.unlabeled_indices) / batch_size_unlabeled)
            self.num_batches = max(self.num_batches_labelled, self.num_batches_unlabelled)

        else:
            self.num_batches = self.num_batches_labelled
            self.num_batches_unlabelled = 0
        self.shuffle = shuffle

    def __iter__(self):
        # shuffle it every epoch
        if self.shuffle:
            random.shuffle(self.relative_labeled_indices)
            random.shuffle(self.relative_unlabeled_indices)

        n = self.num_batches
        i = 0  # tracking the batch index of the labelled
        j = 0  # tracking the batch index of the unlabelled
        while n > 0:

            if i == self.num_batches_labelled:
                i = 0

            batch_labeled_indices = self.relative_labeled_indices[
                                    i * self.batch_size_labeled: min((i + 1) * self.batch_size_labeled,
                                                                     self.num_labelled_subset)]

            if j == self.num_batches_unlabelled:
                j = 0
            batch_unlabeled_indices = self.relative_unlabeled_indices[
                                      j * self.batch_size_unlabeled: min((j + 1) * self.batch_size_unlabeled,
                                                                         self.num_unlabelled_subset)]

            batch_indices = batch_labeled_indices + batch_unlabeled_indices
            yield batch_indices  # return the relative indices of a mini batch
            n -= 1
            i += 1
            j += 1

    def __len__(self):
        return self.num_batches


def load_sliced_dataset(dataset_splits, fold, mode, batch_size_labeled=None, batch_size_unlabeled=None,
                        num_labelled=5933, num_unlabelled=None):
    labeled_indices = dataset_splits[fold]["{}_indices".format(mode)].copy()
    if mode == "train":
        unlabeled_indices = list(range(num_labelled, len(mfc_sents_dataset)))
        random.shuffle(unlabeled_indices)

        if num_unlabelled or num_unlabelled == 0:
            assert num_unlabelled <= len(unlabeled_indices)
            unlabeled_indices = unlabeled_indices[:num_unlabelled]
        labeled_indices.extend(unlabeled_indices)

    sliced_mfc_sents_dataset = Subset(mfc_sents_dataset, torch.LongTensor(labeled_indices))

    if mode == "train":
        mySampler = MySampler(labeled_indices, batch_size_labeled=batch_size_labeled,
                              batch_size_unlabeled=batch_size_unlabeled)
        return sliced_mfc_sents_dataset, mySampler
    else:
        return sliced_mfc_sents_dataset
