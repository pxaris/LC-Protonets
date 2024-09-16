import random
import torch
import itertools
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from mlfsl.datasets.utils import split_spectrogram


class MLFSLSampler(Sampler):
    def __init__(
            self,
            dataset,
            n_way,
            k_shot,
            n_query,
            n_task,
            is_test=False,
            input_length=None,
    ):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_task = n_task
        self.support_set_length = None
        self.is_test = is_test
        self.input_length = input_length

        self.labels = torch.tensor(())
        self._labels = []
        self.indices = []
        self.selected_labels_idx = []
        self.label_combinations_indices = {}

        for idx, single_sample_batch in enumerate(DataLoader(dataset=self.dataset, batch_size=1)):

            _, item_labels = single_sample_batch
            self.labels = torch.cat([self.labels, item_labels])

            item_labels_indices = torch.nonzero(item_labels[0]).T.tolist()[0]
            self._labels.append(item_labels_indices)
            self.indices.append(idx)
            # for each item, get all the combinations of its labels
            # and add them to the label_combinations_features dict
            for r in range(1, len(item_labels_indices)+1):
                labels_combinations = list(
                    itertools.combinations(item_labels_indices, r))
                for combination in labels_combinations:
                    if combination in self.label_combinations_indices:
                        self.label_combinations_indices[combination].append(
                            idx)
                    else:
                        self.label_combinations_indices[combination] = [idx]

        self.per_label_indices = {
            k: v for k, v in self.label_combinations_indices.items() if len(k) == 1}

    def __len__(self):
        return self.n_task
    
    def sample_items(self, selected_labels, n_items, exclude_set):
        # sample set
        sample_set = set()
        # items to sample from
        pool_items_indices = set(self.indices) - exclude_set
        
        # keep running counts to avoid sampling more than necessary items
        label_counts = {label_name: 0 for label_name in selected_labels}        
        for label in selected_labels:
            n_items_to_sample = n_items - label_counts[label]
            if n_items_to_sample > 0:
                selected_samples = random.sample(
                    set(self.per_label_indices[label]) & pool_items_indices, n_items_to_sample)
                sample_set.update(selected_samples)
                # update label counts for all item labels
                for idx in selected_samples:
                    for selected_item_label in self._labels[idx]:
                        if tuple([selected_item_label]) in selected_labels:
                            label_counts[tuple([selected_item_label])] += 1
        return sample_set


    def __iter__(self):
        '''
        Sample n_way labels uniformly at random,
        and then sample k_shot + n_query items for each label, also uniformly at random.
        Yields:
            a list of items ids while storing the support set length to self. 
        '''
        for _ in range(self.n_task):
            batch_ids = []
            selected_labels = random.sample(
                self.per_label_indices.keys(), self.n_way)
            self.selected_labels_idx = torch.tensor([label[0] for label in selected_labels])

            # support set
            support_set = self.sample_items(selected_labels, n_items=self.k_shot, exclude_set=set())
            batch_ids.extend(list(support_set))
            self.support_set_length = len(support_set)

            # query set
            if self.n_query:
                query_set = self.sample_items(selected_labels, n_items=self.n_query, exclude_set=support_set)
            else:
                # if n_query is None, return all query items
                # to be used in evaluation phases
                query_set = set(self.indices) - support_set
            batch_ids.extend(list(query_set))
            yield batch_ids

    def episodic_collate_fn(self, input_data):
        '''
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            - input_data: each element is a tuple containing:
                - a feature vector as a torch Tensor of shape (time_length, mel_bins)
                - the labels of this item as a multi-hot vector
        Returns:
            - support items of shape (support_set_length, time_length, mel_bins),
            - their labels of shape (support_set_length, n_way),
            - query items of shape (query_set_length, time_length, mel_bins)
            - their labels of shape (query_set_length, n_way),
            - the selected labels indices for the current task
        '''
        if self.is_test:
            # in case of test set, the full length of the spectrogram is retuned by the dataset
            all_items = []
            for x in input_data:
                splitted_spectrogram = split_spectrogram(x[0], self.input_length)
                all_items.append(splitted_spectrogram)
            support_items = all_items[:self.support_set_length]
            query_items = all_items[self.support_set_length:]
        else:        
            all_items = torch.cat([torch.tensor(x[0]).unsqueeze(0)
                                for x in input_data])
            # support_set_length is stored to the object to be used here
            support_items = all_items[:self.support_set_length, :, :]
            query_items = all_items[self.support_set_length:, :, :]
        
        all_labels = torch.cat([torch.tensor(x[1]).unsqueeze(0)
                               for x in input_data])
        # filter all labels to keep only the selected ones in the current task
        all_labels = all_labels[:, self.selected_labels_idx]
        support_labels = all_labels[:self.support_set_length, :]
        query_labels = all_labels[self.support_set_length:, :]

        return (
            support_items,
            support_labels,
            query_items,
            query_labels,
            self.selected_labels_idx,
        )
