import torch
import random
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from mlfsl.datasets.utils import split_spectrogram
from mlfsl.utils import chunks


class OnevsRestSampler(Sampler):
    def __init__(
            self,
            dataset,
            n_way,
            k_shot,
            tags,
            n_task=None,
            is_test=False,
            input_length=None,
            shuffle=True
    ):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.is_test = is_test
        self.input_length = input_length
        self.shuffle = shuffle
        self.unique_support_items_inference = None
        self.query_set_inference = set()
        self.batch_items_labels = None

        dataset_labels = [str(label) for label in dataset.label_transformer.classes_.tolist()]
        self.selected_labels_idx = [idx for idx, label in enumerate(dataset_labels) if label in tags]
        
        self.indices = []
        self.per_label_indices = {}
        self.items_labels = {}

        for idx, single_sample_batch in enumerate(DataLoader(dataset=self.dataset, batch_size=1)):
            _, item_labels = single_sample_batch

            item_labels_indices = torch.nonzero(item_labels[0]).T.tolist()[0]
            self.indices.append(idx)
            self.items_labels[idx] = item_labels_indices
            for label_idx in item_labels_indices:
                if label_idx in self.per_label_indices:
                    self.per_label_indices[label_idx].append(idx)
                else:
                    self.per_label_indices[label_idx] = [idx]
        
        self.n_task = n_task or len(self.indices)

    def __len__(self):
        return self.n_task
    
    def __iter__(self):

        if self.is_test:
            for _ in range(self.n_task):
                self.unique_support_items_inference = set()
                # return one support set followed by all query items
                # keep running counts to avoid sampling more than necessary items
                label_supports = {label: [] for label in self.selected_labels_idx}        
                for label in self.selected_labels_idx:
                    n_items_to_sample = self.k_shot - len(label_supports[label])
                    if n_items_to_sample > 0:
                        selected_samples = random.sample(
                                set(self.per_label_indices[label]), n_items_to_sample)
                        # update label supports for all item labels
                        for idx in selected_samples:
                            for selected_item_label in self.items_labels[idx]:
                                if selected_item_label in self.selected_labels_idx:
                                    label_supports[selected_item_label].append(idx)
                
                batch_ids, batch_items_labels = [], []
                for label, support_indices in label_supports.items():
                    if len(support_indices) > self.k_shot:
                        support_indices = random.sample(support_indices, self.k_shot)
                    batch_ids.extend(support_indices)
                    batch_items_labels.extend([label]*len(support_indices))
                    self.unique_support_items_inference.update(support_indices)

                self.query_set_inference = set(self.indices) - self.unique_support_items_inference
                for item_index in self.query_set_inference:
                    batch_ids.append(item_index)
                    batch_items_labels.append(tuple(self.items_labels[item_index]))
                
                self.batch_items_labels = batch_items_labels
                yield batch_ids
        
        else:
            indices = np.random.permutation(self.indices)

            tasks_counter = 0
            for q_item_index in indices:
                tasks_counter += 1
                if tasks_counter > self.n_task:
                    break
                active_labels = list(set(self.items_labels[q_item_index]) & set(self.selected_labels_idx))
                non_active_labels = list(set(self.selected_labels_idx) - set(active_labels))

                active_label_subsets = []
                for label in active_labels:
                    try:
                        label_set = np.random.choice(non_active_labels, size=(self.n_way - 1), replace=False).tolist()
                    except ValueError:
                        # prevent the `ValueError: Cannot take a larger sample than population` in case that `non_active_labels` are less than `n_way-1` 
                        break
                    label_set.append(label)  # [rand_cls_0, rand_cls_1, rand_cls_2, rand_cls_3, query_label]
                    active_label_subsets.append(np.random.permutation(label_set).tolist())  # shuffle the label subset
                
                if not active_label_subsets:
                    continue
                
                # Construct subset where each sample belongs to one and only one class in the label subset
                subsets = []
                for label_set in active_label_subsets:
                    label_set_items_dict = {}
                    for item_idx in indices:
                        item_labels = self.items_labels[item_idx]
                        # a sample is selected if one or more of its labels belong to the label subset
                        if len(set(item_labels) & set(label_set)) >= 1:
                            label_set_items_dict[item_idx] = item_labels
                    subsets.append(label_set_items_dict)
                
                # Randomly select the support samples for each query item 
                batch_ids, batch_items_labels = [], []
                for idx, label_set in enumerate(active_label_subsets):
                    cluster = []
                    for label in label_set:
                        if label in active_labels:
                            current_active_label = label
                        support_samples = []
                        for item_idx, item_labels in subsets[idx].items():
                            if label in item_labels and q_item_index != item_idx:
                                support_samples.append(item_idx)
                        supports_for_label = np.random.choice(support_samples, size=self.k_shot, replace=False).tolist()
                        cluster.extend(supports_for_label)
                        batch_items_labels.extend([label]*len(supports_for_label))
                    cluster.append(q_item_index)  # add the only one query to the end and return the cluster to batch
                    batch_ids.extend(cluster)
                    batch_items_labels.append(current_active_label)  # add the current active label of the query item
                
                self.batch_items_labels = batch_items_labels
                yield batch_ids

    def episodic_collate_fn(self, input_data):
        '''
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Each cluster contains n_way * k_shot items + the one query item at the end
        So, the lists:
            - clusters_items and clusters_labels are returned with size equal to the number of labels of the query item
            - clusters_active_labels list is returned that contain the n_way labels selected at each cluster + the specific one used in this cluster from the query item 
        '''
        if self.is_test:
            # in case of test set, the full length of the spectrogram is retuned by the dataset
            all_items = []
            for x in input_data:
                splitted_spectrogram = split_spectrogram(x[0], self.input_length)
                all_items.append(splitted_spectrogram)            
            
            all_labels = torch.zeros((len(input_data), self.n_way))
            for i, label in enumerate(self.batch_items_labels):
                try:
                    # support items in input data
                    all_labels[i][self.selected_labels_idx.index(label)] = 1
                except:
                    # special case of query item on inference, label is a tuple with all item labels
                    for l in label:
                        all_labels[i][self.selected_labels_idx.index(l)] = 1
            
            clusters_items_list = [all_items]
            clusters_labels_list = [all_labels]
            clusters_active_labels_list = [self.selected_labels_idx]

        else:        
            all_items = torch.cat([torch.tensor(x[0]).unsqueeze(0)
                                for x in input_data])
       
            clusters_length = self.n_way * self.k_shot + 1
            items_labels_clusters = [chunk for chunk in chunks(self.batch_items_labels, clusters_length)]

            clusters_active_labels_list = [] 
            for cluster in items_labels_clusters:
                cluster_active_labels = [cluster[0]]
                for label in cluster[1:]:
                    if label == cluster_active_labels[-1]:
                        continue
                    cluster_active_labels.append(label)
                clusters_active_labels_list.append(cluster_active_labels)
            
            all_labels = torch.zeros((len(input_data), self.n_way))
            for i, label in enumerate(self.batch_items_labels):
                cluster_idx = i // clusters_length
                all_labels[i][clusters_active_labels_list[cluster_idx].index(label)] = 1

            clusters_items_list = torch.split(all_items, clusters_length)
            clusters_labels_list = torch.split(all_labels, clusters_length)

        return (
            clusters_items_list,
            clusters_labels_list,
            clusters_active_labels_list
        )
