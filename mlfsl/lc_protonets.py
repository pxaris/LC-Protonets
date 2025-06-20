import torch
import itertools
from tqdm import tqdm
from mlfsl.protonets import Protonets


class LCProtonets(Protonets):
    def __init__(
            self,
            backbone,
            distance='cos',  # "cos" for cosine distance and "l2" for euclidean distance
            # whether to apply min-max row-wise normalization to distances
            normalize_distances=False,
            is_optimized=True,  # Flag to enable optimization
    ):
        super().__init__(backbone, distance, normalize_distances)
        self.support_label_combinations = []
        self.is_optimized = is_optimized
        # For optimized mode
        self.unique_prototypes = None
        self.prototype_to_combinations = None
        self.prototype_indices = None

    def process_support_set(self, support_items, support_labels, device='cpu'):
        self.support_labels = support_labels.to(device)
        self.support_embeddings = self.compute_embeddings(
            support_items, device=device)
        self.label_combinations_embeddings = self.compute_label_combinations_embeddings(
            self.support_embeddings, self.support_labels, device)
        
        if self.is_optimized:
            self.prototypes, self.support_label_combinations, self.unique_prototypes, self.prototype_to_combinations, self.prototype_indices = self.compute_prototypes_optimized(
                self.label_combinations_embeddings, device)
        else:
            self.prototypes, self.support_label_combinations = self.compute_prototypes(
                self.label_combinations_embeddings, device)

    @staticmethod
    def compute_label_combinations_embeddings(support_embeddings, support_labels, device):
        label_combinations_embeddings = {}

        for embedding, item_labels in zip(support_embeddings, support_labels):
            item_labels_indices = torch.nonzero(item_labels).T.tolist()[0]
            # for each item, get all the combinations of its labels
            # and add them to the label_combinations_embeddings dict
            for r in range(1, len(item_labels_indices)+1):
                labels_combinations = list(
                    itertools.combinations(item_labels_indices, r))
                for combination in labels_combinations:
                    if combination not in label_combinations_embeddings:
                        label_combinations_embeddings[combination] = torch.tensor(
                            ()).to(device)
                    label_combinations_embeddings[combination] = torch.cat(
                        [label_combinations_embeddings[combination], embedding.unsqueeze(0)])

        return label_combinations_embeddings

    @staticmethod
    def compute_prototypes(label_combinations_embeddings, device):
        '''
        Compute label-combinations prototypes from label-combinations and corresponding support embeddings
        Args:
            label_combinations_embeddings: dict with label_combinations for keys and the embeddings of the corresponding support items for each label-combination

        Returns:
            the tensor with the LC-prototypes, each one calculated as the average embedding of the instances belonging to the respective label-combination
            the list of the support_label_combinations
        '''
        prototypes = torch.tensor(()).to(device)
        support_label_combinations = []

        for label_combination, embeddings in label_combinations_embeddings.items():
            support_label_combinations.append(label_combination)
            # Prototype is the mean of all embeddings corresponding to a label-combination
            mean_embedding = embeddings.mean(0)
            prototypes = torch.cat([prototypes, mean_embedding.unsqueeze(0)])

        return prototypes, support_label_combinations
    
    @staticmethod
    def compute_prototypes_optimized(label_combinations_embeddings, device):
        '''
        Compute optimized label-combinations prototypes by identifying unique embeddings
        
        Args:
            label_combinations_embeddings: dict with label_combinations for keys and the embeddings
        
        Returns:
            prototypes: all prototypes (for compatibility with original method)
            support_label_combinations: all label combinations
            unique_prototypes: tensor of unique prototype embeddings
            prototype_to_combinations: mapping from unique prototype indices to label combinations
            prototype_indices: mapping from original prototype indices to unique prototype indices
        '''
        # First pass: compute all prototypes as in the original method
        all_proto_dict = {}
        prototype_list = []
        support_label_combinations = []        
        for i, (label_combination, embeddings) in enumerate(tqdm(label_combinations_embeddings.items(), 'Computing prototypes')):
            support_label_combinations.append(label_combination)
            mean_embedding = embeddings.cpu().mean(0)
            proto_key = tuple(mean_embedding.cpu().detach().numpy().tolist())
            
            if proto_key in all_proto_dict:
                all_proto_dict[proto_key].append(len(support_label_combinations) - 1)
            else:
                all_proto_dict[proto_key] = [len(support_label_combinations) - 1]
            
            prototype_list.append(mean_embedding)
            
        prototypes = torch.stack(prototype_list)  # all prototypes

        # Second pass: identify unique prototypes and create mappings
        prototype_to_combinations = {}
        prototype_indices = {}
        unique_prototype_list = []
        
        for i, (_, indices) in enumerate(all_proto_dict.items()):
            # Add unique prototype
            unique_prototype_list.append(prototypes[indices[0]])  # use the embedding of the first prototype from the same ones
            
            # Map unique prototype index to label combinations
            prototype_to_combinations[i] = [support_label_combinations[idx] for idx in indices]
            
            # Map original indices to unique prototype index
            for idx in indices:
                prototype_indices[idx] = i
        
        unique_prototypes = torch.stack(unique_prototype_list).to(device)
        
        return prototypes, support_label_combinations, unique_prototypes, prototype_to_combinations, prototype_indices

    def expand_labels_multihot_vectors(self, items_labels):
        expanded_items_labels = torch.tensor(())
        for i_labels in items_labels:
            # get the indices of item labels (multi-hot tensor)
            item_labels_indices = torch.nonzero(i_labels).T.tolist()[0]
            # get all combinations of item's labels
            item_label_combinations = self.get_all_combinations(
                item_labels_indices)
            # get the indices of item's combinations in all label_combinations list
            # query label combinations that do not exist in support label combination will get a 0
            item_lc_indices = [self.support_label_combinations.index(
                lc) for lc in item_label_combinations if lc in self.support_label_combinations]
            # create a tensor and fill the specified indices with 1
            item_expanded_labels = torch.zeros(
                len(self.support_label_combinations))
            item_expanded_labels[item_lc_indices] = 1
            expanded_items_labels = torch.cat(
                (expanded_items_labels, item_expanded_labels.unsqueeze(0)), dim=0)
        return expanded_items_labels

    @staticmethod
    def get_all_combinations(items_list):
        result = []
        for r in range(1, len(items_list)+1):
            result += list(itertools.combinations(items_list, r))
        return result
        
    def forward(self, query_items):
        '''
        Calculate the distances between query items and prototypes.
        Args:
            query_items: items of the query set of shape (n_query, **items_shape)
        Returns:
            the distances from the prototypes (n_query, n_prototypes)
        '''
        # extract the embeddings of query items
        query_embeddings = self.compute_embeddings(
            query_items, device=self.unique_prototypes.device)
            
        if not self.is_optimized:
            # Standard implementation - compute distances to all prototypes
            if self.distance == 'cos':
                # distances will lie in [-1, 1] and they can be, in turn, fed to a sigmoid
                similarities = self.cosine_similarity(
                    query_embeddings, self.prototypes)
                distances = -similarities
            else:
                distances = self.euclidean_distance(
                    query_embeddings, self.prototypes)
            return distances
        else:
            # Optimized implementation - compute distances only to unique prototypes
            if self.distance == 'cos':
                similarities = self.cosine_similarity(
                    query_embeddings, self.unique_prototypes)
                unique_distances = -similarities
            else:
                unique_distances = self.euclidean_distance(
                    query_embeddings, self.unique_prototypes)
            
            return unique_distances

