import torch
import numpy as np
from torch import nn
from mlfsl.utils import min_max_row_wise


class Protonets(nn.Module):
    def __init__(
            self,
            backbone,
            distance='cos',  # "cos" for cosine distance and "l2" for euclidean distance
            # whether to apply min-max row-wise normalization to distances
            normalize_distances=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.distance = distance
        self.normalize_distances = normalize_distances
        self.prototypes = torch.tensor(())

    def process_support_set(self, support_items, support_labels, device='cpu'):
        self.support_labels = support_labels.to(device)
        self.support_embeddings = self.compute_embeddings(
            support_items, device=device)
        self.per_label_embeddings = self.compute_per_label_embeddings(
            self.support_embeddings, self.support_labels, device)
        self.prototypes = self.compute_prototypes(
            self.per_label_embeddings, device)

    def compute_embeddings(self, items, device='cpu'):
        if torch.is_tensor(items):
            # training phase
            embeddings = self.backbone(items.float().to(device))
        else:
            # in testing phase items is a list of lists that include all input chunks for each piece
            embeddings = torch.tensor(()).to(device)
            for item_chunks in items:
                chunks = torch.cat(
                    [torch.tensor(chunk[np.newaxis, :, :]) for chunk in item_chunks])
                chunks_embeddings = self.backbone(chunks.float().to(device))
                # mean reduction is applied to chunk embeddings
                embeddings = torch.cat([embeddings, torch.mean(
                    chunks_embeddings, axis=0).unsqueeze(0)])

        return embeddings

    @staticmethod
    def compute_per_label_embeddings(support_embeddings, support_labels, device):
        per_label_embeddings = {}
        for embedding, item_labels in zip(support_embeddings, support_labels):
            item_labels_indices = torch.nonzero(item_labels).T.tolist()[0]
            for label_idx in item_labels_indices:
                if label_idx not in per_label_embeddings:
                    per_label_embeddings[label_idx] = torch.tensor(
                        ()).to(device)
                per_label_embeddings[label_idx] = torch.cat(
                    [per_label_embeddings[label_idx], embedding.unsqueeze(0)])

        return per_label_embeddings

    @staticmethod
    def compute_prototypes(per_label_embeddings, device):
        prototypes = torch.tensor(()).to(device)

        for label_idx in range(len(per_label_embeddings.keys())):
            # Prototype is the mean of all embeddings corresponding to a label-combination
            mean_embedding = per_label_embeddings[label_idx].mean(0)
            prototypes = torch.cat([prototypes, mean_embedding.unsqueeze(0)])

        return prototypes

    def euclidean_distance(self, x, y):
        distances = torch.cdist(x, y)
        if self.normalize_distances:
            # distances to lie in [-1, 1] in order to be fed to a sigmoid
            distances = min_max_row_wise(distances, min=-1, max=1)
        return distances

    @staticmethod
    def cosine_similarity(x, y, eps=1e-8):
        # Normalize x and y
        norm_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)
        norm_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

        # Calculate cosine similarity using broadcasting
        cosine_similarity = torch.matmul(norm_x, norm_y.t())

        return cosine_similarity

    def forward(
        self,
        query_items,
    ):
        '''
        Calculate the distances between query items and prototypes.
        Args:
            query_items: items of the query set of shape (n_query, **items_shape)
        Returns:
            the distances from the prototypes (n_query, n_prototypes)
        '''
        # extract the embeddings of query items
        query_embeddings = self.compute_embeddings(
            query_items, device=self.prototypes.device)

        # compute the distance of queries from the prototypes
        if self.distance == 'cos':
            # distances will lie in [-1, 1] and they can be, in turn, fed to a sigmoid
            similarities = self.cosine_similarity(
                query_embeddings, self.prototypes)
            distances = -similarities
        else:
            distances = self.euclidean_distance(
                query_embeddings, self.prototypes)

        return distances
