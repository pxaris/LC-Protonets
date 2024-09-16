import time
import torch
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


def train_one_epoch(model, train_loader, config):
    model.train()
    total_loss = 0
    for support_items, support_labels, query_items, query_labels, _ in tqdm(train_loader):

        # compute prototypes
        model.process_support_set(
            support_items, support_labels, device=config['device'])

        # get distances
        distances = model(query_items.to(config['device']))

        # compute loss
        if config['method'] == 'LCP':
            # expand query labels in LC-Protonets
            expanded_query_labels = model.expand_labels_multihot_vectors(query_labels)
            loss = config['loss_function'](-distances, expanded_query_labels.float().to(config['device']))
        else:
            # baseline - one prototype per label and hence distances aree equal to labels 
            loss = config['loss_function'](-distances, query_labels.float().to(config['device']))

        # prepare
        config['optimizer'].zero_grad()
        # backward
        loss.backward()
        # optimizer step
        config['optimizer'].step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_one_epoch_ovr(model, train_loader, config):
    model.train()
    total_loss = 0
    for clusters_items_list, clusters_labels_list, _ in tqdm(train_loader):

        for cluster_items, cluster_labels in zip(clusters_items_list, clusters_labels_list):
            support_items = cluster_items[:-1]
            support_labels = cluster_labels[:-1]
            query_item = cluster_items[-1:]
            query_active_label = cluster_labels[-1:]

            # compute prototypes
            model.process_support_set(
                support_items, support_labels, device=config['device'])

            # get distances
            distances = model(query_item.to(config['device']))

            # compute loss
            loss = config['loss_function'](-distances, query_active_label.float().to(config['device']))

            # prepare
            config['optimizer'].zero_grad()
            # backward
            loss.backward()
            # optimizer step
            config['optimizer'].step()
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_model(model, train_loader, val_loader, config):
    start_time = time.time()
    kickoff_message = f'Training started for model "{Path(config["save_path"]).parent.name}/{Path(config["save_path"]).stem}"...'
    print(kickoff_message)    
    config['logger'].info(kickoff_message)
    
    # report the number of trainable parameters
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config['logger'].info(f'number of trainable parameters: {n_trainable_params}')
    
    if not config['early_stopping_patience']:
        # no early-stopping; EarlyStopper will just save the best model
        config['early_stopping_patience'] = config['epochs']
    early_stopper = EarlyStopper(
        model, config['save_path'], patience=config['early_stopping_patience'], more_is_better=True)

    for epoch in range(config['epochs']):
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        if config['method'] == 'OvR':
            train_loss = train_one_epoch_ovr(model, train_loader, config)
        else:
            train_loss = train_one_epoch(model, train_loader, config)
        
        print(f"Training loss {train_loss}")
        
        # evaluate on validation set
        if config['method'] == 'OvR':
            validation_f1, _, _, _ = evaluate_ovr(
                model, val_loader, config, is_test=False)
        else:
            validation_f1, _, _, _ = evaluate(
                model, val_loader, config, is_test=False)
        
        print(
            f'Validation set metrics:\n- macro-f1: {validation_f1}')

        if early_stopper.early_stop(validation_f1):
            print(f'Early Stopping was activated.\nTraining has been completed.')
            break

    done_message = f'\nDone. Epoch with the best model: {early_stopper.best_epoch}/{config["epochs"]}'
    print(done_message)
    config['logger'].info(done_message)
    config['logger'].info(f'Execution time: {round(time.time() - start_time)} seconds')


def evaluate(model, dataloader, config, is_test=True):
    start_time = time.time()
    model.eval()
    y = []
    y_ = []
    sigmoid = torch.nn.Sigmoid()
    tasks_macro_f1, tasks_micro_f1 = 0, 0
    info_messages = []
    with torch.no_grad():
        for support_items, support_labels, query_items, query_labels, selected_labels_idx in tqdm(dataloader):
            # compute prototypes
            model.process_support_set(
                support_items, support_labels, device=config['device'])
            
            # get distances
            distances = model(query_items)

            # predictions
            if config['method'] == 'LCP':
                # get predictions with LC-Protonets method
                predictions = get_lc_protonets_predictions(distances, model.support_label_combinations, query_labels.shape[1], tqdm_active=is_test)
            else:
                # baseline method, one prototype per label
                probabilities = sigmoid(-distances)
                probabilities = probabilities.detach().cpu().numpy()
                predictions = np.around(probabilities)

            y_ += [item_prediction.tolist() for item_prediction in predictions]
            y += [item_labels.tolist()
                  for item_labels in query_labels.detach().numpy()]

            tasks_macro_f1 += f1_score(y, y_, average='macro')
            tasks_micro_f1 += f1_score(y, y_, average='micro')
    
    # add info messages
    info_messages += [
        f'\n[INFO]',
        f'- Method: {config["method"]}',
        f'- # Prototypes: {len(model.prototypes)}',
        f'- # Unique items in support set: {len(support_items)}',
        f'- # Unique items in query set: {len(query_items)}',
        f'- Mean groung truth labels per item: {round(np.count_nonzero(np.array(y))/len(y), 2)}',
        f'- Mean predicted labels per item: {round(np.count_nonzero(np.array(y_))/len(y_), 2)}\n',
        f'Execution time: {round(time.time() - start_time)} seconds'
    ]

    n_tasks = len(dataloader)
    if n_tasks == 1:
        # n_tasks is exepected to be 1 during evaluation on the test set
        all_target_names = [str(label) for label in dataloader.dataset.label_transformer.classes_.tolist()]
        target_names = [all_target_names[i] for i in selected_labels_idx]
        clf_report = classification_report(y, y_, target_names=target_names)
    else:
        clf_report = None
    return tasks_macro_f1/n_tasks, tasks_micro_f1/n_tasks, clf_report, info_messages


def evaluate_ovr(model, dataloader, config, is_test=True):
    start_time = time.time()
    model.eval()
    y = []
    y_ = []
    sigmoid = torch.nn.Sigmoid()
    tasks_macro_f1, tasks_micro_f1 = 0, 0
    info_messages = []
    with torch.no_grad():
        for clusters_items_list, clusters_labels_list, clusters_active_labels_list in tqdm(dataloader):
            # a single cluster is expected during inference containing all labels and k items for each one of them
            cluster_items = clusters_items_list[0]
            cluster_labels = clusters_labels_list[0]
            cluster_active_labels = clusters_active_labels_list[0]
            support_length = dataloader.batch_sampler.n_way * dataloader.batch_sampler.k_shot

            support_items = cluster_items[:support_length]
            support_labels = cluster_labels[:support_length]
            query_item = cluster_items[support_length:]
            query_labels = cluster_labels[support_length:]
            
            # compute prototypes
            model.process_support_set(
                support_items, support_labels, device=config['device'])
            
            # get distances
            distances = model(query_item)
            
            # predictions
            probabilities = sigmoid(-distances)            
            probabilities = probabilities.detach().cpu().numpy()
            predictions = np.around(probabilities)

            y_ += [item_prediction.tolist() for item_prediction in predictions]
            y += [item_labels.tolist()
                  for item_labels in query_labels.detach().numpy()]

            tasks_macro_f1 += f1_score(y, y_, average='macro')
            tasks_micro_f1 += f1_score(y, y_, average='micro')

    # add info messages
    info_messages += [
        f'\n[INFO]',
        f'- Method: {config["method"]}',
        f'- # Prototypes: {len(model.prototypes)}',
        f'- # Unique items in support set: {len(dataloader.batch_sampler.unique_support_items_inference)}',
        f'- # Unique items in query set: {len(dataloader.batch_sampler.query_set_inference)}',
        f'- Mean groung truth labels per item: {round(np.count_nonzero(np.array(y))/len(y), 2)}',
        f'- Mean predicted labels per item: {round(np.count_nonzero(np.array(y_))/len(y_), 2)}\n',
        f'Execution time: {round(time.time() - start_time)} seconds'
    ]
    
    n_tasks = len(dataloader)
    if is_test:
        all_target_names = [str(label) for label in dataloader.dataset.label_transformer.classes_.tolist()]
        target_names = [all_target_names[i] for i in cluster_active_labels]
        clf_report = classification_report(y, y_, target_names=target_names)
    else:
        clf_report = None
    return tasks_macro_f1/n_tasks, tasks_micro_f1/n_tasks, clf_report, info_messages


def get_lc_protonets_predictions(items_distances_from_lc_protonets, label_combinations, n_labels, tqdm_active=True):
    predictions = []
    for item_distances_from_protonets in tqdm(items_distances_from_lc_protonets, disable=(not tqdm_active)):
        min_distance = np.inf
        predicted_lc = tuple([])        
        for distance_from_lc_protonet, protonet_lc in zip(item_distances_from_protonets, label_combinations):
            if distance_from_lc_protonet < min_distance:
                predicted_lc = protonet_lc
                min_distance = distance_from_lc_protonet
            elif distance_from_lc_protonet == min_distance:
                # in case of same distances, prefer one with the more labels 
                # this approach supports hierarchically related labels
                if len(protonet_lc) > len(predicted_lc):
                    predicted_lc = protonet_lc
                    min_distance = distance_from_lc_protonet
                else:
                    continue
            else:
                continue
        item_predictions = np.zeros(n_labels)
        item_predictions[list(predicted_lc)] = 1
        predictions.append(item_predictions)
    return predictions


def get_distances_from_singletons(distances, label_combinations, n_labels):
    singletons_indices = sorted([label_combinations.index(tuple([idx])) for idx in range(n_labels)])
    return distances[:, singletons_indices]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def min_max_row_wise(tensor, min=0.0, max=1.0):
    """
    Performs row-wise min-max normalization on a PyTorch tensor.

    Args:
    tensor: The input tensor.
    min: The minimum value for the normalized range (default: 0.0).
    max: The maximum value for the normalized range (default: 1.0).

    Returns:
    A new tensor with the same shape as the input, containing the normalized values.
    """
    eps = 1e-5  # To avoid division by zero

    # Find minimum and maximum values along each row
    min_vals = torch.amin(tensor, dim=1, keepdim=True)
    max_vals = torch.amax(tensor, dim=1, keepdim=True)

    # Normalize each row using min and max values
    normalized = (tensor - min_vals) / (max_vals - min_vals + eps) * (max - min) + min

    return normalized


class EarlyStopper:
    def __init__(self, model, save_path, patience=5, min_delta=0, more_is_better=True, start_epoch=1):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.epochs_counter = 0
        self.best_epoch = None
        self.start_epoch = start_epoch
        self.more_is_better = more_is_better
        if self.more_is_better:
            self.best_validation_metric = 0
        else:
            self.best_validation_metric = np.inf

    def early_stop(self, validation_metric):
        self.epochs_counter += 1
        if self.epochs_counter < self.start_epoch:
            return False
        metric_improved = validation_metric > (
            self.best_validation_metric + self.min_delta) if self.more_is_better else validation_metric < (self.best_validation_metric - self.min_delta)
        if metric_improved:
            self.best_validation_metric = validation_metric
            self.counter = 0
            print('best model!')
            torch.save(self.model.state_dict(), self.save_path)
            self.best_epoch = self.epochs_counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
