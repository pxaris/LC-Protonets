import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from distutils.util import strtobool

# add project root to path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(current_dir)))

from mlfsl.lc_protonets import LCProtonets
from mlfsl.protonets import Protonets
from mlfsl.samplers.mlfsl_sampler import MLFSLSampler
from mlfsl.samplers.ovr_sampler import OnevsRestSampler
from mlfsl.utils import train_model
from mlfsl.models.vgg_ish import VGGish
from mlfsl.datasets.utils import get_train_val_dataset
from config import MODELS_CONFIG, SAVED_MODELS_DIR, FSL_CONFIG, DATASETS, PRETRAINED_DIR, LABELS_DIR, DATA_DIR, LOGS_DIR, SEEDS


def run_training(config):
    # ensure results reproducibility
    random_seed = SEEDS[config['run_idx']]
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # flush the CUDA cache at the beginning
    torch.cuda.empty_cache()

    tags_file = os.path.join(LABELS_DIR, f'{config["dataset"]}.json')
    # dataset tags split
    with open(tags_file) as json_file:
        config['tags'] = json.load(json_file)

    # split training tags for the validation set
    config['tags']['val'] = config['tags']['train'][-config['n_valid_labels']:]
    config['tags']['train'] = config['tags']['train'][:-config['n_valid_labels']]

    # train and validation datasets
    train_dataset, val_dataset = get_train_val_dataset(config)

    tags_info_message = f'Number of tags:\n- training: {len(config["tags"]["train"])}\n- validation: {len(config["tags"]["val"])}'
    print(tags_info_message)

    # backbone model
    if config['backbone_model'] == 'vgg_ish':
        backbone_model = VGGish(n_class=len(train_dataset._tags),
                                is_backbone=True).to(config['device'])
    else:
        raise NotImplementedError(
            f'No model specified for name: {config["model"]}')

    # load pre-trained model
    if config['source']:
        source_model = torch.load(os.path.join(
            PRETRAINED_DIR, config['source'], f'{config["backbone_model"]}.pth'), map_location=torch.device(config['device']))
        # remove weights of output layer
        del source_model[f'{config["final_layer"]}.weight']
        del source_model[f'{config["final_layer"]}.bias']
        # initialize target model with the state of source model
        backbone_model.load_state_dict(source_model, strict=False)
        if config['freeze']:
            # freeze all the network except the last embedding layer
            embedding_layer_params = [
                f'{config["penultimate_layer"]}.weight', f'{config["penultimate_layer"]}.bias']
            for name, param in backbone_model.named_parameters():
                if name not in embedding_layer_params:
                    param.requires_grad = False

    # few-shot model
    if config['method'] == 'LCP':
        fsl_model = LCProtonets(backbone_model, distance=config['dist'], normalize_distances=(
            config['dist'] == 'l2')).to(config['device'])
    else:
        fsl_model = Protonets(backbone_model, distance=config['dist'], normalize_distances=(
            config['dist'] == 'l2')).to(config['device'])

    # training batch sampler for multi-label few-shot learning
    n_way = config['n_way'] or len(config['tags']['train'])
    if config['method'] == 'OvR':
        train_sampler = OnevsRestSampler(
            train_dataset, n_way=n_way, k_shot=config[
                'k_shot'], tags=config['tags']['train'], n_task=config['n_task']
        )
        val_sampler = OnevsRestSampler(
            val_dataset, n_way=len(config['tags']['val']), k_shot=config[
                'k_shot'], tags=config['tags']['val'], n_task=config['n_valid_tasks'], is_test=True, input_length=val_dataset.input_length
        )
    else:
        train_sampler = MLFSLSampler(
            train_dataset, n_way=n_way, k_shot=config[
                'k_shot'], n_query=config['n_query'], n_task=config['n_task']
        )
        # validation batch sampler
        val_sampler = MLFSLSampler(
            val_dataset, n_way=len(config['tags']['val']), k_shot=config['k_shot'], n_query=None, n_task=config['n_valid_tasks']
        )

    # train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    # validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    source = f'_from_{config["source"]}' if config["source"] else ''
    freeze = '_f' if config['freeze'] else ''
    model_filename = f'{config["method"]}{source}{freeze}_{config["dist"]}.pth'

    saved_models_dir = os.path.join(SAVED_MODELS_DIR, config['dataset'])
    Path(saved_models_dir).mkdir(parents=True, exist_ok=True)

    config['save_path'] = os.path.join(saved_models_dir, model_filename)

    # logger
    config['logger'] = logging.getLogger()
    config['logger'].setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    logs_dir = os.path.join(LOGS_DIR)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(
        logs_dir, f'{config["dataset"]}_{model_filename.replace(".pth", ".log")}'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    config['logger'].addHandler(file_handler)
    # write info to logger
    config['logger'].info(config)
    config['logger'].info(tags_info_message)

    # training setup
    learning_rate = config['lr'] if config['source'] else config['finetuning_lr']
    config['optimizer'] = torch.optim.Adam(
        fsl_model.parameters(), learning_rate)
    config['loss_function'] = nn.BCEWithLogitsLoss()

    train_model(fsl_model, train_loader, val_loader, config)

    # free memory at the end
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str,
                        default='lyra', choices=DATASETS)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(DATA_DIR, 'lyra'))
    parser.add_argument('--method', type=str, default='LCP',
                        help='Method to be used for multi-label few-shot learning: "baseline" for "ML-PNs", "OvR" for "One-vs.-Rest", "LCP" for "LC-Protonets"', choices=['baseline', 'OvR', 'LCP'])
    parser.add_argument('--backbone_model', type=str,
                        default='vgg_ish', choices=['vgg_ish'], help='backbone architecture to be used')
    parser.add_argument('--dist', type=str, default='cos',
                        help='Distance to be used from the prototypes: "l2" for euclidean, "cos" for cosine distance.', choices=['l2', 'cos'])
    parser.add_argument("--freeze", type=lambda x: bool(strtobool(x)), nargs='?', const=True, default=False,
                        help='whether to freeze backbone model weights except the final embedding layer')
    parser.add_argument('--source', type=str,
                        default=None, choices=DATASETS + [None])
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='use "cpu" when no GPU is available, otherwise set the cuda index appropriately, e.g. "cuda:1".')
    parser.add_argument('--run_idx', type=str, default='1',
                        choices=['1', '2', '3', '4', '5'])
    args = parser.parse_args()

    # initialize dict from dataset configuration
    config = FSL_CONFIG['training'].copy()
    # include model configuration
    config.update(MODELS_CONFIG[args.backbone_model])
    # add other parameters
    config['dataset'] = args.dataset
    config['data_dir'] = args.data_dir
    config['method'] = args.method
    config['backbone_model'] = args.backbone_model
    config['dist'] = args.dist
    config['freeze'] = args.freeze
    config['source'] = args.source
    config['device'] = torch.device(args.device)
    config['run_idx'] = args.run_idx

    print(config)
    run_training(config)
