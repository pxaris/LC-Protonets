import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# add project root to path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(current_dir)))

from mlfsl.datasets.utils import get_test_dataset
from mlfsl.models.vgg_ish import VGGish
from mlfsl.utils import evaluate, evaluate_ovr
from mlfsl.samplers.mlfsl_sampler import MLFSLSampler
from mlfsl.samplers.ovr_sampler import OnevsRestSampler
from mlfsl.protonets import Protonets
from mlfsl.lc_protonets import LCProtonets
from config import MODELS_CONFIG, SAVED_MODELS_DIR, DATASETS, DATA_DIR, LABELS_DIR, EVALUATIONS_DIR, SEEDS


def run_evaluation(config):
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

    # select the classes according to the defined type
    if config['type'] == 'novel':
        config['selected_tags'] = config['tags']['test'][:config['n_way']]
    elif config['type'] == 'base':
        config['selected_tags'] = config['tags']['train'][:config['n_way']]
    elif config['type'] == 'both':
        if config['n_way'] >= len(config['tags']['train']):
            config['selected_tags'] = [*config['tags']['train'],
                                       *config['tags']['test']][:config['n_way']]
        else:
            n_base = int(config['n_way']/2)
            n_novel = config['n_way'] - n_base
            config['selected_tags'] = config['tags']['train'][:n_base] + \
                config['tags']['test'][:n_novel]
    else:
        raise ValueError(
            f'Incorrect type: {config["type"]}')

    print(f'Number of tags: {len(config["selected_tags"])}')

    # test dataset
    test_dataset = get_test_dataset(config)

    # backbone model
    backbone_model = VGGish(n_class=len(test_dataset._tags),
                            is_backbone=True).to(config['device'])

    # load pre-trained model state
    if 'pretrained' in config['source']:
        model_path = os.path.join(
            SAVED_MODELS_DIR, config['source'], f'{config["model"]}.pth')
    else:
        model_path = os.path.join(
            SAVED_MODELS_DIR, config['source'], f'{config["model"]}_{config["dist"]}.pth')
    stored_model = torch.load(
        model_path, map_location=torch.device(config['device']))

    # initialize few-shot model and load the trained model state
    if 'pretrained' in config['source']:
        # remove weights of output layer
        del stored_model[f'{config["final_layer"]}.weight']
        del stored_model[f'{config["final_layer"]}.bias']
        # initialize target model with the state of source model
        backbone_model.load_state_dict(stored_model, strict=False)
        if config['method'] == 'LCP':
            fsl_model = LCProtonets(backbone_model, distance=config['dist'], normalize_distances=(
                config['dist'] == 'l2')).to(config['device'])
        else:
            fsl_model = Protonets(backbone_model, distance=config['dist'], normalize_distances=(
                config['dist'] == 'l2')).to(config['device'])
    else:
        if config['method'] == 'LCP':
            fsl_model = LCProtonets(backbone_model, distance=config['dist'], normalize_distances=(
                config['dist'] == 'l2')).to(config['device'])
        else:
            fsl_model = Protonets(backbone_model, distance=config['dist'], normalize_distances=(
                config['dist'] == 'l2')).to(config['device'])
        fsl_model.load_state_dict(stored_model, strict=False)

    # test sampler for multi-label few-shot learning
    if config['method'] == 'OvR':
        test_sampler = OnevsRestSampler(
            test_dataset, n_way=config['n_way'], k_shot=config[
                'k_shot'], tags=config['selected_tags'], n_task=1, is_test=True, input_length=test_dataset.input_length
        )
    else:
        test_sampler = MLFSLSampler(
            test_dataset, n_way=config['n_way'], k_shot=config['k_shot'], n_query=None, n_task=1, is_test=True, input_length=test_dataset.input_length
        )
    # test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    if config['method'] == 'OvR':
        macro_f1, micro_f1, clf_report, info_messages = evaluate_ovr(
            fsl_model, test_loader, config)
    else:
        macro_f1, micro_f1, clf_report, info_messages = evaluate(
            fsl_model, test_loader, config)
    keyword = f'{config["source"]}/{config["model"]}' if 'pretrained' in config['source'] else f'{config["source"]}/{config["model"]}_{config["dist"]}'
    result = [
        f'\nEvaluation of model "{keyword}" on "{config["dataset"]}" test set, with\n- N-way: {config["n_way"]}\n- K-shot: {config["k_shot"]}\n- distance: {config["dist"]}']
    result += [
        f'\nTest set evaluation:\n- macro-f1: {macro_f1}\n- micro-f1: {micro_f1}\n']
    result += [f'\nClassification report:\n{clf_report}']
    result += info_messages

    # write result to file
    output_dir = os.path.join(EVALUATIONS_DIR, args.dataset)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    keyword = f'{config["method"]}_pretrained_' if 'pretrained' in config['source'] else ''
    output_filename = f'{config["n_way"]}_way_{config["type"]}_{keyword}{config["model"]}_{config["dist"]}.txt'
    output_file = os.path.join(output_dir, output_filename)

    with open(output_file, 'w', encoding='utf-8') as f:
        for value in result:
            f.write(str(value) + '\n')
    print(f'\tDone. Values were written to file {output_file}')

    # free memory at the end
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str,
                        default='magnatagatune', choices=DATASETS)
    parser.add_argument('--data_dir', type=str,
                        help='directory where the "mel-spectrograms" and "split" dirs are expected to be found', default=os.path.join(DATA_DIR, 'magnatagatune'))
    parser.add_argument('--method', type=str, default='LCP',
                        help='Method to be used for multi-label few-shot learning: "baseline" for "ML-PNs", "OvR" for "One-vs.-Rest", "LCP" for "LC-Protonets"', choices=['baseline', 'OvR', 'LCP'])
    parser.add_argument('--dist', type=str, default='cos',
                        help='Distance to be used from the prototypes: "l2" for euclidean, "cos" for cosine distance.', choices=['l2', 'cos'])
    parser.add_argument('--model', type=str, default='baseline',
                        help='trained model to be used')
    parser.add_argument('--N', type=int, default=5,
                        help='N-way, the number of labels to be included.')
    parser.add_argument('--K', type=int, default=3,
                        help='K-shot, the number of support items per label.')
    parser.add_argument('--type', type=str, default='novel',
                        help='Whether to use "base", "novel" or "both" types of classes during the evaluation', choices=['base', 'novel', 'both'])
    parser.add_argument('--source', type=str, default='magnatagatune',
                        help='directory from which to load the model, e.g. "magnatagatune", "pretrained/makam" etc.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='use "cpu" when no GPU is available, otherwise set the cuda index appropriately, e.g. "cuda:1".')
    parser.add_argument('--run_idx', type=str, default='1',
                        help='set the run_idx so that a different seed will be used for different runs', choices=['1', '2', '3', '4', '5'])
    args = parser.parse_args()

    # initialize dict with model configuration
    config = MODELS_CONFIG['vgg_ish'].copy()
    # add other parameters
    config['dataset'] = args.dataset
    config['data_dir'] = args.data_dir
    config['method'] = args.method
    config['dist'] = args.dist
    config['model'] = args.model
    config['n_way'] = args.N
    config['k_shot'] = args.K
    config['type'] = args.type
    config['source'] = args.source or args.dataset
    config['device'] = torch.device(args.device)
    config['run_idx'] = args.run_idx

    print(config)
    run_evaluation(config)
