import numpy as np
from mlfsl.datasets.lyra import LyraDatasetFSL
from mlfsl.datasets.magnatagatune import MagnatagatuneFSL
from mlfsl.datasets.ccml_dataset import CCMLDatasetFSL


def get_train_val_dataset(config):

    if config['dataset'] == 'lyra':
        dataset_class = LyraDatasetFSL
    elif config['dataset'] == 'magnatagatune':
        dataset_class = MagnatagatuneFSL
    elif config['dataset'] in ['fma', 'makam', 'carnatic', 'hindustani']:
        dataset_class = CCMLDatasetFSL
    else:
        raise NotImplementedError(
            f'No implementation found for dataset {config["dataset"]}')

    train_dataset = dataset_class(
        config['data_dir'],
        config['tags']['train'],
        input_length_in_secs=config['input_length_in_secs'],
        split='train',
    )
    val_dataset = dataset_class(
        config['data_dir'],
        config['tags']['val'],
        input_length_in_secs=config['input_length_in_secs'],
        split='valid',
    )

    return train_dataset, val_dataset


def get_test_dataset(config):

    if config['dataset'] == 'lyra':
        dataset_class = LyraDatasetFSL
    elif config['dataset'] == 'magnatagatune':
        dataset_class = MagnatagatuneFSL
    elif config['dataset'] in ['fma', 'makam', 'carnatic', 'hindustani']:
        dataset_class = CCMLDatasetFSL
    else:
        raise NotImplementedError(
            f'No implementation found for dataset {config["dataset"]}')

    test_dataset = dataset_class(
        config['data_dir'],
        config['selected_tags'],
        input_length_in_secs=config['input_length_in_secs'],
        split='test',
    )

    return test_dataset


def split_spectrogram(spectrogram, split_length, keep_residual=False):
    spectr_length = spectrogram.shape[0]
    num_spectrs, residual_length = int(
        spectr_length/split_length), spectr_length % split_length

    if num_spectrs == 0:
        return None

    specgram_to_split = spectrogram[:-
                                    residual_length] if residual_length else spectrogram
    splitted_spectrogram = np.split(specgram_to_split, num_spectrs)
    if keep_residual and residual_length:
        splitted_spectrogram += [spectrogram[-residual_length:]]

    return splitted_spectrogram
