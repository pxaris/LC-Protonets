import os

# dir paths used in several places
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
LABELS_DIR = os.path.join(ROOT_DIR, 'labels')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
PRETRAINED_DIR = os.path.join(SAVED_MODELS_DIR, 'pretrained')
EVALUATIONS_DIR = os.path.join(ROOT_DIR, 'evaluation')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# seeds for the five runs
SEEDS = {
    "1": 3,
    "2": 4,
    "3": 7,
    "4": 13,
    "5": 21,
}

# FSL configuration per dataset
FSL_CONFIG = {
    'training': {
        'n_way': 10,  # number of labels for each task/episode
        'k_shot': 3,  # number of support items per label
        'n_query': 3,  # number of query items per label
        'n_task': 50,  # number of tasks (episodes) per epoch
        'n_valid_labels': 5,  # number of labels to be used for validation
        'n_valid_tasks': 20,  # number of tasks during validation
        'epochs': 200,  # total number of epochs
        'lr': 1e-5,  # learning rate while training from scratch
        'finetuning_lr': 1e-6,  # learning rate while fine-tuning a pre-trained model
        # number of consecutive epochs with worse validation metric before early stopping is activated
        'early_stopping_patience': 20
    }
}

DATASETS = [
    'magnatagatune',
    'fma',
    'lyra',
    'makam',
    'hindustani',
    'carnatic'
]

MODELS_CONFIG = {
    'vgg_ish': {
        'input_length_in_secs': 3.69,
        'final_layer': 'dense2',
        'penultimate_layer': 'dense1'
    },
}

# audio and mel-spectrograms attributes
SPECTROGRAMS_ATTRIBUTES = {
    'audio_sr': 16000,
    'n_fft': 512,
    'hop_length': 256,
    'f_min': 0.0,
    'f_max': 8000.0,
    'n_mels': 128,
}
