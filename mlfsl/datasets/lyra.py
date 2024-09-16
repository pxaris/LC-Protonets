import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from config import SPECTROGRAMS_ATTRIBUTES


class LyraDatasetFSL(Dataset):

    val_size = 0.1  # ratio to be used on dataset splits

    def __init__(self,
                 data_path,
                 tags,
                 mel_specs_config=SPECTROGRAMS_ATTRIBUTES,
                 input_length_in_secs=3,
                 split='train',
                 ):

        if split not in ['train', 'valid', 'test']:
            raise ValueError(
                f'split argument must take one of the following values: "train", "valid", "test"')
        self.split = split
        self._tags = tags
        self.mel_specs_config = mel_specs_config

        self.input_length_in_secs = input_length_in_secs
        self.input_length = int(self.input_length_in_secs *
                                self.mel_specs_config['audio_sr']/self.mel_specs_config['hop_length'])

        self.data_path = data_path
        keyword = 'training' if self.split in ['train', 'valid'] else 'test'
        annotations_file = os.path.join(
            data_path, 'split', keyword + '.tsv')
        self.specs_dir = os.path.join(data_path, 'mel-spectrograms')

        self.all_ids, self._all_labels = self.get_ids_labels(annotations_file)

        self.ids, self._labels = [], []
        self.mel_specs = []
        self.labels = []
        for piece_id, label_list in zip(self.all_ids, self._all_labels):
            spectrogram_file = os.path.join(self.specs_dir, piece_id + '.npy')
            if self.split in ['train', 'valid']:
                piece_mel = self.get_random_spec_segment_from_file(
                    spectrogram_file)
            else:
                piece_mel = self.get_spectrogram_from_file(spectrogram_file)

            # include only the item's labels that exist in the specified target labels
            target_label_list = list(set(label_list) & set(self._tags))
            # do not include items that do not have any target label
            if target_label_list:
                self.mel_specs += piece_mel
                self.labels += [target_label_list]
                self.ids.append(piece_id)
                self._labels += [label_list]

        self.len_labels = len(self._tags)

        self.label_transformer = MultiLabelBinarizer(classes=self._tags)
        self.labels = np.array(
            self.label_transformer.fit_transform(self.labels)
        ).astype("int64")

    def get_random_spec_segment_from_file(self, spectrogram_file):
        spectrogram = np.load(spectrogram_file).T
        random_idx = int(np.floor(np.random.random(
            1) * (len(spectrogram)-self.input_length)))
        return [np.array(spectrogram[random_idx:random_idx+self.input_length])]

    @staticmethod
    def get_spectrogram_from_file(spectrogram_file):
        return [np.load(spectrogram_file).T]

    def get_ids_labels(self, annotations_file):
        data_df = pd.read_csv(annotations_file, sep='\t',
                              keep_default_na=False)
        if self.split in ['train', 'valid']:
            n_val_items = int(len(data_df) * self.val_size)
            n_samples = n_val_items if self.split == 'valid' else len(
                data_df) - n_val_items
            # a same item can be sampled for both training and validation sets
            # but its labels will be totally different in the two sets
            data_df = data_df.sample(n=n_samples, random_state=12)

        ids, labels = [], []
        for _, row in data_df.iterrows():
            ids.append(row['id'])
            label_list = []
            for col_name in ['instruments', 'genres', 'place']:
                label_list += [
                    f'{col_name}--{label}' for label in row[col_name].split('|')]

            labels.append(label_list)

        return ids, labels

    def __getitem__(self, item):
        return self.mel_specs[item], self.labels[item]

    def __len__(self):
        return len(self.labels)
