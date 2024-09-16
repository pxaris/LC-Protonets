import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from config import SPECTROGRAMS_ATTRIBUTES


class CCMLDatasetFSL(Dataset):
    def __init__(self,
                 data_path,
                 tags,
                 mel_specs_config=SPECTROGRAMS_ATTRIBUTES,
                 input_length_in_secs=3,
                 split='train',
                 ):
        self.data_path = data_path
        self.specs_dir = os.path.join(data_path, 'mel-spectrograms')
        self.mel_specs_config = mel_specs_config
        self._tags = tags

        self.input_length_in_secs = input_length_in_secs
        self.input_length = int(self.input_length_in_secs *
                                self.mel_specs_config['audio_sr']/self.mel_specs_config['hop_length'])

        self.split_path = os.path.join(data_path, 'split')
        self.split = split

        if self.split not in ['train', 'test', 'valid']:
            raise ValueError(
                f'split must take one of the values: "train", "test", "valid"')

        self.all_ids = np.load(os.path.join(
            self.split_path, f'{self.split}.npy'), allow_pickle=True).tolist()
        
        self.metadata = np.load(os.path.join(
            self.split_path, 'metadata.npy'), allow_pickle=True).item()
        self._all_labels = [self.metadata[id] for id in self.all_ids]

        self.ids, self._labels = [], []
        self.labels = []
        for piece_id, label_list in zip(self.all_ids, self._all_labels):
            # include only the item's labels that exist in the specified target labels 
            target_label_list = list(set(label_list) & set(self._tags))
            # do not include items that do not have any target label
            if target_label_list:
                self.labels += [target_label_list]
                self.ids.append(piece_id)
                self._labels += [label_list]
        
        self.len_labels = len(self._tags)

        self.label_transformer = MultiLabelBinarizer(classes=self._tags)
        self.labels = np.array(
            self.label_transformer.fit_transform(self.labels)
        ).astype("int64")

    def __getitem__(self, item):
        spectrogram, tags_binary = self.get_spectrogram_and_tags(item)
        return spectrogram.astype('float32'), tags_binary.astype('float32')

    def get_spectrogram_and_tags(self, item):
        spec_path = os.path.join(self.specs_dir, f'{self.ids[item]}.npy')
        spectrogram = np.load(spec_path, mmap_mode='r').T

        if self.split in ['train', 'valid']:
            # get a single chunk randomly
            random_idx = int(np.floor(np.random.random(1) *
                                      (len(spectrogram)-self.input_length)))
            spectrogram = np.array(
                spectrogram[random_idx:random_idx+self.input_length])

        tags_binary = self.labels[item]

        return spectrogram, tags_binary

    def __len__(self):
        return len(self.ids)
