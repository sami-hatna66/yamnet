import torch
import pandas as pd
import torchaudio


class ESC50Data(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        data_dir = data_dir if data_dir.endswith('/') else data_dir + '/'
        self.annotations = pd.read_csv(data_dir + "meta/esc50.csv")
        self.audio_dir = data_dir + "audio/"
    
    def __len__(self):
        return len(self.annotations)
    
    def get_labels(self):
        return torch.Tensor(self.annotations['target'].unique().tolist())
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        audio_file_path = self.audio_dir + self.annotations.iloc[idx, 0]
        label = int(self.annotations.iloc[idx, 2])
        
        wav, sr = torchaudio.load(audio_file_path)
        
        return {
            "id": str(idx),
            "data": wav,
            "sr": sr,
            "label": label
        }


class FSD50KData(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        data_dir = data_dir if data_dir.endswith('/') else data_dir + '/'
        self.annotations = pd.read_csv(data_dir + "FSD50K.metadata/collection/collection_dev.csv")
        self.audio_dir = data_dir

        self.annotations['labels'] = self.annotations['labels'].str.split(',').str[0]

        unique_labels = self.annotations['labels'].unique()
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.annotations['label_id'] = self.annotations['labels'].map(lambda x: label_to_id[x])

    def __len__(self):
        return len(self.annotations)
    
    def get_labels(self):
        return torch.Tensor(self.annotations['label_id'].unique().tolist())
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        audio_path = self.audio_dir + str(self.annotations.iloc[idx, 0]) + ".wav"
        label = int(self.annotations.iloc[idx, 3])

        wav, sr = torchaudio.load(audio_path)
    
        return {
            'id': str(idx),
            'data': wav,
            'sr': sr,
            'label': label
        }
        