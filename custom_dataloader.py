import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import numpy as np
import pandas as pd
from torchaudio.transforms import Resample
from utils import clean_transcription_text
import pickle
import torchaudio.transforms as T
from audiomentations import Trim

class CustomAudioTextDataset(Dataset):
    def __init__(self, csv_file, wav2vec2_model_name, fbank_params, bert_dir,egmaps_dir,trill_embeds,phoneme_model,max_length=100):
        self.data = pd.read_csv(csv_file)
        with open(wav2vec2_model_name, 'rb') as f:
             self.w2v2 = pickle.load(f) # list of numpy arrays
        with open(phoneme_model, 'rb') as f:
             self.phonemes = pickle.load(f)
        with open(bert_dir,'rb') as f:
            self.bert_feats = pickle.load(f)
        with open(egmaps_dir,'rb') as f:
            self.egmaps_feats = pickle.load(f)
        with open(trill_embeds,'rb') as f:
            self.trill_embeds = pickle.load(f)
        
        self.data['transcription_text'] = self.data['transcription_text'].apply(clean_transcription_text)
        label_mapping = {
                'MCI': 0,
                'HC': 1,
                'Dementia': 2
            }

        self.data['class_label'] = self.data['class_label'].map(label_mapping)

        self.fbank_params = fbank_params
        self.max_length = max_length
        self.phoneme_model = phoneme_model
        self.speed_pertubation = T.SpeedPerturbation(16000, [0.9, 1.1, 1.0, 1.0, 1.0])
        self.freq_masking = T.FrequencyMasking(freq_mask_param=30) 
        self.time_masking = T.TimeMasking(time_mask_param=40)
        self.trim = Trim(top_db=25.0,p=1.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        text = row['transcription_text']
        classification_label = row['class_label']
        regression_label = row['converted_mmse']
        # wav2vec2 embeddings
        wav2vec2_embeddings = self.w2v2[idx]
        # phonetic features
        phonetic_features = self.phonemes[idx]
        # bert embeddings
        bert_embeddings = self.bert_feats[idx]
        # egmaps features
        egmaps_feats = self.egmaps_feats[idx].values
        # trill embeddings
        trill_embeddings = self.trill_embeds[idx]

        # FBanks
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16_000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16_000)
            waveform = resampler(waveform)
        
        #trim
        waveform = self.trim(waveform,16000)
        # speed pertubation
        waveform,_ = self.speed_pertubation(waveform)
        # fbank featurs
        fbank = torchaudio.compliance.kaldi.fbank(waveform=waveform, num_mel_bins=self.fbank_params['num_mel_bins'],frame_length=self.fbank_params['frame_length'],frame_shift=self.fbank_params['frame_shift'])
        # spec augment
        fbank = self.freq_masking(fbank)
        fbank = self.time_masking(fbank)
        features = {
            'fbank' : fbank.clone().detach().to(dtype=torch.float),
            'wav2vec2_embeddings' : torch.tensor(wav2vec2_embeddings,dtype=torch.float),
            'egmaps_feats' : torch.tensor(egmaps_feats,dtype=torch.float),
            'trill_embeddings' : torch.tensor(trill_embeddings,dtype=torch.float),
            'phonetic_features' : torch.tensor(phonetic_features,dtype=torch.float),
            'bert_embeddings' : torch.tensor(bert_embeddings,dtype=torch.float)
        }

        return features, (torch.tensor(classification_label, dtype=torch.long), torch.tensor(regression_label, dtype=torch.float))


def collate_fn(batch):
    features, labels = zip(*batch)

    fbank_features = [item['fbank'] for item in features]
    wav2vec2_features = [item['wav2vec2_embeddings'] for item in features]
    egmap_features = [item['egmaps_feats'] for item in features]
    trill_features = [item['trill_embeddings'] for item in features]
    phonetic_features = [item['phonetic_features'] for item in features]
    bert_features = [item['bert_embeddings'] for item in features]

    # def normalize(features):
    #     features = torch.stack(features)
    #     mean = features.mean(dim=0, keepdim=True)
    #     std = features.std(dim=0, keepdim=True)
    #     return (features - mean) / (std + 1e-8)
    def normalize(features):
        norms = torch.norm(features, p=2, dim=2, keepdim=True)
        features= features/ norms
        return features
    

    fbank_features = pad_sequence(fbank_features, batch_first=True)
    wav2vec2_features = pad_sequence(wav2vec2_features, batch_first=True)
    egmap_features = pad_sequence(egmap_features, batch_first=True)
    trill_features = pad_sequence(trill_features, batch_first=True)
    phonetic_features = pad_sequence(phonetic_features, batch_first=True)

    egmap_features = normalize(egmap_features)
    trill_features = normalize(trill_features)
    phonetic_features = normalize(phonetic_features)
    bert_features = torch.stack(bert_features)


    classification_labels = torch.stack([label[0] for label in labels])
    regression_labels = torch.stack([label[1] for label in labels])

    feats = {
        'fbank_features': fbank_features,
        'wav2vec2_features': wav2vec2_features,
        'egmap_features': egmap_features,
        'trill_features': trill_features,
        'phonetic_features': phonetic_features,
        'bert_features' : bert_features
    }
    # total_batch_size_in_mb = calculate_batch_size_in_mb(feats)
    # print(f"Total batch size: {total_batch_size_in_mb:.2f} MB")

    return feats, (classification_labels, regression_labels)

def get_tensor_size_in_mb(tensor):
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size() / (1024 ** 2)

def calculate_batch_size_in_mb(batch_features):
    total_size = 0
    for name, feature in batch_features.items():
        total_size += get_tensor_size_in_mb(feature)
    return total_size
