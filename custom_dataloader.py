import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import audiofile
import librosa
import opensmile
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoTokenizer, AutoModel
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()


class CustomAudioTextDataset(Dataset):
    def __init__(self, csv_file, wav2vec2_model_name, fbank_params, max_length=100):
        self.data = pd.read_csv(csv_file)
        self.wav2vec2_featureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(wav2vec2_model_name).to('cuda')
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            sampling_rate=8000,
            resample=True,
        )
        self.non_semantic_module = hub.load('https://kaggle.com/models/google/nonsemantic-speech-benchmark/frameworks/TensorFlow2/variations/frill/versions/1')


        self.fbank_params = fbank_params
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        # text = row['text']
        classification_label = row['classification_label']
        regression_label = row['regression_label']

        # FBanks
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16_000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16_000)
            waveform = resampler(waveform)
        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, **self.fbank_params)

        # wav2vec2 embeddings
        input_values = self.wav2vec2_featureExtractor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            input_values = input_values.to('cuda')
            wav2vec2_embeddings = self.wav2vec2_model(input_values).extract_features.squeeze(dim=0).cpu().numpy()
        
        # eGeMAPS
        signal,sampling_rate = audiofile.read(audio_path)
        egmaps_feats = self.smile.process_signal(
            signal,
            sampling_rate
        )

        # trill embeddings
        signal, sampling_rate = librosa.load(audio_path, sr=None)
        if sampling_rate != 16_000:
            signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=16_000)
        signal = np.expand_dims(signal,axis=0)
        trill_embeddings = self.non_semantic_module(signal)['embedding'].numpy()

        # phonetic features - allosaurus
        phonetic_features = None

        features = {
            'fbank' : torch.tensor(fbank,dtype=torch.float),
            'wav2vec2_embeddings' : torch.tensor(wav2vec2_embeddings,dtype=torch.float),
            'egmaps_feats' : torch.tensor(egmaps_feats,dtype=torch.float),
            'trill_embeddings' : torch.tensor(trill_embeddings,dtype=torch.float),
            'phonetic_features' : torch.tensor(phonetic_features,dtype=torch.float)
        }

        return features, (torch.tensor(classification_label, dtype=torch.long), torch.tensor(regression_label, dtype=torch.float))


def collate_fn(batch):
    features, labels = zip(*batch)

    fbank_features = [item['fbank'] for item in features]
    wav2vec2_features = [item['wav2vec2_embeddings'] for item in features]
    egmap_features = [item['egmaps_feats'] for item in features]
    trill_features = [item['trill_embeddings'] for item in features]
    phonetic_features = [item['phonetic_features'] for item in features]

    fbank_features = pad_sequence(fbank_features, batch_first=True)
    wav2vec2_features = pad_sequence(wav2vec2_features, batch_first=True)
    egmap_features = pad_sequence(egmap_features, batch_first=True)
    trill_features = pad_sequence(trill_features, batch_first=True)
    phonetic_features = pad_sequence(phonetic_features, batch_first=True)

    classification_labels = torch.stack([label[0] for label in labels])
    regression_labels = torch.stack([label[1] for label in labels])

    feats = {
        'fbank_features': fbank_features,
        'wav2vec2_features': wav2vec2_features,
        'egmap_features': egmap_features,
        'trill_features': trill_features,
        'phonetic_features': phonetic_features
    }

    return feats, (classification_labels, regression_labels)
