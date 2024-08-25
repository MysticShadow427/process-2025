import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
import pandas as pd


class CustomAudioTextDataset(Dataset):
    def __init__(self, csv_file, wav2vec2_model_name, bert_model_name, fbank_params, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.fbank_params = fbank_params
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the row
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        text = row['text']
        classification_label = row['classification_label']
        regression_label = row['regression_label']

        # 1. Compute FBanks
        waveform, sample_rate = torchaudio.load(audio_path)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, **self.fbank_params)

        # 2. Compute wav2vec2 embeddings
        input_values = self.wav2vec2_processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values
        with torch.no_grad():
            wav2vec2_embeddings = self.wav2vec2_model(input_values).last_hidden_state.squeeze(0)

        # 3. Compute BERT embeddings (excluding CLS token)
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=self.max_length)
        with torch.no_grad():
            bert_outputs = self.bert_model(**inputs)
            bert_embeddings = bert_outputs.last_hidden_state.squeeze(0)[1:]  # Exclude the CLS token
        
        # Return list of features and both classification and regression labels
        features = [fbank, wav2vec2_embeddings, bert_embeddings]
        return features, (torch.tensor(classification_label, dtype=torch.long), torch.tensor(regression_label, dtype=torch.float))

def collate_fn(batch):
    # Unpack the batch into features and labels
    features, labels = zip(*batch)
    
    # Stack each type of feature separately
    fbank_features = [item[0] for item in features]
    wav2vec2_features = [item[1] for item in features]
    bert_features = [item[2] for item in features]
    
    fbank_features = torch.nn.utils.rnn.pad_sequence(fbank_features, batch_first=True)
    wav2vec2_features = torch.nn.utils.rnn.pad_sequence(wav2vec2_features, batch_first=True)
    bert_features = torch.nn.utils.rnn.pad_sequence(bert_features, batch_first=True)
    
    # Stack the labels
    classification_labels = torch.stack([label[0] for label in labels])
    regression_labels = torch.stack([label[1] for label in labels])
    
    return [fbank_features, wav2vec2_features, bert_features], (classification_labels, regression_labels)



# Create the DataLoader
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)



# not sure to use bert features as cross attention as we dont have text in the test data, so we will use bert as classification head