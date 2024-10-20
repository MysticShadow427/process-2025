import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer
import librosa
import re



def generate_csv_dataset():
    demo_info_path = "/content/drive/MyDrive/process2025/dem-info.csv"
    base_folder_path = "/content/drive/MyDrive/PROCESS-V1"
    train_output_csv = "/content/drive/MyDrive/process2025/train_dataset_process.csv"
    val_output_csv = "/content/drive/MyDrive/process2025/val_dataset_process.csv"

    demo_df = pd.read_csv(demo_info_path)

    train_data = []
    dev_data = []

    for idx, row in demo_df.iterrows():
        record_id = row['Record-ID']  # The folder name
        train_or_dev = row['TrainOrDev']  # Train or dev set
        classification_label = row['Class']  # Dementia, MCI, HC
        mmse_score = row['Converted-MMSE']  # MMSE score

        subject_folder = os.path.join(base_folder_path, record_id)
        
        # Iterate through the three tasks (CTD, PFT, SFT)
        for task in ['CTD', 'PFT', 'SFT']:
            
            audio_file = os.path.join(subject_folder, f"{record_id}__{task}.wav")
            transcription_file = os.path.join(subject_folder, f"{record_id}__{task}.txt")

            if os.path.exists(transcription_file):
                with open(transcription_file, 'r') as f:
                    transcription_text = f.read().strip()
            else:
                transcription_text = ""  

            if train_or_dev == 'train':
                train_data.append([audio_file, transcription_text, classification_label, mmse_score])
            else:
                dev_data.append([audio_file, transcription_text, classification_label, mmse_score])

    random.shuffle(train_data)
    random.shuffle(dev_data)

    train_df = pd.DataFrame(train_data, columns=['audio_path', 'transcription_text', 'class_label', 'converted_mmse'])
    dev_df = pd.DataFrame(dev_data, columns=['audio_path', 'transcription_text', 'class_label', 'converted_mmse'])

    train_df.to_csv(train_output_csv, index=False)
    dev_df.to_csv(val_output_csv, index=False)

    print(f"Train CSV saved to: {train_output_csv}")
    print(f"Dev CSV saved to: {val_output_csv}")

def clean_transcription_text(text):
    # Remove '{any_name}:' prefix
    text = re.sub(r'^[^:]+:\s*', '', text)
    
    # Remove patterns like (x seconds), (1 second), {x sounds}
    text = re.sub(r'\(\d+\s*(second|seconds)\)', '', text)
    text = re.sub(r'{\d+\s*(sound|sounds)}', '', text)
    
    # Remove any text inside parentheses ()
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def plot_audio_length_distribution(csv_file_path):
    df = pd.read_csv(csv_file_path)

    audio_lengths = []
    
    for idx, row in df.iterrows():
        audio_path = row['audio_path'] 
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            audio_lengths.append(duration)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")

    plt.figure(figsize=(10, 6))
    plt.hist(audio_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Audio Lengths (seconds)', fontsize=16)
    plt.xlabel('Length (seconds)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()


def plot_bert_tokenized_text_length_distribution(csv_file_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(csv_file_path)
    
    token_lengths = []

    for idx, row in df.iterrows():
        text = row['transcription_text'] 
        
        tokens = tokenizer.tokenize(text)
        
        token_lengths.append(len(tokens))

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color='lightcoral', edgecolor='black')
    plt.title('Distribution of BERT Tokenized Text Lengths', fontsize=16)
    plt.xlabel('Number of Tokens', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

def extract_and_save_columns(df):
    selected_columns = df[['transcription_text', 'class_label', 'converted_mmse']]
    return selected_columns


# Function to fill missing values in 'converted_mmse'
def fill_missing_mmse(df):
    for i, row in df[df['Converted-MMSE'].isnull()].iterrows():
        # Filter data for the same 'TrainorDev' and 'class'
        filtered_df = df[(df['TrainOrDev'] == row['TrainOrDev']) & (df['Class'] == row['Class'])]
        
        # Calculate the mean of 'converted_mmse' for the filtered data
        mean_mmse = filtered_df['Converted-MMSE'].mean()
        
        # Fill the missing value with the mean
        df.at[i, 'Converted-MMSE'] = mean_mmse
    
    return df

def update_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # Modify the 'audio_path' column by removing '/drive/MyDrive'
    df['audio_path'] = df['audio_path'].str.replace('/drive/MyDrive', '/scratch/dkayande/process/data', regex=False)
    
    # Get the original file name
    file_name = os.path.basename(csv_file_path)
    
    # Save the updated CSV file to /content/ with the same name
    output_file_path = os.path.join('/scratch/dkayande/process/data', file_name)
    
    df.to_csv(output_file_path, index=False)
    
    print(f"Updated CSV saved at: {output_file_path}")
    return output_file_path

def generate_submission():
    pass

def model_size_in_mb(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()  # Number of elements * size per element (in bytes)
    
    for buffer in model.buffers():
        total_params += buffer.numel() * buffer.element_size()  # Also consider buffers

    total_params_in_mb = total_params / (1024 ** 2)  # Convert from bytes to megabytes
    return total_params_in_mb
# FEATURE EXTRACTION CODE 

# import audiofile
# import librosa
# import opensmile
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoTokenizer, AutoModel
# from text2phonemesequence import Text2PhonemeSequence
# import tensorflow_hub as hub
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
# assert tf.executing_eagerly()

# self.wav2vec2_featureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_model_name).to('cuda')
        # self.wav2vec2_model = Wav2Vec2Model.from_pretrained(wav2vec2_model_name).to('cuda')
        # for param in self.wav2vec2_featureExtractor:
        #     param.requires_grad = False
        # for param in self.wav2vec2_model:
        #     param.requires_grad = False
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # self.bert = AutoModel.from_pretrained(bert_dir).to('cuda')
        # for param in self.bert:
        #     param.requires_grad = False
        # self.smile = opensmile.Smile(
        #     feature_set=opensmile.FeatureSet.eGeMAPSv02,
        #     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        #     sampling_rate=16000,
        #     resample=True,
        # )
        # self.non_semantic_module = hub.load('https://kaggle.com/models/google/nonsemantic-speech-benchmark/frameworks/TensorFlow2/variations/frill/versions/1')

        # self.phoneme_model = None
        # if phoneme_model == 'vinai/xphonebert-base':
        #     self.phoneme_model = AutoModel.from_pretrained("vinai/xphonebert-base").to('cuda')
        #     self.phoneme_tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        #     self.t2p = Text2PhonemeSequence(language='eng-us', is_cuda=True)
        # else:
        #     self.phoneme_model = Wav2Vec2Model.from_pretrained('vitouphy/wav2vec2-xls-r-300m-phoneme').to('cuda')
        #     self.phoneme_featureExtractor = Wav2Vec2FeatureExtractor.from_pretrained('vitouphy/wav2vec2-xls-r-300m-phoneme').to('cuda')
        # for param in self.phoneme_model:
        #     param.requires_grad = False
        # for param in self.phoneme_featureExtractor:
        #     param.requires_grad = False

        # wav2vec2 embeddings
        # input_values = self.wav2vec2_featureExtractor(waveform.to('cuda').squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        # with torch.no_grad():
        #     input_values = input_values.to('cuda')
        #     wav2vec2_embeddings = self.wav2vec2_model(input_values).extract_features.squeeze(dim=0).cpu().numpy()
        
        # eGeMAPS
        # signal,sampling_rate = audiofile.read(audio_path)
        # egmaps_feats = self.smile.process_signal(
        #     signal,
        #     sampling_rate
        # )

        # trill embeddings
        # signal, sampling_rate = librosa.load(audio_path, sr=None)
        # if sampling_rate != 16_000:
        #     signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=16_000)
        # signal = np.expand_dims(signal,axis=0)
        # trill_embeddings = self.non_semantic_module(signal)['embedding'].numpy()

        # phonetic features - allosaurus :( not getting it so using diff
        # phonetic_features = None
        # if self.phoneme_model == 'vinai/xphonebert-base':
        #     input_phonemes = self.t2p.infer_sentence(text)
        #     input_ids = self.phoneme_tokenizer(input_phonemes, return_tensors="pt")

        #     with torch.no_grad():
        #         phonetic_features = self.phoneme_model(**input_ids)
        # else:
        #     input_values_p = self.wav2vec2_featureExtractor(waveform.to('cuda').squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        # with torch.no_grad():
        #     input_values_p = input_values_p.to('cuda')
        #     phonetic_features = self.phoneme_model(input_values_p).extract_features.squeeze(dim=0).cpu().numpy()

        # bert text embeddings
        # encoding = self.tokenizer.encode_plus(
        #     text,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     return_token_type_ids=False,
        #     padding='max_length',
        #     return_attention_mask=True,
        #     return_tensors='pt',
        #     truncation=True
        # )

        # input_ids = encoding['input_ids']
        # attention_mask = encoding['attention_mask']

        # with torch.no_grad():
        #     bert_embeddings,_ = self.bert(input_ids,attention_mask)



if __name__ == '__main__':
    pass