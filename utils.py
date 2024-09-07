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


def generate_submission():
    pass

if __name__ == '__main__':
    pass