import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer
import librosa


def generate_csv_dataset():
    demo_info_path = "/content/drive/MyDrive/PROCESS-V1/demo-info.csv"
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

            if train_or_dev == 'Train':
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


def preprocess_text(csv_file):
    """
    There is something irrelevant in the text so you need to remove that and save the new csv file at the same location.
    """

def generate_submission():
    pass

