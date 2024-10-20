import torch
from trainer import Trainer 
from model import CustomModel  
from custom_dataloader import CustomAudioTextDataset 
from utils import model_size_in_mb, update_csv
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = '/content/process/config.yaml'
    cfg = load_config(config_path)
    print(cfg)
    print("\033[34mConfig Loaded...\033[0m")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\033[34mDevice: 'cuda'\033[0m")
    input_dims = [512,25,2048]#768]#512] #egmaps = 25, trill = 2048,w2v2=512,w2v2phonetic = 512, bert - 768
    bert_dir = cfg['bert_model_name']
    num_labels = cfg['model']['num_labels']
    embed_dim = cfg['model']['embed_dim']
    num_heads = cfg['model']['num_head']
    depth = cfg['model']['depth']
    update_bert = cfg['model']['update_bert']
    phoneme_model_train = cfg['phoneme_model_train']
    phoneme_model_val = cfg['phoneme_model_val']
    egmaps_train = cfg['egmpas_feats_train']
    egmaps_val = cfg['egmpas_feats_val']
    trill_train = cfg['trill_feats_train']
    trill_val = cfg['trill_feats_val']
    bert_dir_train = cfg['bert_dir_train']
    bert_dir_val = cfg['bert_dir_val']
    epochs=cfg['training']['epochs']

    model = CustomModel(embed_dim,num_heads,num_labels,bert_dir,input_dims,update_bert,depth)
    print(f"Model size: {model_size_in_mb(model):.2f} MB")
    print("\033[34mModel Loaded...\033[0m")
    train_csv_path = cfg['data']['train_csv_path']
    val_csv_path = cfg['data']['val_csv_path']
    train_csv_path = update_csv(train_csv_path)
    val_csv_path = update_csv(val_csv_path) 
    
    fbank_params = {
        "num_mel_bins": cfg['fbanks']['num_mel_bins'],
        "sample_frequency": cfg['fbanks']['sample_frequency'],
        "use_energy": cfg['fbanks']['use_energy'],
        "frame_shift":cfg['fbanks']['frame_shift'],
        "frame_length":cfg['fbanks']['frame_length']
    }

    wav2vec2_model_name_train = cfg['wav2vec2_model_name_train']
    wav2vec2_model_name_val = cfg['wav2vec2_model_name_val']
    max_len = cfg['data']['max_len']

    train_dataset = CustomAudioTextDataset(train_csv_path, wav2vec2_model_name_train, fbank_params,bert_dir_train,egmaps_train,trill_train,phoneme_model_train,max_len)
    val_dataset = CustomAudioTextDataset(val_csv_path, wav2vec2_model_name_val, fbank_params,bert_dir_val,egmaps_val,trill_val,phoneme_model_val,max_len)
    print("\033[34mDataset created...!\033[0m")
    print("\033[34mTrainer ready...\033[0m")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg['training']['batch_size'],
        learning_rate=cfg['training']['learning_rate'],
        wt_decay= cfg['training']['weight_decay'],
        device=device,
        epochs = epochs,
        warmup_steps=cfg['training']['warmup_steps'],
        embed_dim=embed_dim
    )
    
    print("\033[34mTraining started...\033[0m")
    trainer.fit(epochs=cfg['training']['epochs'])

    checkpoint_path = cfg['training']['checkpoint_path']
    trainer.save_checkpoint(checkpoint_path)
    print("\033[34mModel checkpoint created...\033[0m")
if __name__ == "__main__":
    main()