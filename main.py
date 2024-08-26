import torch
from trainer import Trainer 
from model import CustomModel  
from custom_dataloader import CustomAudioTextDataset 
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = '/content/process/config.yaml'
    cfg = load_config(config_path)
    print(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dims = []
    bert_dir = cfg['bert_model_name']
    num_labels = cfg['num_labels']
    embed_dim = cfg['embed_dim']
    num_heads = cfg['num_head']
    model = CustomModel(embed_dim,num_heads,num_labels,bert_dir)
    
    train_csv_path = cfg['data']['train_csv_path']
    val_csv_path = cfg['data']['val_csv_path'] 
    
    fbank_params = {
        "num_mel_bins": cfg['fbanks']['num_mel_bins'],
        "sample_frequency": cfg['fbanks']['sample_frequency'],
        "use_energy": cfg['fbanks']['use_energy']
    }

    wav2vec2_model_name = cfg['wav2vec2_model_name']
    max_len = cfg['max_len']

    train_dataset = CustomAudioTextDataset(train_csv_path, wav2vec2_model_name, fbank_params,bert_dir,max_len)
    val_dataset = CustomAudioTextDataset(val_csv_path, wav2vec2_model_name, fbank_params,bert_dir,max_len)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg['training']['batch_size'],
        learning_rate=cfg['training']['learning_rate'],
        wt_decay= cfg['weight_decay'],
        device=device
    )

    trainer.fit(epochs=cfg['training']['epochs'])

    checkpoint_path = cfg['training']['checkpoint_path']
    trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()