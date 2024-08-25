import torch
from trainer import Trainer  # Import Trainer class
from model import YourModel  # Replace with your actual model import
from custom_dataloader import YourDataset  
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = 'config.yaml'  # Path to your YAML configuration file
    cfg = load_config(config_path)
    print(cfg)

    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = YourModel()
    
    # Load datasets
    train_csv_path = cfg['data']['train_csv_path']
    val_csv_path = cfg['data']['val_csv_path'] 
    
    fbank_params = {
        "num_mel_bins": cfg['fbanks']['num_mel_bins'],
        "sample_frequency": cfg['fbanks']['sample_frequency'],
        "use_energy": cfg['fbanks']['use_energy']
    }

    wav2vec2_model_name = cfg['wav2vec2_model_name']
    bert_model_name = cfg['bert_model_name']

    train_dataset = YourDataset(train_csv_path, wav2vec2_model_name, bert_model_name, fbank_params)
    val_dataset = YourDataset(val_csv_path, wav2vec2_model_name, bert_model_name, fbank_params)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg['training']['batch_size'],
        learning_rate=cfg['training']['learning_rate'],
        device=device
    )

    # Training
    trainer.fit(epochs=cfg['training']['epochs'])

    # Save checkpoint
    checkpoint_path = cfg['training']['checkpoint_path']
    trainer.save_checkpoint(checkpoint_path)

    # Load checkpoint
    # trainer.load_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()