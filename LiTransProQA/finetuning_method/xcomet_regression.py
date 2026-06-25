# modified from https://github.com/Unbabel/COMET/
import os
import sys
import json
import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import wandb
import pandas as pd
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    TrainerCallback,
    PretrainedConfig,
    XLMRobertaTokenizerFast
)
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset
import argparse
from transformers import set_seed
from torch.utils.data import DataLoader

set_seed(42)

# Setup COMET path
COMET_ROOT = Path("COMET")
sys.path.append(str(COMET_ROOT))

from comet.models import download_model, load_from_checkpoint

logger = logging.getLogger(__name__)

class XCOMETRegressionModel(PreTrainedModel):
    """Custom model class for XCOMET regression."""
    
    def __init__(self, config, xcomet_model):
        super().__init__(config)
        self.xcomet_model = xcomet_model
        self.encoder = xcomet_model.encoder
        self.estimator = xcomet_model.estimator  # Get the estimator from xcomet_model
        
    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0) // 2  # Changed from 3 to 2 since we only have src and mt
        
        # Split inputs into source and target
        src_input_ids = input_ids[:batch_size]
        mt_input_ids = input_ids[batch_size:]
        
        src_attention_mask = attention_mask[:batch_size]
        mt_attention_mask = attention_mask[batch_size:]
        
        # Get embeddings from encoder
        src_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        mt_outputs = self.encoder(
            input_ids=mt_input_ids,
            attention_mask=mt_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get sentence embeddings
        src_sentemb = src_outputs['sentemb']
        mt_sentemb = mt_outputs['sentemb']
        
        # Compute scores using the estimator (same as XCOMET)
        src_scores = self.estimator(src_sentemb)
        mt_scores = self.estimator(mt_sentemb)
        
        # Combine scores (similar to XCOMET's weighted average)
        scores = (src_scores + mt_scores) / 2
        scores = scores.squeeze()  # Reshape from [batch_size, 1] to [batch_size]
        
        # Calculate loss if labels are provided
        if labels is not None:
            labels = labels.to(scores.device)
            loss = F.mse_loss(scores, labels)
        else:
            loss = None
        
        return {"loss": loss, "score": scores}
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model from pretrained checkpoint or local directory."""
        # First load the base XCOMET model
        xcomet_path = download_model('Unbabel/xcomet-xl')
        xcomet_model = load_from_checkpoint(xcomet_path)
        
        # Check if we're loading from a local checkpoint
        if os.path.isdir(pretrained_model_name_or_path):
            logger.info(f"Loading from local checkpoint: {pretrained_model_name_or_path}")
            # Load encoder state dict
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                encoder_state_dict = torch.load(model_path)
                xcomet_model.encoder.load_state_dict(encoder_state_dict)
                logger.info("Successfully loaded encoder state from checkpoint")
        
        # Freeze all parameters first
        for param in xcomet_model.encoder.parameters():
            param.requires_grad = False
        
        # Get training configuration
        train_layers = kwargs.pop('train_layers', 'quarter')
        total_layers = 36  # XLM-RoBERTa-XL has 36 layers
        
        # Calculate which layers to train
        layer_ranges = {
            'test': range(34, 36),    # Last 2 layers
            'quarter': range(27, 36),  # Last 9 layers
            'half': range(18, 36),    # Last 18 layers
            'full': range(0, 36)      # All layers
        }
        
        if train_layers not in layer_ranges:
            raise ValueError(f"Invalid train_layers value: {train_layers}")
            
        trainable_layer_names = [f'layer.{i}' for i in layer_ranges[train_layers]]
        
        # Unfreeze selected layers
        unfrozen_params = 0
        total_params = 0
        for name, param in xcomet_model.encoder.named_parameters():
            total_params += param.numel()
            if any(layer_name in name for layer_name in trainable_layer_names):
                param.requires_grad = True
                unfrozen_params += param.numel()
        
        logger.info(f"Training {train_layers} layers: {unfrozen_params}/{total_params} parameters ({(unfrozen_params/total_params)*100:.2f}%)")
        
        # Create config and model
        config = PretrainedConfig()
        config.hidden_size = kwargs.pop('hidden_size', 1024)
        config.dropout = kwargs.pop('dropout', 0.1)
        config.train_layers = train_layers
        return cls(config, xcomet_model)
    
    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        """Save only the encoder state."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Use provided state dict or get from encoder
        if state_dict is None:
            state_dict = self.encoder.state_dict()
        
        # Save only the encoder state dict
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save the config
        self.config.save_pretrained(save_directory)
        
        logger.info(f"Encoder saved to {save_directory}")

class RegressionDataCollator(DataCollatorMixin):
    """Custom data collator for regression tasks."""
    
    def __init__(self, pretrained_model: str = "facebook/xlm-roberta-xl", max_length: int = 128):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model)
        self.max_length = max_length
        
    def __call__(self, features):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            # Tokenize source and target
            src_encoding = self.tokenizer(
                feature["src"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            mt_encoding = self.tokenizer(
                feature["mt"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Combine encodings in order: src, mt
            batch["input_ids"].extend([
                src_encoding["input_ids"][0],
                mt_encoding["input_ids"][0]
            ])
            batch["attention_mask"].extend([
                src_encoding["attention_mask"][0],
                mt_encoding["attention_mask"][0]
            ])
            batch["labels"].append(float(feature["label"]))
        
        # Stack tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.tensor(batch["labels"])
        
        return batch

class ExperimentManager:
    """Manages experiment configuration and results."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
        
    def _load_config(self) -> Dict:
        """Load experiment configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path) as f:
            config = json.load(f)
            
        # Add timestamp and create experiment name
        config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get training configuration details
        train_layers = config.get('train_layers', 'quarter')
        training_data_name = Path(config['training_data_path']).stem
        base_model = Path(config['xcomet_model_path']).stem
        
        # Create more descriptive experiment name
        config['experiment_name'] = (
            f"regression_{training_data_name}"
            f"_layers-{train_layers}"
            f"_{config['timestamp']}"
        )
        
        # Create model save directory with layer info
        config['model_save_dir'] = os.path.join(
            config['output_dir'],
            'regression',
            f"{train_layers}_layers",
            base_model,
            training_data_name,
            config['timestamp']
        )
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'xcomet-regression'),
                name=config['experiment_name'],
                config=config,
                tags=[
                    f"regression",
                    f"layers_{train_layers}",
                    f"data_{training_data_name}",
                    f"model_{base_model}",
                    f"epochs_{config['num_epochs']}",
                    f"batch_{config['batch_size']}"
                ]
            )
            
        return config
        
    def _setup_directories(self):
        """Create necessary directories for the experiment."""
        model_dir = Path(self.config['model_save_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.dirs = {
            'base': Path(self.config['output_dir']),
            'models': model_dir,
            'results': model_dir / 'results',
            'logs': model_dir / 'logs',
            'checkpoints': model_dir / 'checkpoints'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        config_save_path = self.dirs['models'] / 'config.json'
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_results_path(self) -> str:
        return str(self.dirs['results'] / f"{self.config['experiment_name']}_results.json")
    
    def save_results(self, results: Dict):
        results_path = self.get_results_path()
        with open(results_path, 'w') as f:
            json.dump({**self.config, **results}, f, indent=4)
        logger.info(f"Results saved to {results_path}")

def process_columns(example):
    """Process columns for the dataset."""
    # Rename columns to match model expectations
    column_mapping = {
        'source': 'src',
        'target': 'mt',
        'score': 'label'
    }
    
    # Create new example with renamed columns
    new_example = {}
    for old_col, new_col in column_mapping.items():
        if old_col in example:
            new_example[new_col] = str(example[old_col])
        else:
            raise ValueError(f"Missing required column: {old_col}")
    
    return new_example

def train_model(
    experiment_manager: ExperimentManager,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None
):
    # Log GPU availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model with XCOMET's default parameters
    model = XCOMETRegressionModel.from_pretrained(
        experiment_manager.config['xcomet_model_path'],
        train_layers=experiment_manager.config.get('train_layers', 'quarter')
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Trainable parameters: {trainable_params}/{total_params}")
    
    # Calculate max_steps based on actual dataset size
    total_samples = experiment_manager.config['n_samples'] if not experiment_manager.config.get('debug', False) else 10
    steps_per_epoch = total_samples // experiment_manager.config['batch_size']
    max_steps = steps_per_epoch * experiment_manager.config['num_epochs']
    
    logger.info('***** Running training *****')
    logger.info(f"  Num epochs = {experiment_manager.config['num_epochs']}")
    logger.info(f"  Per device batch size = {experiment_manager.config['batch_size']}")
    logger.info(f"  Total samples = {total_samples}")
    logger.info(f"  Total optimization steps = {max_steps}")
    
    # Setup training arguments with XCOMET defaults
    training_args = TrainingArguments(
        output_dir=str(experiment_manager.dirs['checkpoints']),
        num_train_epochs=experiment_manager.config['num_epochs'],
        max_steps=max_steps,
        warmup_ratio=0.05,
        per_device_train_batch_size=experiment_manager.config['batch_size'],
        per_device_eval_batch_size=experiment_manager.config['batch_size'],
        gradient_accumulation_steps=experiment_manager.config.get('gradient_accumulation_steps', 4),
        learning_rate=experiment_manager.config.get('encoder_learning_rate', 2.0e-05),  # XCOMET default
        weight_decay=experiment_manager.config.get('encoder_weight_decay', 0.01),
        evaluation_strategy="steps",
        eval_steps=experiment_manager.config.get('eval_steps', 100),
        logging_strategy="steps",
        logging_steps=experiment_manager.config.get('logging_steps', 100),
        save_strategy="steps",
        save_steps=experiment_manager.config.get('save_steps', 100),  # Changed to match eval_steps
        save_total_limit=2,  # Keep only the last 2 checkpoints
        load_best_model_at_end=False,  # Enable loading best model
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if experiment_manager.config.get('use_wandb', True) else None,
        fp16=True,
        no_cuda=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        disable_tqdm=False,
        label_names=["labels"],
    )
    
    # Initialize data collator
    data_collator = RegressionDataCollator(
        pretrained_model="facebook/xlm-roberta-xl",
        max_length=experiment_manager.config.get('max_length', 512)
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,     
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train()
    
    # Finish wandb run
    if experiment_manager.config.get('use_wandb', True):
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description='Train XCOMET Regression Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Use debug configuration and small dataset')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load experiment configuration
    config_path = "configs/regression_debug.json" if args.debug else args.config
    experiment_manager = ExperimentManager(config_path)
    
    # Override wandb if specified
    if args.no_wandb:
        experiment_manager.config['use_wandb'] = False
    # Load datasets in streaming mode
    df = pd.read_csv(experiment_manager.config['training_data_path'])
    experiment_manager.config['n_samples'] = len(df)
    del df

    train_dataset = Dataset.from_csv(
        experiment_manager.config['training_data_path'],
        streaming=True  # Enable streaming mode
    )

    # Apply processing to streaming datasets
    train_dataset = train_dataset.map(process_columns)
    eval_dataset = train_dataset.take(int(experiment_manager.config['n_samples'] * 0.2))

    # Shuffle the datasets
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)

    if args.debug:
        experiment_manager.config['debug'] = True
        experiment_manager.config['num_epochs'] = 1
        experiment_manager.config['batch_size'] = 2
        train_dataset = train_dataset.take(10)
        eval_dataset = train_dataset.take(4)
        print("Debug mode enabled")
    
    logger.info("Datasets loaded and shuffled")
    
    # Train the model
    train_model(experiment_manager, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
