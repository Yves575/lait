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

class XCOMETRankingModel(PreTrainedModel):
    """Custom model class for XCOMET ranking."""
    
    def __init__(self, config, xcomet_model):
        super().__init__(config)
        
        # Store the full xcomet_model for tokenizer access
        self.xcomet_model = xcomet_model
        # Only keep the encoder part for training
        self.encoder = xcomet_model.encoder
        
        # Explicitly set encoder to train mode and unfreeze parameters
        #self.encoder.train()
        #for param in self.encoder.parameters():
        #    param.requires_grad = False
              
        self.ranking_loss = nn.TripletMarginLoss(margin=config.ranking_margin, p=2)
        
    def get_sentence_embedding(self, input_ids, attention_mask):
        """Get sentence embeddings using the encoder."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # CHECKUP: Debug prints for encoder outputs
        #logger.info(f"Encoder output keys: {outputs.keys()}")
        #logger.info(f"Encoder output type: {type(outputs)}")
    
        return outputs['sentemb']  # Use the sentence embedding directly
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Split inputs into source, reference, positive, and negative
        batch_size = input_ids.size(0) // 4
        src_input_ids = input_ids[:batch_size]
        ref_input_ids = input_ids[batch_size:2*batch_size]
        pos_input_ids = input_ids[2*batch_size:3*batch_size]
        neg_input_ids = input_ids[3*batch_size:]
        
        src_attention_mask = attention_mask[:batch_size]
        ref_attention_mask = attention_mask[batch_size:2*batch_size]
        pos_attention_mask = attention_mask[2*batch_size:3*batch_size]
        neg_attention_mask = attention_mask[3*batch_size:]
        
        # Get embeddings
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
        pos_sentemb = self.get_sentence_embedding(pos_input_ids, pos_attention_mask)
        neg_sentemb = self.get_sentence_embedding(neg_input_ids, neg_attention_mask)
        
        # Calculate loss
        loss = self.ranking_loss(src_sentemb, pos_sentemb, neg_sentemb) + \
               self.ranking_loss(ref_sentemb, pos_sentemb, neg_sentemb)
        
        # Calculate distances for metrics
        distance_src_pos = F.pairwise_distance(pos_sentemb, src_sentemb)
        distance_ref_pos = F.pairwise_distance(pos_sentemb, ref_sentemb)
        distance_pos = (2 * distance_src_pos * distance_ref_pos) / (distance_src_pos + distance_ref_pos)
        
        distance_src_neg = F.pairwise_distance(neg_sentemb, src_sentemb)
        distance_ref_neg = F.pairwise_distance(neg_sentemb, ref_sentemb)
        distance_neg = (2 * distance_src_neg * distance_ref_neg) / (distance_src_neg + distance_ref_neg)
        
        outputs = {
            "loss": loss,  # Always include loss
            "distance_pos": distance_pos,
            "distance_neg": distance_neg,
            "embeddings": torch.cat([src_sentemb, ref_sentemb, pos_sentemb, neg_sentemb], dim=0)
        }
        
        return outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if not pretrained_model_name_or_path.endswith(".ckpt"):
            pretrained_model_name_or_path = download_model(pretrained_model_name_or_path)
        
        xcomet_model = load_from_checkpoint(pretrained_model_name_or_path)
        
        # CHANGE: First freeze all parameters
        for param in xcomet_model.encoder.parameters():
            param.requires_grad = False
        
        # Get training configuration
        train_layers = kwargs.pop('train_layers', 'quarter')  # 'quarter', 'half', or 'full'
        
        # Calculate which layers to train
        total_layers = 36  # XLM-RoBERTa-XL has 36 layers
        
        if train_layers == 'test':
            trainable_layer_names = [f'layer.{i}' for i in range(34, 36)]  # Last 2 layers
        elif train_layers == 'quarter':
            trainable_layer_names = [f'layer.{i}' for i in range(27, 36)]  # Last 9 layers
        elif train_layers == 'half':
            trainable_layer_names = [f'layer.{i}' for i in range(18, 36)]  # Last 18 layers
        elif train_layers == 'full':
            trainable_layer_names = [f'layer.{i}' for i in range(0, 36)]  # All layers
        else:
            raise ValueError(f"Invalid train_layers value: {train_layers}")
        
        unfrozen_params = 0
        total_params = 0
        
        # Count total parameters and unfreeze specific layers
        for name, param in xcomet_model.encoder.named_parameters():
            total_params += param.numel()
            # Print all parameter names to debug
            logger.info(f"Parameter name: {name}")
            if any(layer_name in name for layer_name in trainable_layer_names):
                param.requires_grad = True
                unfrozen_params += param.numel()
                logger.info(f"Unfrozen parameter: {name}")
        
        # CHANGE: Verify trainable parameters
        trainable_found = False
        for name, param in xcomet_model.encoder.named_parameters():
            if param.requires_grad:
                trainable_found = True
                logger.info(f"Trainable parameter found: {name}")
        
        if not trainable_found:
            raise ValueError("No trainable parameters found in the encoder!")
        
        # Log parameter stats
        logger.info(f"Training {train_layers} layers: {unfrozen_params}/{total_params} parameters ({(unfrozen_params/total_params)*100:.2f}%)")
        
        # Create config and model
        config = PretrainedConfig()
        config.ranking_margin = kwargs.pop('ranking_margin', 1.0)
        config.train_layers = train_layers  # Store training configuration
        model = cls(config, xcomet_model)
        
        return model
    
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

class RankingDataCollator(DataCollatorMixin):
    """Custom data collator for ranking tasks."""
    
    def __init__(self, pretrained_model: str = "facebook/xlm-roberta-xl", max_length: int = 128):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model)
        self.max_length = max_length
        
    def __call__(self, features):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []  # Add labels key
        }
        
        for feature in features:
            # Tokenize all sentences
            src_encoding = self.tokenizer(
                feature["src"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            ref_encoding = self.tokenizer(
                feature["ref"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            pos_encoding = self.tokenizer(
                feature["pos"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            neg_encoding = self.tokenizer(
                feature["neg"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Combine all encodings in order: src, ref, pos, neg
            batch["input_ids"].extend([
                src_encoding["input_ids"][0],
                ref_encoding["input_ids"][0],
                pos_encoding["input_ids"][0],
                neg_encoding["input_ids"][0]
            ])
            batch["attention_mask"].extend([
                src_encoding["attention_mask"][0],
                ref_encoding["attention_mask"][0],
                pos_encoding["attention_mask"][0],
                neg_encoding["attention_mask"][0]
            ])
        
        # Stack tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.zeros(len(features))  # Dummy labels since we compute loss internally
        
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
        
        # Create more descriptive experiment name
        config['experiment_name'] = (
            f"{training_data_name}"
            f"_layers-{train_layers}"
            f"_lr{config['encoder_learning_rate']}"
            f"_e{config['num_epochs']}"
            f"_b{config['batch_size']}"
            f"_{config['timestamp']}"
        )
        
        # Create model save directory with detailed information
        config['model_save_dir'] = os.path.join(
            config['base_directory'],
            'models',
            f"{train_layers}_layers",
            training_data_name,
            config['timestamp']
        )
        
        # Initialize wandb with detailed config
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'xcomet-ranking'),
                name=config['experiment_name'],
                config=config,
                tags=[
                    f"layers_{train_layers}",
                    f"data_{training_data_name}",
                    f"epochs_{config['num_epochs']}",
                    f"batch_{config['batch_size']}"
                ]
            )
            
        return config
        
    def _setup_directories(self):
        """Create necessary directories for the experiment."""
        # Create model save directory with detailed structure
        model_dir = Path(self.config['model_save_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create other directories
        self.dirs = {
            'base': Path(self.config['base_directory']),
            'models': model_dir,
            'results': model_dir / 'results',
            'logs': model_dir / 'logs',
            'checkpoints': model_dir / 'checkpoints'
        }
        
        # Create directories if they don't exist
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Save config file in the experiment directory
        config_save_path = self.dirs['models'] / 'config.json'
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_results_path(self) -> str:
        """Get path for saving results."""
        return str(self.dirs['results'] / f"{self.config['experiment_name']}_results.json")
    
    def save_results(self, results: Dict):
        """Save training results."""
        results_path = self.get_results_path()
        with open(results_path, 'w') as f:
            json.dump({**self.config, **results}, f, indent=4)
        logger.info(f"Results saved to {results_path}")

def verify_encoder_only_training(model: XCOMETRankingModel):
    """Verify that only encoder parameters are being trained."""
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    #logger.info("Trainable parameters:")
    #for param in trainable_params:
    #    logger.info(f"  {param}")
    
    logger.info("\nFrozen parameters:")
    for param in frozen_params:
        logger.info(f"  {param}")

def train_model(
    experiment_manager: ExperimentManager,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None
):
    # Log GPU availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model with layer configuration
    model = XCOMETRankingModel.from_pretrained(
        experiment_manager.config['xcomet_model_path'],
        ranking_margin=experiment_manager.config.get('ranking_margin', 1.0),
        train_layers=experiment_manager.config.get('train_layers', 'quarter')
    )
    
    # Verify encoder is unfrozen and in train mode
    logger.info("Checking encoder training status:")
    logger.info(f"Encoder training mode: {model.encoder.training}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Trainable parameters: {trainable_params}/{total_params}")
    
    #if trainable_params == 0:
     #   logger.error("Encoder is completely frozen! Unfreezing parameters...")
      #  model.encoder.train()
       # for param in model.encoder.parameters():
        #    param.requires_grad = True
        #trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        #logger.info(f"After unfreezing: {trainable_params}/{total_params} parameters are trainable")
    
    #verify_encoder_only_training(model)
    
    # Add batch size and steps calculation before training args
    # Calculate max_steps based on epochs and batch size
    total_samples = experiment_manager.config.get('total_train_samples', 50000)  # Get from config or estimate
    steps_per_epoch = total_samples // experiment_manager.config['batch_size']
    max_steps = steps_per_epoch * experiment_manager.config['num_epochs']
    
    logger.info('***** Running training *****')
    logger.info(f"  Num epochs = {experiment_manager.config['num_epochs']}")
    logger.info(f"  Per device batch size = {experiment_manager.config['batch_size']}")
    logger.info(f"  Total optimization steps = {max_steps}")
    
    # Setup training arguments with GPU settings
    training_args = TrainingArguments(
        output_dir=str(experiment_manager.dirs['checkpoints']),
        num_train_epochs=experiment_manager.config['num_epochs'],
        max_steps = max_steps,
        warmup_ratio = 0.05, 
        per_device_train_batch_size=experiment_manager.config['batch_size'],
        per_device_eval_batch_size=experiment_manager.config['batch_size'],
        gradient_accumulation_steps=experiment_manager.config.get('gradient_accumulation_steps', 1),
        learning_rate=experiment_manager.config.get('encoder_learning_rate', 1e-5),
        weight_decay=experiment_manager.config.get('encoder_weight_decay', 0.01),
        evaluation_strategy="steps",
        eval_steps=25,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=experiment_manager.config.get('save_steps', 150),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if experiment_manager.config.get('use_wandb', True) else None,
        fp16=True,
        no_cuda=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        disable_tqdm=False,
        label_names=["labels"],
    )
    
    # Initialize data collator with direct tokenizer loading
    data_collator = RankingDataCollator(
        pretrained_model="facebook/xlm-roberta-xl",  # or your specific model name
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
    parser = argparse.ArgumentParser(description='Train XCOMET Ranking Model')
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
    config_path = "configs/ranking_debug.json" if args.debug else args.config
    experiment_manager = ExperimentManager(config_path)
    
    # Override wandb if specified
    if args.no_wandb:
        experiment_manager.config['use_wandb'] = False
    
    # Load training and validation datasets
    '''
    train_df = pd.read_csv(experiment_manager.config['training_data_path'])
    if args.debug:
        train_df = train_df.iloc[:100]
        eval_df  = train_df
    else:
        eval_df  = pd.read_csv(experiment_manager.config['validation_data_path'])
    '''
    # Load datasets in streaming mode
    train_dataset = Dataset.from_csv(
        experiment_manager.config['training_data_path'],
        streaming=True  # Enable streaming mode
    )
    eval_dataset = Dataset.from_csv(
        experiment_manager.config['validation_data_path'],
        streaming=True  # Enable streaming mode
    ) if not args.debug else train_dataset

    # Process required columns
    def process_columns(example):
        if 'ref' not in example:
            example['ref'] = example['src']
        return {k: str(v) for k, v in example.items()}


    '''
    # Ensure required columns exist
    required_columns = ['src', 'ref', 'pos', 'neg']
    if 'ref' not in train_df.columns:
        train_df['ref'] = train_df['src']
        eval_df['ref'] = eval_df['src']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
   ''' 
    # Apply processing to streaming datasets
    train_dataset = train_dataset.map(process_columns)
    eval_dataset = eval_dataset.map(process_columns)

    if args.debug:
        train_dataset = train_dataset.take(10)
        eval_dataset = train_dataset
        
    #logger.info(f"Training with {len(train_dataset)} examples")
    #logger.info(f"Evaluating with {len(eval_dataset)} examples")
    
    # Train the model
    train_model(experiment_manager, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
