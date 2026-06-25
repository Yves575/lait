# -*- coding: utf-8 -*-
import os
import json
import logging
import torch
import sys
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from transformers import PreTrainedModel
import pandas as pd
import argparse

COMET_ROOT = Path("COMET")
sys.path.append(str(COMET_ROOT))
from comet.models import download_model, load_from_checkpoint
from comet.models.utils import Prediction

file_ROOT = Path(path_to_file)
sys.path.append(str(file_ROOT))
from xcomet_ranking import XCOMETRankingModel

logger = logging.getLogger(__name__)


class XCOMETInference:
    """Class to run inference with XCOMET using a finetuned encoder"""
    
    def __init__(
        self,
        xcomet_model_path: str = "Unbabel/xcomet-xl",
        finetuned_encoder_path: str = None,
        batch_size: int = 8,
        max_length: int = 128,
        output_dir: str = "inference_outputs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store model paths for reference
        self.model_info = {
            "xcomet_model": xcomet_model_path,
            "finetuned_encoder": finetuned_encoder_path,
            "inference_timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }

        # Load XCOMET model - add download step
        logger.info(f"Loading XCOMET model from {xcomet_model_path}")
        if not xcomet_model_path.endswith(".ckpt"):
            xcomet_model_path = download_model(xcomet_model_path)
        self.model = load_from_checkpoint(xcomet_model_path)
        
        # Load finetuned encoder if provided
        if finetuned_encoder_path:
            logger.info(f"Loading finetuned encoder from {finetuned_encoder_path}")
            # Load the finetuned encoder state dict
            encoder_state_dict = torch.load(os.path.join(finetuned_encoder_path, "pytorch_model.bin"))
            # Load it into the model's encoder
            self.model.encoder.load_state_dict(encoder_state_dict)
            logger.info("Successfully loaded finetuned encoder")
            
        # Move model to device
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model moved to {device} and set to eval mode")

    def prepare_inputs(
        self, 
        sources: List[str],
        translations: List[str],
        references: Optional[List[str]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Prepare inputs for XCOMET model."""
        # CHANGE: Format inputs as COMET expects
        samples = []
        for src, mt in zip(sources, translations):
            sample = {
                "src": src,
                "mt": mt,
            }
            if references:
                sample["ref"] = references[i]
            samples.append(sample)
        
        return samples

    @torch.no_grad()
    def score(
        self,
        sources: List[str],
        translations: List[str],
        references: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        save_output: bool = True,
        output_name: Optional[str] = None,
    ) -> Dict:
        """Score translations using XCOMET model."""
        if batch_size is None:
            batch_size = self.batch_size

        # CHANGE: Prepare inputs in COMET format
        samples = self.prepare_inputs(sources, translations, references)

        # Get predictions using COMET's reference-free mode
        outputs = self.model.predict(
            samples=samples,
            batch_size=batch_size,
            gpus=1 if self.device == "cuda" else 0,
            progress_bar=show_progress,
        )

        # Organize results
        results = {
            "model_info": self.model_info,
            "system_score": outputs.system_score,
            "segment_scores": outputs.scores,
            "segments": []
        }

        # Add detailed segment information
        for i, (src, mt, score) in enumerate(zip(sources, translations, outputs.scores)):
            segment = {
                "id": i,
                "source": src,
                "translation": mt,
                "score": score
            }
            
            if references:
                segment["reference"] = references[i]
                
            if hasattr(outputs, "metadata") and hasattr(outputs.metadata, "error_spans"):
                if outputs.metadata.error_spans[i]:
                    segment["error_spans"] = outputs.metadata.error_spans[i]
                    
            results["segments"].append(segment)

        # Save results if requested
        if save_output:
            output_name = output_name or f"xcomet_output_{self.model_info['inference_timestamp']}.json"
            output_path = self.output_dir / output_name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results

def main(model_name:str, args, df:pd.DataFrame):
    # Initialize inference

    inference = XCOMETInference(
        xcomet_model_path="Unbabel/xcomet-xl",
        finetuned_encoder_path= "outputs/half_layers/models/half_layers/" + model_name if args.finetuned_model else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir
    )

    # Test examples
    if args.test:
        sources = df["src"].iloc[:10].tolist()
        translations = df["tgt"].iloc[:10].tolist()
    else:
        sources = df["src"].tolist()
        translations = df["tgt"].tolist()

    print("\n=== Reference-free Scoring with Final Model ===")
    if not args.finetuned_model:
        model_name = "Unbabel/xcomet-xl"
    results = inference.score(
        sources=sources,
        translations=translations,  # No references needed
        output_name="referencefree_output_{model_name}.json".format(model_name=model_name.replace("/", "_"))
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XCOMET Inference')
    parser.add_argument('--df_path', type=str, default='../cleaned_benchmark_dataset/benchmark_dataset_all_src_tgt.csv', help='Path to the benchmark dataset')
    parser.add_argument('--test', action='store_true', help='Run on test examples')
    parser.add_argument('--model_name', type=str, default='20250320_030754', help='Model name')
    parser.add_argument('--finetuned_model', action='store_true', help='Use finetuned model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Max length')
    parser.add_argument('--output_dir', type=str, default='experiments/xcomet_ranking/', help='Output directory')
    args = parser.parse_args()

    df = pd.read_csv(args.df_path)
    main(args.model_name, args, df) 
