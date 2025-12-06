##---------------------- Only to run on the colab environment------------------------##

import torch
import pandas as pd
import sys
from pathlib import Path

sys.path.append("/content/pyHGT-implementation")

from train import HGTConfig, HGTTrainer

def load_trained_model(checkpoint_path: str) -> HGTTrainer:
    """
    Load a trained HGT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        
    Returns:
        An instance of HGTTrainer with loaded weights.
    """
    config = HGTConfig()  # Use default config or modify as needed
    trainer = HGTTrainer(config)
    
    checkpoint = torch.load(checkpoint_path)
    trainer.gnn_model.load_state_dict(checkpoint['gnn_model'])
    trainer.disease_predictor.load_state_dict(checkpoint['disease_predictor'])
    
    print(f"Loaded trained model from {checkpoint_path}")
    return trainer

def run_testing(trainer: HGTTrainer):
    """Run testing on the trained model and print metrics."""
    test_metrics = trainer.evaluate(trainer.test_ids, split_name='test')
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return test_metrics

def generate_predictions(trainer, save_csv=True):
    """Get per-patient disease predictions and probabilities."""

    patient_embeddings = trainer.get_patient_embeddings().detach()

    #test patient indices
    test_indices = [trainer.entity_maps['patient'][pid] for pid in trainer.test_ids]
    test_embeddings = patient_embeddings[test_indices]

    disease_logits = trainer.disease_predictor(test_embeddings)
    disease_probs = torch.sigmoid(disease_logits)

    #convert to DataFrame
    df = pd.DataFrame(disease_probs.numpy(), columns=trainer.disease_names)
    df.insert(0, 'PatientID', trainer.test_ids)

    if save_csv:
        output_path = 'outputs/predictions.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
    return df

# ==============================================================================

def main():
    trainer = load_trained_model('outputs/checkpoints/best_model.pt')
    run_testing(trainer)
    df = generate_predictions(trainer)
    print(df.head())

if __name__ == '__main__':
    main()

    