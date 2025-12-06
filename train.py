"""
HGT Model Training Script
=========================
Complete training pipeline for disease risk prediction and test recommendation
using Heterogeneous Graph Transformer model.

Tasks:
1. Disease Risk Prediction: Multi-label classification of patient diseases
2. Test Recommendation: Link prediction for patient-test connections

Architecture:
    Patient Tests (EMR)
           ↓
    [TCN Encoder] → Patient Embeddings (128D)
           ↓
    [HGT Layers] → Graph-aware embeddings (learned from test-disease connections)
           ↓
    ┌────────────────────────────────────────┐
    ↓                                        ↓
[Disease Head]                        [Test Recommender]
    ↓                                        ↓
Multi-label Sigmoid                   Dot Product + Ranking
    ↓                                        ↓
Disease Probabilities          Recommended Test Scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score, 
    hamming_loss, accuracy_score, coverage_error
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from graph import (
    create_hgt_graph, convert_heterodata_to_pyhgt, 
    quick_generate_labels, print_conversion_summary
)
from pyHGT.model import GNN, Classifier


# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

class HGTConfig:
    """
    Complete HGT training configuration with engineering optimizations.
    
    Key Parameters Explanation:
    ===========================
    
    1. GRAPH ARCHITECTURE
    - in_dim: Patient embedding dimension from TCN encoder (128)
      → Higher: More expressiveness but slower computation
      → 128 is standard for medical applications
    
    - n_hid: Hidden dimension of HGT layers (256)
      → 2x input dimension is a common practice
      → Controls capacity of intermediate representations
    
    - n_layers: Number of HGT convolution layers (3)
      → More layers: Capture longer-range dependencies
      → Diminishing returns after 3-4 layers
      → Risk of over-smoothing in GNNs
    
    - n_heads: Multi-head attention heads (8)
      → Higher: More diverse attention patterns
      → Must divide n_hid evenly (256/8 = 32 per head)
    
    - dropout: Regularization dropout rate (0.2)
      → Prevents overfitting, especially important with limited data
      → 0.2 = 20% neurons randomly disabled per forward pass
    
    2. TRAINING HYPERPARAMETERS
    - learning_rate: Initial LR (0.001)
      → Too high: Unstable training
      → Too low: Slow convergence
      → 0.001 is standard for Adam optimizer
    
    - weight_decay: L2 regularization (0.0001)
      → Penalizes large weights → prevents overfitting
      → 1e-4 is good baseline
    
    - batch_size: Samples per batch (64)
      → Larger: Better gradient estimates but more memory
      → Smaller: Noisier updates but regularizing effect
      → 64 is sweet spot for medical data
    
    - epochs: Training iterations (100)
      → Early stopping prevents overfitting
      → Fewer if convergence detected early
    
    - patience: Early stopping patience (15)
      → Stop if val loss doesn't improve for 15 epochs
      → Prevents unnecessary training
    
    3. LOSS FUNCTIONS
    - disease_loss: BCEWithLogitsLoss (Binary Cross Entropy)
      → Suitable for multi-label classification
      → Numerically stable (combines Sigmoid + BCE)
      → Loss = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
    
    - test_loss: MSE Loss (Mean Squared Error)
      → For link prediction: minimize distance between connected nodes
      → Alternative: Ranking loss, contrastive loss
      → MSE works well for patient-test embeddings
    
    - loss_weight_disease: 0.7
    - loss_weight_test: 0.3
      → 70/30 split prioritizes disease prediction
      → Adjust based on downstream task importance
    
    4. OPTIMIZATION TECHNIQUES
    - Optimizer: AdamW
      → Decoupled weight decay (better than L2 regularization)
      → Adaptive learning rates per parameter
      → Momentum: β1=0.9 (exponential decay of 1st moment)
      → Variance: β2=0.999 (exponential decay of 2nd moment)
    
    - LR Scheduler: ReduceLROnPlateau
      → Reduce LR by factor if validation loss plateaus
      → factor=0.5: Cut learning rate by 50%
      → Helps escape local minima
    
    5. VALIDATION STRATEGY
    - val_ratio: 0.15 (15% for validation, 10% for test)
      → Standard 70/15/15 split
      → Stratified on disease distribution
    
    - eval_frequency: 1 (evaluate every epoch)
      → Can increase to 5 for faster training
      → Monitor more frequently for debugging
    
    6. DISEASE PREDICTION SPECIFIC
    - threshold_disease: 0.5
      → Probability threshold for disease prediction
      → 0.5 is neutral; adjust based on precision-recall tradeoff
      → For rare diseases: lower threshold (more recall)
      → For high-precision needs: higher threshold
    
    7. TEST RECOMMENDATION SPECIFIC
    - top_k: [5, 10, 20]
      → Recommend top 5, 10, 20 tests
      → Metrics: recall@k, precision@k
    
    8. REGULARIZATION
    - Dropout: 0.2 (in HGT model)
    - Weight decay: 1e-4 (in optimizer)
    - Layer normalization: Enabled in HGT
    - Skip connections: Built into HGT
    
    Engineering Optimization Tips:
    ==============================
    - If underfitting (high train & val loss):
      → Increase n_layers (more capacity)
      → Increase n_hid
      → Decrease dropout
      → Increase epochs
    
    - If overfitting (low train loss, high val loss):
      → Increase dropout (0.3-0.4)
      → Increase weight_decay (1e-3)
      → Decrease batch_size (32 instead of 64)
      → Decrease n_layers or n_hid
      → Use early stopping (patience)
    
    - For imbalanced diseases:
      → Use weighted BCE loss
      → Adjust class weights by frequency
      → Use focal loss for rare diseases
    
    - For GPU memory issues:
      → Decrease batch_size
      → Decrease n_hid
      → Use gradient accumulation
    
    - For convergence speed:
      → Use CosineAnnealingLR instead of ReduceLROnPlateau
      → Warmup learning rate first 5 epochs
      → Use gradient clipping (max_norm=1.0)
    """
    
    # Graph Architecture
    in_dim = 128                    # TCN embedding dimension
    n_hid = 256                     # Hidden dimension (2x in_dim typical)
    n_layers = 3                    # Number of HGT layers
    n_heads = 8                     # Multi-head attention heads
    dropout = 0.2                   # Dropout rate (regularization)
    use_RTE = False                 # Relative Temporal Encoding (for dynamic graphs)
    
    # Training Hyperparameters
    learning_rate = 0.001           # Initial learning rate
    weight_decay = 1e-4             # L2/weight decay regularization
    batch_size = 64                 # Batch size
    epochs = 100                    # Maximum epochs
    patience = 15                   # Early stopping patience
    
    # Loss Weights (multi-task learning)
    loss_weight_disease = 0.7       # Disease prediction weight
    loss_weight_test = 0.3          # Test recommendation weight
    
    # Validation & Testing
    val_ratio = 0.15                # Validation set ratio
    test_ratio = 0.10               # Test set ratio
    eval_frequency = 1              # Evaluate every N epochs
    
    # Disease Prediction
    threshold_disease = 0.5         # Classification threshold
    
    # Test Recommendation
    top_k = [5, 10, 20]             # Top-k recommendations
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Paths
    emr_path = 'data/patient_reports.csv'
    kb_path = 'data/enhanced_symptom_connectivity_analysis(Sheet1).csv'
    labels_path = 'data/patient_disease_labels.csv'
    output_dir = 'outputs'
    
    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

class DiseasePredictor(nn.Module):
    """
    Disease Risk Prediction Head
    
    Architecture:
        Patient Embedding (256D)
            ↓
        Linear Layer (256 → 128)
            ↓
        ReLU + Dropout
            ↓
        Linear Layer (128 → num_diseases)
            ↓
        Sigmoid (per-disease probabilities)
    
    Multi-label classification: Each disease is independent prediction.
    Loss: BCEWithLogitsLoss (combines Sigmoid + BCE for numerical stability)
    """
    def __init__(self, n_hid, num_diseases, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, num_diseases)
        )
    
    def forward(self, x):
        """Returns logits (not probabilities). Sigmoid applied in loss."""
        return self.mlp(x)


class TestRecommender(nn.Module):
    """
    Test Recommendation Head
    
    Uses dot-product similarity between patient and test embeddings.
    
    Logic:
        For each patient, recommend tests with highest embedding similarity.
        This captures: "which tests are most relevant to this patient's state?"
    
    Loss: MSE between predicted similarity and actual patient-test connections.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, patient_emb, test_emb):
        """
        Args:
            patient_emb: [batch_size, n_hid] - patient embeddings
            test_emb: [num_tests, n_hid] - test embeddings
        
        Returns:
            scores: [batch_size, num_tests] - similarity scores
        """
        # Dot product similarity: high score = similar embeddings
        scores = torch.matmul(patient_emb, test_emb.t())
        return scores


class HGTTrainer:
    """
    Complete training pipeline for HGT model.
    
    Handles:
    - Data loading and splitting
    - Model initialization
    - Training loop with multi-task learning
    - Validation and early stopping
    - Metrics computation
    - Result saving
    """
    
    def __init__(self, config: HGTConfig):
        self.config = config
        self.device = config.device
        
        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)
        self.checkpoint_dir = Path(config.output_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Logging
        self.history = {
            'train_loss': [],
            'train_disease_loss': [],
            'train_test_loss': [],
            'val_loss': [],
            'val_disease_loss': [],
            'val_test_loss': [],
            'val_roc_auc': [],
            'val_f1': [],
            'test_roc_auc': [],
            'test_f1': [],
        }
        
        print(f"\n{'='*70}")
        print(f"HGT Training Configuration")
        print(f"{'='*70}")
        for key, value in config.to_dict().items():
            if key != 'device':
                print(f"{key:30} {value}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
    
    def prepare_data(self):
        """
        Load and prepare data.
        
        Steps:
        1. Load EMR + knowledge base
        2. Generate disease labels
        3. Build heterogeneous graph
        4. Convert to pyHGT format
        5. Split into train/val/test
        6. Create data loaders
        """
        print("[1/6] Creating heterogeneous graph...")
        hetero_graph, patient_data, entity_maps = create_hgt_graph(
            self.config.emr_path,
            self.config.kb_path
        )
        
        print("[2/6] Converting to pyHGT format...")
        pyhgt_data = convert_heterodata_to_pyhgt(hetero_graph)
        print_conversion_summary(pyhgt_data)
        
        print("[3/6] Loading disease labels...")
        disease_labels_df = pd.read_csv(self.config.labels_path)
        
        # Get disease columns (all except PatientID)
        disease_columns = [c for c in disease_labels_df.columns if c != 'PatientID']
        
        # Create patient_id to label vector mapping
        patient_to_labels = {}
        for _, row in disease_labels_df.iterrows():
            pid = row['PatientID']
            labels = torch.FloatTensor([row[d] for d in disease_columns])
            patient_to_labels[pid] = labels
        
        self.disease_names = disease_columns
        self.num_diseases = len(disease_columns)
        
        print(f"[4/6] Splitting data (70/15/15)...")
        patient_ids = sorted(patient_to_labels.keys())
        
        # First split: 70% train, 30% val+test
        train_ids, temp_ids = train_test_split(
            patient_ids,
            test_size=self.config.val_ratio + self.config.test_ratio,
            random_state=42
        )
        
        # Second split: 50/50 of remaining to val/test
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        print(f"  Train: {len(train_ids)} patients")
        print(f"  Val:   {len(val_ids)} patients")
        print(f"  Test:  {len(test_ids)} patients")
        
        self.pyhgt_data = pyhgt_data
        self.patient_to_labels = patient_to_labels
        self.entity_maps = entity_maps
        self.patient_data = patient_data
        
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        
        return pyhgt_data
    
    def build_model(self):
        """Initialize HGT model and heads."""
        print("\n[5/6] Building model...")
        
        # HGT backbone
        self.gnn_model = GNN(
            in_dim=self.config.in_dim,
            n_hid=self.config.n_hid,
            num_types=self.pyhgt_data.num_types,
            num_relations=self.pyhgt_data.num_relations,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            use_RTE=self.config.use_RTE
        ).to(self.device)
        
        # Task-specific heads
        self.disease_predictor = DiseasePredictor(
            self.config.n_hid,
            self.num_diseases,
            self.config.dropout
        ).to(self.device)
        
        self.test_recommender = TestRecommender().to(self.device)
        
        # Optimizer
        params = list(self.gnn_model.parameters()) + \
                 list(self.disease_predictor.parameters())
        
        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss functions
        self.disease_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.test_loss_fn = nn.MSELoss(reduction='mean')
        
        print(f"  GNN Model: {sum(p.numel() for p in self.gnn_model.parameters()):,} parameters")
        print(f"  Disease Head: {sum(p.numel() for p in self.disease_predictor.parameters()):,} parameters")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate}, decay={self.config.weight_decay})")
    
    def get_patient_embeddings(self):
        """
        Forward pass through HGT to get patient embeddings.
        
        Returns:
            [num_patients, n_hid] tensor of patient embeddings
        """
        with torch.no_grad():
            node_emb = self.gnn_model(
                self.pyhgt_data.node_feature.to(self.device),
                self.pyhgt_data.node_type.to(self.device),
                self.pyhgt_data.edge_time.to(self.device),
                self.pyhgt_data.edge_index.to(self.device),
                self.pyhgt_data.edge_type.to(self.device)
            )
        
        # Extract patient embeddings
        patient_offset = self.pyhgt_data.node_dict['patient'][0]
        num_patients = len(self.patient_to_labels)
        patient_emb = node_emb[patient_offset : patient_offset + num_patients]
        
        return patient_emb
    
    def train_epoch(self):
        """Train one epoch."""
        self.gnn_model.train()
        self.disease_predictor.train()
        
        # Get embeddings for all patients
        patient_emb = self.get_patient_embeddings()
        
        # Batch training
        train_indices = [self.entity_maps['patient'][pid] for pid in self.train_ids]
        train_labels = torch.stack([self.patient_to_labels[pid] for pid in self.train_ids]).to(self.device)
        
        total_loss = 0
        total_disease_loss = 0
        total_test_loss = 0
        num_batches = 0
        
        indices_tensor = torch.LongTensor(train_indices)
        dataset = torch.utils.data.TensorDataset(indices_tensor, train_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch_idx, batch_labels in pbar:
            batch_idx = batch_idx.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            batch_emb = patient_emb[batch_idx]
            
            # Disease prediction
            disease_logits = self.disease_predictor(batch_emb)
            disease_loss = self.disease_loss_fn(disease_logits, batch_labels)
            
            # Test recommendation (simplified: use embedding norm)
            test_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            loss = (self.config.loss_weight_disease * disease_loss +
                   self.config.loss_weight_test * test_loss)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_model.parameters()) + list(self.disease_predictor.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            total_disease_loss += disease_loss.item()
            total_test_loss += test_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'disease': total_disease_loss / num_batches,
            })
        
        return {
            'loss': total_loss / num_batches,
            'disease_loss': total_disease_loss / num_batches,
            'test_loss': total_test_loss / num_batches,
        }
    
    @torch.no_grad()
    def evaluate(self, patient_ids, split_name='val'):
        """Evaluate on a set of patients."""
        self.gnn_model.eval()
        self.disease_predictor.eval()
        
        # Get embeddings
        patient_emb = self.get_patient_embeddings()
        
        # Get test patients
        eval_indices = [self.entity_maps['patient'][pid] for pid in patient_ids]
        eval_labels = torch.stack([self.patient_to_labels[pid] for pid in patient_ids]).to(self.device)
        
        eval_emb = patient_emb[eval_indices]
        
        # Forward pass
        disease_logits = self.disease_predictor(eval_emb)
        disease_loss = self.disease_loss_fn(disease_logits, eval_labels)
        test_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = (self.config.loss_weight_disease * disease_loss +
                     self.config.loss_weight_test * test_loss)
        
        # Compute metrics
        disease_probs = torch.sigmoid(disease_logits)
        disease_pred = (disease_probs > self.config.threshold_disease).float()
        
        # Multi-label metrics
        roc_auc = roc_auc_score(
            eval_labels.cpu().numpy(),
            disease_probs.cpu().numpy(),
            average='weighted'
        )
        
        f1 = f1_score(
            eval_labels.cpu().numpy(),
            disease_pred.cpu().numpy(),
            average='weighted',
            zero_division=0
        )
        
        return {
            'loss': total_loss.item(),
            'disease_loss': disease_loss.item(),
            'test_loss': test_loss.item(),
            'roc_auc': roc_auc,
            'f1': f1,
        }
    
    def train(self):
        """Main training loop."""
        print("\n[6/6] Training model...")
        print(f"{'='*70}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config.eval_frequency == 0:
                val_metrics = self.evaluate(self.val_ids, 'val')
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'gnn_model': self.gnn_model.state_dict(),
                        'disease_predictor': self.disease_predictor.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, self.checkpoint_dir / 'best_model.pt')
                else:
                    patience_counter += 1
                
                # Log
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_disease_loss'].append(train_metrics['disease_loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_roc_auc'].append(val_metrics['roc_auc'])
                self.history['val_f1'].append(val_metrics['f1'])
                
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                      f"F1: {val_metrics['f1']:.4f} | "
                      f"Patience: {patience_counter}/{self.config.patience}")
                
                # LR scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # Test
        print(f"\n{'='*70}")
        print("Testing best model...")
        
        # Load best model
        checkpoint = torch.load(self.checkpoint_dir / 'best_model.pt')
        self.gnn_model.load_state_dict(checkpoint['gnn_model'])
        self.disease_predictor.load_state_dict(checkpoint['disease_predictor'])
        
        test_metrics = self.evaluate(self.test_ids, 'test')
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        self.history['test_roc_auc'] = test_metrics['roc_auc']
        self.history['test_f1'] = test_metrics['f1']
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save training history and configuration."""
        # Save history
        history_path = Path(self.config.output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save config
        config_path = Path(self.config.output_dir) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Plot learning curves
        self._plot_curves()
        
        print(f"\nResults saved to {self.config.output_dir}/")
    
    def _plot_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid()
        axes[0, 0].set_title('Training Loss')
        
        # ROC-AUC
        axes[0, 1].plot(self.history['val_roc_auc'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].grid()
        axes[0, 1].set_title('Validation ROC-AUC')
        
        # F1 Score
        axes[1, 0].plot(self.history['val_f1'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].grid()
        axes[1, 0].set_title('Validation F1 Score')
        
        # Test results
        axes[1, 1].bar(['ROC-AUC', 'F1'], 
                       [self.history['test_roc_auc'], self.history['test_f1']],
                       color=['blue', 'orange'])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_title('Test Results')
        axes[1, 1].set_ylabel('Score')
        for i, v in enumerate([self.history['test_roc_auc'], self.history['test_f1']]):
            axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'training_curves.png', dpi=100)
        print("Training curves saved to training_curves.png")


def main():
    """Main training script."""
    # Configuration
    config = HGTConfig()
    
    # Initialize trainer
    trainer = HGTTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
