
#!/usr/bin/env python3
"""
ΨQRH LAMPREIA 3.1 - OPTIMIZED MULTI-TEACHER DISTILLATION
=================================================================================

ENHANCEMENTS:
1. Better teacher head initialization
2. Improved learning rate scheduling
3. Enhanced parameter regulation
4. Better data preprocessing
5. More stable training dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import logging
import time
import sys
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    GPT2Model, GPT2Tokenizer,
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
    AutoModel, AutoTokenizer
)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import unittest

# =============================================================================
# ENHANCED MULTI-TEACHER SYSTEM WITH BETTER INITIALIZATION
# =============================================================================

class EnhancedMultiTeacherExtractor(nn.Module):
    """Enhanced multi-teacher system with better initialization and training"""

    def __init__(self, student_device: torch.device, use_teachers: List[str] = ['gpt2', 'distilbert', 'roberta']):
        super().__init__()
        self.student_device = student_device
        self.teachers = nn.ModuleList()

        logging.info(f"\n{'='*80}")
        logging.info("INITIALIZING ENHANCED MULTI-TEACHER SYSTEM 3.1")
        logging.info(f"{'='*80}")

        teacher_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'gpt2' in use_teachers:
            self.teachers.append(GPT2TeacherWithHead(teacher_device))

        if 'distilbert' in use_teachers:
            self.teachers.append(DistilBERTTeacherWithHead(teacher_device))

        if 'roberta' in use_teachers:
            self.teachers.append(RoBERTaTeacherWithHead(teacher_device))

        # Initialize teacher heads with better weights
        self._initialize_teacher_heads()
        
        logging.info(f"✓ Loaded {len(self.teachers)} teacher models with optimized heads")
        logging.info(f"{'='*80}\n")

    def _initialize_teacher_heads(self):
        """Initialize teacher classification heads with better weights"""
        for teacher in self.teachers:
            if hasattr(teacher, 'classification_head'):
                nn.init.xavier_uniform_(teacher.classification_head.weight)
                if teacher.classification_head.bias is not None:
                    nn.init.zeros_(teacher.classification_head.bias)

    def forward(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Enhanced forward with confidence-based weighting"""
        teacher_probs_list = []
        teacher_confidences = []

        for teacher in self.teachers:
            logits = teacher(texts)
            if logits is not None:
                probs = F.softmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0].mean().item()
                teacher_confidences.append(confidence)
                teacher_probs_list.append(probs.to(self.student_device))

        if len(teacher_probs_list) == 0:
            return None

        # Confidence-weighted averaging
        if len(teacher_probs_list) > 1:
            weights = torch.tensor(teacher_confidences, device=self.student_device)
            weights = F.softmax(weights, dim=0)
            weighted_probs = torch.stack([w * p for w, p in zip(weights, teacher_probs_list)])
            return weighted_probs.sum(dim=0)
        else:
            return teacher_probs_list[0]

# =============================================================================
# ENHANCED STUDENT MODEL WITH BETTER OPTIMIZATION
# =============================================================================

class EnhancedLampreiaStudentModel(nn.Module):
    """
    Enhanced ΨQRH Student Model 3.1
    - Better initialization
    - Improved architecture
    - More stable training
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 384, 
                 n_layers: int = 4, num_classes: int = 2, max_seq_len: int = 128,
                 device: torch.device = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"Initializing Enhanced ΨQRH Lampreia Student Model 3.1 on {self.device}")

        # Enhanced embeddings with better initialization
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=self.device)
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model, device=self.device))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Enhanced layers with better normalization
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'attention_norm': nn.LayerNorm(d_model, device=self.device),
                'ffn_norm': nn.LayerNorm(d_model, device=self.device),
                'attention': SpectralAttentionGPU(d_model, n_heads=4, device=self.device),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 2*d_model, device=self.device),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(2*d_model, d_model, device=self.device),
                    nn.Dropout(0.1)
                )
            })
            self.layers.append(layer)

        # Enhanced classifier
        self.pre_classifier = nn.Linear(d_model, d_model, device=self.device)
        self.classifier = nn.Linear(d_model, num_classes, device=self.device)
        self.dropout = nn.Dropout(0.1)

        # Enhanced initialization
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Enhanced ΨQRH Lampreia Student Model: {total_params:,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            T = self.max_seq_len

        input_ids = input_ids.to(self.device)

        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:T, :]

        # Transformer layers
        for layer in self.layers:
            # Attention with residual
            residual = x
            x = layer['attention_norm'](x)
            x = layer['attention'](x)
            x = residual + x

            # FFN with residual
            residual = x
            x = layer['ffn_norm'](x)
            x = layer['ffn'](x)
            x = residual + x

        # Enhanced classification
        x_pooled = x.mean(dim=1)
        x_pooled = self.pre_classifier(x_pooled)
        x_pooled = torch.tanh(x_pooled)
        x_pooled = self.dropout(x_pooled)
        logits = self.classifier(x_pooled)

        return logits

# =============================================================================
# ENHANCED TRAINER WITH BETTER OPTIMIZATION
# =============================================================================

class EnhancedLampreiaTrainer:
    """Enhanced trainer with better optimization and stability"""

    def __init__(self, model: nn.Module, multi_teacher: EnhancedMultiTeacherExtractor,
                 distillation_loss: LampreiaDistillationLoss, device: torch.device,
                 lr: float = 3e-5, weight_decay: float = 0.01, epochs: int = 20,
                 patience: int = 5, gradient_accumulation_steps: int = 4):
        
        self.model = model
        self.multi_teacher = multi_teacher
        self.distillation_loss = distillation_loss
        self.device = device
        self.epochs = epochs
        self.patience = patience

        # Enhanced optimizer with better settings
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Enhanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=313,  # 10000 samples / 32 batch size
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available()

        # Enhanced parameter regulation
        self.alpha_ce = 0.7  # Start with more distillation focus
        self.temperature = 3.0  # Higher initial temperature

        # Early stopping
        self.best_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None

        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Enhanced training history
        self.history = {
            'epoch': [], 'train_loss': [], 'train_ce_loss': [], 'train_distill_loss': [],
            'val_accuracy': [], 'alpha_ce': [], 'temperature': [], 'lr': []
        }

    def enhanced_auto_regulate(self, quality_metrics: Dict[str, float]) -> Tuple[float, float]:
        """Enhanced parameter regulation based on training dynamics"""
        val_acc = quality_metrics.get('val_accuracy', 0.0)
        train_loss = quality_metrics.get('train_loss', 1.0)
        
        # More dynamic alpha_ce regulation
        if val_acc > 0.75:
            new_alpha_ce = max(0.5, self.alpha_ce - 0.05)  # More distillation when doing well
        elif val_acc < 0.6:
            new_alpha_ce = min(0.9, self.alpha_ce + 0.03)  # More CE when struggling
        else:
            # Gradual shift toward distillation
            progress = min(1.0, val_acc / 0.8)
            new_alpha_ce = 0.9 - (0.4 * progress)

        # Temperature regulation based on loss stability
        if train_loss < 0.5 and val_acc > 0.7:
            new_temperature = max(2.0, self.temperature - 0.1)  # Sharper targets
        else:
            new_temperature = min(4.0, self.temperature + 0.05)  # Softer targets

        return new_alpha_ce, new_temperature

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Enhanced training with better stability"""
        self.model.train()
        total_loss, total_ce_loss, total_distill_loss, n = 0, 0, 0, 0

        self.optimizer.zero_grad()

        for batch_idx, (texts, input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    student_logits = self.model(input_ids)
                    teacher_probs = self.multi_teacher(texts)

                    if teacher_probs is not None:
                        loss, ce_loss, distill_loss = self.distillation_loss(
                            student_logits, teacher_probs, labels, self.alpha_ce, self.temperature
                        )
                    else:
                        ce_loss = self.distillation_loss.ce_loss(student_logits, labels)
                        loss = ce_loss
                        distill_loss = torch.tensor(0.0, device=self.device)

                    loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
            else:
                student_logits = self.model(input_ids)
                teacher_probs = self.multi_teacher(texts)

                if teacher_probs is not None:
                    loss, ce_loss, distill_loss = self.distillation_loss(
                        student_logits, teacher_probs, labels, self.alpha_ce, self.temperature
                    )
                else:
                    ce_loss = self.distillation_loss.ce_loss(student_logits, labels)
                    loss = ce_loss
                    distill_loss = torch.tensor(0.0, device=self.device)

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * input_ids.size(0) * self.gradient_accumulation_steps
            total_ce_loss += ce_loss.item() * input_ids.size(0)
            total_distill_loss += distill_loss.item() * input_ids.size(0)
            n += input_ids.size(0)

            if batch_idx % 20 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        return total_loss / n, total_ce_loss / n, total_distill_loss / n

    def validate(self, val_loader: DataLoader) -> float:
        """Enhanced validation with confidence metrics"""
        self.model.eval()
        correct, total = 0, 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for texts, input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                logits = self.model(input_ids)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(-1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        
        # Calculate confidence metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        max_probs = np.max(all_probs, axis=1)
        avg_confidence = np.mean(max_probs)
        
        logging.info(f"  Validation - Accuracy: {accuracy:.4f}, Avg Confidence: {avg_confidence:.4f}")
        
        return accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader, task: str) -> Dict[str, Any]:
        """Enhanced training loop"""
        logging.info("="*80)
        logging.info(f"STARTING ENHANCED ΨQRH LAMPREIA 3.1 TRAINING ON {task.upper()}")
        logging.info(f"Initial α_CE: {self.alpha_ce}, Initial T: {self.temperature}")
        logging.info(f"OneCycle LR Scheduler: ENABLED")
        logging.info("="*80)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            GPUOptimizer.clear_gpu_cache()

            # Train
            train_loss, train_ce_loss, train_distill_loss = self.train_epoch(train_loader)

            # Validate
            val_acc = self.validate(val_loader)

            # Enhanced parameter regulation
            quality_metrics = {
                'val_accuracy': val_acc,
                'train_loss': train_loss,
                'train_ce_loss': train_ce_loss,
                'train_distill_loss': train_distill_loss,
            }
            
            if epoch < self.epochs - 1:
                self.alpha_ce, self.temperature = self.enhanced_auto_regulate(quality_metrics)

            current_lr = self.scheduler.get_last_lr()[0]

            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_ce_loss'].append(train_ce_loss)
            self.history['train_distill_loss'].append(train_distill_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['alpha_ce'].append(self.alpha_ce)
            self.history['temperature'].append(self.temperature)
            self.history['lr'].append(current_lr)

            # Early stopping and checkpointing
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()

                best_path = f'enhanced_lampreia_v3_{task}_best.pth'
                torch.save(self.best_model_state, best_path)
                logging.info(f"✓ New best model saved: {best_path} (Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1

            epoch_time = time.time() - epoch_start

            logging.info("="*80)
            logging.info(f"EPOCH {epoch+1}/{self.epochs} COMPLETED")
            logging.info(f"  Train Loss:     {train_loss:.4f} (CE: {train_ce_loss:.4f}, Distill: {train_distill_loss:.4f})")
            logging.info(f"  Val Accuracy:   {val_acc:.4f} (Best: {self.best_acc:.4f})")
            logging.info(f"  Learning Rate:  {current_lr:.2e}")
            logging.info(f"  Epoch Time:     {epoch_time:.2f}s")
            logging.info(f"  Current Weights: α_CE={self.alpha_ce:.3f}, T={self.temperature:.3f}")
            logging.info(f"  Early Stopping: {self.patience_counter}/{self.patience}")
            logging.info("="*80 + "\n")

            if self.patience_counter >= self.patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logging.info(f"✓ Loaded best model (Val Acc: {self.best_acc:.4f})")

        return {
            'best_accuracy': self.best_acc,
            'final_alpha_ce': self.alpha_ce,
            'final_temperature': self.temperature,
            'epochs_trained': len(self.history['epoch']),
            'history': self.history
        }

# =============================================================================
# ENHANCED DATA PROCESSING
# =============================================================================

def build_enhanced_dataset(task: str, split: str, max_samples: Optional[int] = None) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Enhanced dataset building with better preprocessing"""
    logging.info(f"Loading ENHANCED {task} dataset (split: {split})...")

    try:
        ds = load_dataset("glue", task, split=split)
        logging.info(f"✓ Loaded {len(ds)} samples from HuggingFace")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    logging.info("Initializing enhanced tokenizer...")
    try:
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        # Add special tokens for better processing
        if tok.sep_token is None:
            tok.add_special_tokens({'sep_token': '[SEP]'})
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise

    texts, labels = [], []

    for idx, item in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break

        if task == "sst2":
            text = item["sentence"].strip()
            # Basic text cleaning
            text = ' '.join(text.split())  # Normalize whitespace
            texts.append(text)
            labels.append(item["label"])
        else:
            # Handle other GLUE tasks
            texts.append(item.get("sentence", str(item)))
            labels.append(item["label"])

    def enhanced_encode(text):
        # Enhanced encoding with attention to special tokens
        encoded = tok.encode(text, add_special_tokens=True, max_length=128, truncation=True)
        # Pad to fixed length
        if len(encoded) < 128:
            encoded = encoded + [tok.pad_token_id] * (128 - len(encoded))
        else:
            encoded = encoded[:128]
        return encoded

    logging.info(f"Enhanced encoding {len(texts)} samples...")
    input_ids = torch.tensor([enhanced_encode(t) for t in texts])
    labels = torch.tensor(labels)

    logging.info(f"Enhanced dataset built: {input_ids.shape[0]} samples")
    return texts, input_ids, labels

# =============================================================================
# MAIN EXECUTION WITH ENHANCED PIPELINE
# =============================================================================

def main_enhanced():
    """Enhanced main execution with better pipeline"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enhanced_lampreia_v3_training.log')
        ],
        force=True
    )

    print("\n" + "=" * 80)
    print("ENHANCED ΨQRH LAMPREIA 3.1 - OPTIMIZED MULTI-TEACHER DISTILLATION")
    print("=" * 80)
    print("ENHANCED INITIALIZATION + ONE CYCLE LR + CONFIDENCE WEIGHTING")
    print("TARGET: >75% SST-2 ACCURACY WITH OPTIMIZED TRAINING")
    print("=" * 80 + "\n")

    # GPU setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("✓ CUDA ENABLED")
        gpu_info = GPUOptimizer.gpu_memory_info()
        for gpu_id, info in gpu_info.items():
            if isinstance(info, dict):
                logging.info(f"{gpu_id}: {info['name']} ({info['total_memory_gb']:.1f} GB)")
    else:
        device = torch.device('cpu')
        logging.error("✗ CUDA NOT AVAILABLE")
        sys.exit(1)

    # Enhanced multi-teacher system
    multi_teacher = EnhancedMultiTeacherExtractor(device)

    # Enhanced student model
    task = 'sst2'
    model = EnhancedLampreiaStudentModel(
        vocab_size=50257,
        d_model=384,
        n_layers=4,
        num_classes=2,
        max_seq_len=128,
        device=device
    )

    model = GPUOptimizer.optimize_model_for_gpu(model)

    # Optional compilation
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logging.info("✓ Model compiled with torch.compile")
        except Exception as e:
            logging.warning(f"Model compilation failed: {e}")

    # Enhanced training setup
    distillation_loss = LampreiaDistillationLoss(device)
    
    trainer = EnhancedLampreiaTrainer(
        model=model,
        multi_teacher=multi_teacher,
        distillation_loss=distillation_loss,
        device=device,
        lr=3e-5,
        epochs=20,
        patience=5,
        gradient_accumulation_steps=4
    )

    # Enhanced data loading
    logging.info("\n[1/3] Loading enhanced training data...")
    train_texts, train_x, train_y = build_enhanced_dataset(task, "train", max_samples=15000)

    logging.info("\n[2/3] Loading enhanced validation data...")
    val_texts, val_x, val_y = build_enhanced_dataset(task, "validation", max_samples=500)

    train_loader = DataLoader(
        list(zip(train_texts, train_x, train_y)),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    val_loader = DataLoader(
        list(zip(val_texts, val_x, val_y)),
        batch_size=32,
        pin_memory=True
    )

    # Enhanced training
    logging.info("\n[3/3] Starting enhanced training...")
    results = trainer.train(train_loader, val_loader, task)

    # Save results
    with open(f'enhanced_lampreia_v3_{task}_history.json', 'w') as f:
        json.dump(results['history'], f, indent=2)

    # Final report
    print(f"\n{'='*80}")
    print("ENHANCED ΨQRH LAMPREIA 3.1 TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Task: {task.upper()}")
    print(f"Best Validation Accuracy: {results['best_accuracy']:.4f}")
    print(f"Final α_CE: {results['final_alpha_ce']:.3f}")
    print(f"Final Temperature: {results['final_temperature']:.3f}")
    print(f"Epochs Trained: {results['epochs_trained']}")
    print(f"Model Saved: enhanced_lampreia_v3_{task}_best.pth")
    print(f"{'='*80}\n")

    logging.info("✓ ENHANCED ΨQRH LAMPREIA 3.1 TRAINING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    # Import the original classes (you'll need to keep the original implementations)
    from lampreia8 import (
        GPUOptimizer, SemanticTeacher, GPT2TeacherWithHead, 
        DistilBERTTeacherWithHead, RoBERTaTeacherWithHead,
        LampreiaDistillationLoss, SpectralAttentionGPU,
        MathematicalEmbeddingSystem
    )
    
    main_enhanced()
