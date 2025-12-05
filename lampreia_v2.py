#!/usr/bin/env python3
"""
LAMPREIA v3 - With Teacher Classifier Pre-training
===================================================

Key improvement: Pre-train teacher classification heads on SST-2 before
distillation. This ensures teachers provide meaningful soft targets.

Flow:
1. Load teachers (frozen backbones + random classifiers)
2. Pre-train classifiers on SST-2 (2-3 epochs)
3. Freeze classifiers OR use very low LR
4. Distill to student

Usage:
    python lampreia_v3.py --params 10.5 --teachers distilbert
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
import argparse
import gc
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from datasets import load_dataset
    from transformers import (GPT2Model, GPT2Tokenizer, DistilBertModel, 
                              DistilBertTokenizer, RobertaModel, RobertaTokenizer)
    HAS_HF = True
except ImportError:
    HAS_HF = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    
    @classmethod
    def from_param_count(cls, target_millions: float) -> 'ModelConfig':
        configs = {
            8:    (256, 6, 4, 1024, 0.1),
            10.5: (288, 6, 6, 1152, 0.1),
            15:   (320, 8, 8, 1280, 0.1),
        }
        closest = min(configs.keys(), key=lambda x: abs(x - target_millions))
        d_model, n_layers, n_heads, d_ff, dropout = configs[closest]
        return cls(d_model=d_model, n_layers=n_layers, n_heads=n_heads, 
                   d_ff=d_ff, dropout=dropout)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# STUDENT MODEL
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StudentModel(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int = 50257, 
                 max_seq_len: int = 128, num_classes: int = 2):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, config.d_model))
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_classes)
        )
        
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Student Model: {total_params:,} params ({total_params/1e6:.2f}M)")
        
    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            T = self.max_seq_len
        
        x = self.token_emb(input_ids) + self.pos_emb[:, :T, :]
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        return self.classifier(x.mean(dim=1))


# =============================================================================
# TEACHERS
# =============================================================================

class Teacher(nn.Module):
    """
    Teacher with frozen backbone and trainable classifier.
    Stays on GPU during pre-training, then can be offloaded.
    """
    
    def __init__(self, teacher_type: str, device: torch.device):
        super().__init__()
        self.teacher_type = teacher_type
        self.device = device
        self._on_gpu = False
        
        logging.info(f"Loading {teacher_type} teacher...")
        
        if teacher_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backbone = GPT2Model.from_pretrained('gpt2')
            self.hidden_size = 768
            self.pooling = 'mean'
            
        elif teacher_type == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.hidden_size = 768
            self.pooling = 'cls'
            
        elif teacher_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.backbone = RobertaModel.from_pretrained('roberta-base')
            self.hidden_size = 768
            self.pooling = 'cls'
        else:
            raise ValueError(f"Unknown teacher: {teacher_type}")
        
        # Freeze backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Trainable classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 2)
        )
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        params = sum(p.numel() for p in self.backbone.parameters())
        clf_params = sum(p.numel() for p in self.classifier.parameters())
        logging.info(f"  {teacher_type}: backbone={params:,}, classifier={clf_params:,}")
    
    def to_gpu(self):
        if not self._on_gpu:
            self.backbone = self.backbone.to(self.device)
            self.classifier = self.classifier.to(self.device)
            self._on_gpu = True
        return self
    
    def to_cpu(self):
        if self._on_gpu:
            self.backbone = self.backbone.cpu()
            self.classifier = self.classifier.cpu()
            self._on_gpu = False
            clear_memory()
        return self
    
    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def _get_pooled(self, texts: List[str]) -> torch.Tensor:
        """Get pooled representations from backbone (no grad for backbone)"""
        inputs = self._tokenize(texts)
        
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        if self.pooling == 'mean':
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:  # cls
            pooled = outputs.last_hidden_state[:, 0, :]
        
        return pooled
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass - returns logits"""
        pooled = self._get_pooled(texts)
        return self.classifier(pooled)
    
    @torch.no_grad()
    def forward_no_grad(self, texts: List[str]) -> torch.Tensor:
        """Forward without gradients (for distillation)"""
        pooled = self._get_pooled(texts)
        return self.classifier(pooled)


class TeacherEnsemble:
    """Manages multiple teachers with pre-training capability"""
    
    def __init__(self, device: torch.device, teacher_names: List[str] = None):
        self.device = device
        teacher_names = teacher_names or ['distilbert']
        
        logging.info("="*60)
        logging.info("INITIALIZING TEACHER ENSEMBLE")
        logging.info("="*60)
        
        self.teachers: Dict[str, Teacher] = {}
        for name in teacher_names:
            self.teachers[name] = Teacher(name, device)
        
        logging.info(f"Teachers: {list(self.teachers.keys())}")
        logging.info("="*60)
    
    def get_classifier_params(self) -> List[nn.Parameter]:
        """Get all classifier parameters for optimization"""
        params = []
        for teacher in self.teachers.values():
            params.extend(teacher.classifier.parameters())
        return params
    
    def pretrain_classifiers(self, train_loader: DataLoader, 
                            epochs: int = 2, lr: float = 1e-4) -> Dict[str, float]:
        """
        Pre-train teacher classifiers on the task.
        
        This is crucial: without this, teachers give random soft targets!
        """
        logging.info("="*60)
        logging.info(f"PRE-TRAINING TEACHER CLASSIFIERS ({epochs} epochs)")
        logging.info("="*60)
        
        results = {}
        
        for name, teacher in self.teachers.items():
            logging.info(f"\nPre-training {name}...")
            
            # Move to GPU
            teacher.to_gpu()
            teacher.classifier.train()
            
            optimizer = optim.AdamW(
                teacher.classifier.parameters(), 
                lr=lr, 
                weight_decay=0.01
            )
            
            best_acc = 0.0
            
            for epoch in range(epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (texts, input_ids, labels) in enumerate(train_loader):
                    labels = labels.to(self.device)
                    
                    # Forward (backbone frozen, classifier trainable)
                    logits = teacher.forward(texts)
                    loss = F.cross_entropy(logits, labels)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(teacher.classifier.parameters(), 1.0)
                    optimizer.step()
                    
                    # Stats
                    total_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    if batch_idx % 200 == 0:
                        logging.info(f"  [{name}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
                acc = correct / total
                avg_loss = total_loss / total
                
                if acc > best_acc:
                    best_acc = acc
                
                logging.info(f"  [{name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
            
            results[name] = best_acc
            
            # Set classifier to eval mode
            teacher.classifier.eval()
            
            # Offload to CPU
            teacher.to_cpu()
            
            logging.info(f"  [{name}] Pre-training complete. Best acc: {best_acc:.4f}")
        
        logging.info("="*60)
        logging.info("TEACHER PRE-TRAINING COMPLETE")
        for name, acc in results.items():
            logging.info(f"  {name}: {acc:.4f}")
        logging.info("="*60)
        
        return results
    
    @torch.no_grad()
    def get_soft_targets(self, texts: List[str], temperature: float = 4.0) -> Optional[torch.Tensor]:
        """
        Get soft targets from teacher ensemble.
        
        Returns: softmax(logits / T) averaged across teachers
        """
        all_soft = []
        
        for name, teacher in self.teachers.items():
            teacher.to_gpu()
            teacher.classifier.eval()
            
            try:
                logits = teacher.forward_no_grad(texts)
                soft = F.softmax(logits / temperature, dim=-1)
                all_soft.append(soft.to(self.device))
            except Exception as e:
                logging.warning(f"Teacher {name} failed: {e}")
            finally:
                teacher.to_cpu()
        
        if not all_soft:
            return None
        
        if len(all_soft) == 1:
            return all_soft[0]
        
        # Average soft targets
        return torch.stack(all_soft, dim=0).mean(dim=0)


# =============================================================================
# DISTILLATION LOSS
# =============================================================================

class DistillationLoss(nn.Module):
    """
    L = α * CE(student, labels) + (1-α) * T² * KL(student_soft || teacher_soft)
    """
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_soft: torch.Tensor,
                labels: torch.Tensor, 
                alpha: float, 
                temperature: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        ce_loss = self.ce(student_logits, labels)
        
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (temperature ** 2)
        
        total = alpha * ce_loss + (1 - alpha) * kl_loss
        
        return total, ce_loss, kl_loss


# =============================================================================
# DATASET
# =============================================================================

class SST2Dataset(Dataset):
    def __init__(self, texts, input_ids, labels):
        self.texts = texts
        self.input_ids = input_ids
        self.labels = labels
        
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, idx): 
        return self.texts[idx], self.input_ids[idx], self.labels[idx]


def load_sst2_data(split: str, max_samples: Optional[int] = None, max_seq_len: int = 128):
    logging.info(f"Loading SST-2 {split}...")
    ds = load_dataset("glue", "sst2", split=split)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    texts, labels = [], []
    for idx, item in enumerate(ds):
        if max_samples and idx >= max_samples:
            break
        texts.append(' '.join(item['sentence'].split()))
        labels.append(item['label'])
    
    def encode(text):
        ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_len, truncation=True)
        if len(ids) < max_seq_len:
            ids = ids + [tokenizer.pad_token_id] * (max_seq_len - len(ids))
        return ids[:max_seq_len]
    
    input_ids = torch.tensor([encode(t) for t in texts])
    labels_tensor = torch.tensor(labels)
    logging.info(f"  Loaded {len(texts)} samples")
    return texts, input_ids, labels_tensor


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model: StudentModel, teachers: TeacherEnsemble, 
                 device: torch.device, lr: float = 3e-5, epochs: int = 20, 
                 patience: int = 5, grad_accum: int = 8):
        self.model = model
        self.teachers = teachers
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.grad_accum = grad_accum
        
        self.loss_fn = DistillationLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        self.scheduler = None
        self.base_lr = lr
        
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        self.alpha = 0.5
        self.temperature = 4.0
        
        self.best_acc = 0.0
        self.patience_counter = 0
        self.best_state = None
        
        self.history = {
            'epoch': [], 'train_loss': [], 'train_ce': [], 'train_kl': [],
            'val_acc': [], 'alpha': [], 'temperature': []
        }
    
    def setup_scheduler(self, steps_per_epoch: int):
        total_steps = steps_per_epoch * self.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.base_lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos'
        )
        logging.info(f"Scheduler: {total_steps} total steps")
    
    def adjust_params(self, epoch: int, val_acc: float):
        progress = epoch / self.epochs
        
        # Temperature: 4.0 -> 2.0
        self.temperature = 4.0 - 2.0 * progress
        self.temperature = max(2.0, self.temperature)
        
        # Alpha based on performance
        if val_acc > 0.82:
            self.alpha = 0.4
        elif val_acc > 0.75:
            self.alpha = 0.5
        else:
            self.alpha = 0.6
    
    def train_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss, total_ce, total_kl, n = 0.0, 0.0, 0.0, 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (texts, input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Get teacher soft targets
            teacher_soft = self.teachers.get_soft_targets(texts, self.temperature)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    student_logits = self.model(input_ids)
                    
                    if teacher_soft is not None:
                        loss, ce, kl = self.loss_fn(
                            student_logits, teacher_soft, labels,
                            self.alpha, self.temperature
                        )
                    else:
                        ce = F.cross_entropy(student_logits, labels)
                        loss = ce
                        kl = torch.tensor(0.0, device=self.device)
                    
                    loss_scaled = loss / self.grad_accum
                
                self.scaler.scale(loss_scaled).backward()
            else:
                student_logits = self.model(input_ids)
                
                if teacher_soft is not None:
                    loss, ce, kl = self.loss_fn(
                        student_logits, teacher_soft, labels,
                        self.alpha, self.temperature
                    )
                else:
                    ce = F.cross_entropy(student_logits, labels)
                    loss = ce
                    kl = torch.tensor(0.0, device=self.device)
                
                loss_scaled = loss / self.grad_accum
                loss_scaled.backward()
            
            if (batch_idx + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            total_ce += ce.item() * bs
            total_kl += kl.item() * bs
            n += bs
            
            if batch_idx % 100 == 0:
                logging.info(f"  Batch {batch_idx}/{len(loader)} | "
                           f"Loss: {loss.item():.4f} (CE: {ce.item():.4f}, KL: {kl.item():.4f})")
        
        return total_loss / n, total_ce / n, total_kl / n
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        
        for texts, input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(input_ids).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return correct / total
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        steps_per_epoch = len(train_loader) // self.grad_accum
        self.setup_scheduler(steps_per_epoch)
        
        logging.info("="*70)
        logging.info(f"DISTILLATION TRAINING: {self.epochs} epochs")
        logging.info(f"  Batch: {train_loader.batch_size} x {self.grad_accum} = {train_loader.batch_size * self.grad_accum}")
        logging.info(f"  Initial: α={self.alpha}, T={self.temperature}")
        logging.info("="*70)
        
        for epoch in range(self.epochs):
            t0 = time.time()
            clear_memory()
            
            train_loss, train_ce, train_kl = self.train_epoch(train_loader)
            val_acc = self.validate(val_loader)
            
            self.adjust_params(epoch, val_acc)
            
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_ce'].append(train_ce)
            self.history['train_kl'].append(train_kl)
            self.history['val_acc'].append(val_acc)
            self.history['alpha'].append(self.alpha)
            self.history['temperature'].append(self.temperature)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_state, 'best_model.pth')
                logging.info(f"  ★ New best: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            logging.info("="*70)
            logging.info(f"Epoch {epoch+1}/{self.epochs} | {time.time()-t0:.1f}s")
            logging.info(f"  Loss: {train_loss:.4f} (CE: {train_ce:.4f}, KL: {train_kl:.4f})")
            logging.info(f"  Val Acc: {val_acc:.4f} (Best: {self.best_acc:.4f})")
            logging.info(f"  α={self.alpha:.2f}, T={self.temperature:.2f} | Patience: {self.patience_counter}/{self.patience}")
            logging.info("="*70)
            
            if self.patience_counter >= self.patience:
                logging.info("Early stopping!")
                break
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        return {
            'best_accuracy': self.best_acc, 
            'epochs': len(self.history['epoch']), 
            'history': self.history
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=float, default=10.5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--teacher-pretrain-epochs', type=int, default=2)
    parser.add_argument('--teacher-pretrain-lr', type=float, default=1e-4)
    parser.add_argument('--train-samples', type=int, default=None)
    parser.add_argument('--val-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teachers', nargs='+', default=['distilbert'])
    parser.add_argument('--skip-teacher-pretrain', action='store_true')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('training.log')]
    )
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "="*70)
    print("LAMPREIA v3 - WITH TEACHER PRE-TRAINING")
    print(f"Target: ~{args.params}M params | Teachers: {args.teachers}")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        logging.info(f"  GPU: {props.name} ({props.total_memory/1e9:.1f} GB)")
    
    # Load data first (needed for teacher pre-training)
    train_texts, train_ids, train_labels = load_sst2_data('train', args.train_samples)
    val_texts, val_ids, val_labels = load_sst2_data('validation', args.val_samples)
    
    train_dataset = SST2Dataset(train_texts, train_ids, train_labels)
    val_dataset = SST2Dataset(val_texts, val_ids, val_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        num_workers=2, pin_memory=True
    )
    
    # Create teacher ensemble
    teachers = TeacherEnsemble(device, args.teachers)
    
    # PRE-TRAIN TEACHER CLASSIFIERS
    if not args.skip_teacher_pretrain:
        teacher_results = teachers.pretrain_classifiers(
            train_loader, 
            epochs=args.teacher_pretrain_epochs,
            lr=args.teacher_pretrain_lr
        )
    else:
        logging.info("Skipping teacher pre-training (--skip-teacher-pretrain)")
    
    # Create student model
    config = ModelConfig.from_param_count(args.params)
    logging.info(f"Config: d={config.d_model}, L={config.n_layers}, h={config.n_heads}, d_ff={config.d_ff}")
    
    model = StudentModel(config).to(device)
    
    # Train with distillation
    trainer = Trainer(
        model, teachers, device, 
        lr=args.lr, epochs=args.epochs, grad_accum=args.grad_accum
    )
    results = trainer.train(train_loader, val_loader)
    
    # Save
    with open('history.json', 'w') as f:
        json.dump(results['history'], f, indent=2)
    
    print("\n" + "="*70)
    print(f"DONE! Best Accuracy: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    if not HAS_HF:
        print("Install: pip install transformers datasets")
        sys.exit(1)
    main()
