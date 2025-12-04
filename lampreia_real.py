#!/usr/bin/env python3
"""
LAMPREIA - Multi-Teacher Knowledge Distillation (Corrected)
============================================================
Self-contained implementation for SST-2 sentiment classification.

Fixes applied:
1. All dependencies included (no external lampreia8 imports)
2. Scheduler calculates steps_per_epoch dynamically
3. Clear parameter regulation logic
4. Proper tokenizer handling
5. Flexible model size configuration

Usage:
    python lampreia_corrected.py --params 10.5
    python lampreia_corrected.py --params 8  
    python lampreia_corrected.py --params 15
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
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from datasets import load_dataset
    from transformers import (GPT2Model, GPT2Tokenizer, DistilBertModel, 
                              DistilBertTokenizer, RobertaModel, RobertaTokenizer)
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("Warning: Install with: pip install transformers datasets")


@dataclass
class ModelConfig:
    """Model configuration based on target parameter count"""
    d_model: int
    n_layers: int
    n_heads: int
    d_ff_mult: int
    dropout: float
    
    @classmethod
    def from_param_count(cls, target_millions: float) -> 'ModelConfig':
        # Configurations tuned for different parameter counts
        configs = {
            8: (320, 4, 4, 3, 0.1),
            10.5: (384, 4, 6, 3, 0.1),
            15: (448, 5, 8, 3, 0.1),
        }
        closest = min(configs.keys(), key=lambda x: abs(x - target_millions))
        d_model, n_layers, n_heads, d_ff_mult, dropout = configs[closest]
        return cls(d_model=d_model, n_layers=n_layers, n_heads=n_heads, 
                   d_ff_mult=d_ff_mult, dropout=dropout)


class GPUOptimizer:
    @staticmethod
    def get_device() -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def memory_info() -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {'device': 'cpu'}
        return {
            'device': torch.cuda.get_device_name(0),
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
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
    """Student transformer model for distillation"""
    
    def __init__(self, config: ModelConfig, vocab_size: int = 50257, 
                 max_seq_len: int = 128, num_classes: int = 2):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, config.d_model))
        self.emb_dropout = nn.Dropout(config.dropout)
        
        d_ff = config.d_model * config.d_ff_mult
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.Tanh(),
            nn.Dropout(config.dropout), nn.Linear(config.d_model, num_classes)
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
                if m.bias is not None: nn.init.zeros_(m.bias)
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


class BaseTeacher(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
    def _tokenize(self, texts: List[str], max_length: int = 128):
        enc = self.tokenizer(texts, padding=True, truncation=True, 
                             max_length=max_length, return_tensors='pt')
        return {k: v.to(self.device) for k, v in enc.items()}
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class GPT2Teacher(BaseTeacher):
    def __init__(self, device: torch.device):
        super().__init__(device)
        logging.info("Loading GPT-2 teacher...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained('gpt2').to(device).eval()
        for p in self.model.parameters(): p.requires_grad = False
        
        hs = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hs // 2, 2)
        ).to(device)
        self._init_classifier()
        logging.info(f"  GPT-2: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        out = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


class DistilBERTTeacher(BaseTeacher):
    def __init__(self, device: torch.device):
        super().__init__(device)
        logging.info("Loading DistilBERT teacher...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device).eval()
        for p in self.model.parameters(): p.requires_grad = False
        
        hs = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hs // 2, 2)
        ).to(device)
        self._init_classifier()
        logging.info(f"  DistilBERT: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        out = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return self.classifier(out.last_hidden_state[:, 0, :])


class RoBERTaTeacher(BaseTeacher):
    def __init__(self, device: torch.device):
        super().__init__(device)
        logging.info("Loading RoBERTa teacher...")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base').to(device).eval()
        for p in self.model.parameters(): p.requires_grad = False
        
        hs = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hs // 2, 2)
        ).to(device)
        self._init_classifier()
        logging.info(f"  RoBERTa: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        out = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return self.classifier(out.last_hidden_state[:, 0, :])


class TeacherEnsemble(nn.Module):
    """Ensemble of teacher models with confidence-based weighting"""
    
    def __init__(self, device: torch.device, teachers: List[str] = None):
        super().__init__()
        self.device = device
        teachers = teachers or ['gpt2', 'distilbert', 'roberta']
        
        logging.info("="*60)
        logging.info("INITIALIZING TEACHER ENSEMBLE")
        logging.info("="*60)
        
        self.teachers = nn.ModuleDict()
        if 'gpt2' in teachers: self.teachers['gpt2'] = GPT2Teacher(device)
        if 'distilbert' in teachers: self.teachers['distilbert'] = DistilBERTTeacher(device)
        if 'roberta' in teachers: self.teachers['roberta'] = RoBERTaTeacher(device)
        
        logging.info(f"Loaded {len(self.teachers)} teachers")
        logging.info("="*60)
    
    def get_trainable_params(self):
        params = []
        for t in self.teachers.values():
            params.extend(t.classifier.parameters())
        return params
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> Optional[torch.Tensor]:
        all_probs, all_conf = [], []
        
        for name, teacher in self.teachers.items():
            try:
                logits = teacher(texts)
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                conf = 1.0 / (1.0 + entropy.mean().item())
                all_probs.append(probs)
                all_conf.append(conf)
            except Exception as e:
                logging.warning(f"Teacher {name} failed: {e}")
        
        if not all_probs: return None
        if len(all_probs) == 1: return all_probs[0]
        
        weights = F.softmax(torch.tensor(all_conf, device=self.device), dim=0)
        return sum(w * p for w, p in zip(weights, all_probs))


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_probs, labels, alpha=0.5, temp=2.0):
        ce_loss = self.ce(student_logits, labels)
        
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        teacher_soft = torch.pow(teacher_probs, 1/temp)
        teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)
        
        kl_loss = self.kl(student_soft, teacher_soft) * (temp ** 2)
        total = alpha * ce_loss + (1 - alpha) * kl_loss
        
        return total, ce_loss, kl_loss


class SST2Dataset(Dataset):
    def __init__(self, texts, input_ids, labels):
        self.texts, self.input_ids, self.labels = texts, input_ids, labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx], self.input_ids[idx], self.labels[idx]


def load_sst2_data(split: str, max_samples: Optional[int] = None, max_seq_len: int = 128):
    """Load SST-2 dataset"""
    logging.info(f"Loading SST-2 {split}...")
    ds = load_dataset("glue", "sst2", split=split)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    texts, labels = [], []
    for idx, item in enumerate(ds):
        if max_samples and idx >= max_samples: break
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


class Trainer:
    """Trainer with dynamic scheduler and parameter regulation"""
    
    def __init__(self, model, teachers, device, lr=3e-5, epochs=20, patience=5, grad_accum=4):
        self.model = model
        self.teachers = teachers
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.grad_accum = grad_accum
        
        self.loss_fn = DistillationLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        teacher_params = list(teachers.get_trainable_params())
        self.teacher_opt = optim.AdamW(teacher_params, lr=lr * 0.1) if teacher_params else None
        
        self.scheduler = None
        self.base_lr = lr
        
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        self.alpha = 0.7
        self.temperature = 3.0
        self.best_acc = 0.0
        self.patience_counter = 0
        self.best_state = None
        
        self.history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'alpha': [], 'temp': []}
    
    def setup_scheduler(self, steps_per_epoch: int):
        """Configure scheduler with correct total steps"""
        total_steps = steps_per_epoch * self.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.base_lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos'
        )
        logging.info(f"Scheduler: {total_steps} total steps")
    
    def adjust_params(self, val_acc: float):
        """Adjust distillation parameters based on performance"""
        if val_acc > 0.80:
            self.alpha = max(0.4, self.alpha - 0.05)
            self.temperature = max(1.5, self.temperature - 0.2)
        elif val_acc > 0.70:
            self.alpha = 0.5 + 0.1 * (0.8 - val_acc) / 0.1
            self.temperature = max(2.0, self.temperature - 0.1)
        else:
            self.alpha = min(0.8, self.alpha + 0.03)
            self.temperature = min(4.0, self.temperature + 0.1)
    
    def train_epoch(self, loader):
        self.model.train()
        for t in self.teachers.teachers.values(): t.classifier.train()
        
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad()
        if self.teacher_opt: self.teacher_opt.zero_grad()
        
        for batch_idx, (texts, input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids)
                    teacher_probs = self.teachers(texts)
                    
                    if teacher_probs is not None:
                        loss, _, _ = self.loss_fn(logits, teacher_probs, labels, self.alpha, self.temperature)
                    else:
                        loss = F.cross_entropy(logits, labels)
                    loss = loss / self.grad_accum
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids)
                teacher_probs = self.teachers(texts)
                
                if teacher_probs is not None:
                    loss, _, _ = self.loss_fn(logits, teacher_probs, labels, self.alpha, self.temperature)
                else:
                    loss = F.cross_entropy(logits, labels)
                loss = loss / self.grad_accum
                loss.backward()
            
            if (batch_idx + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                if self.teacher_opt:
                    self.teacher_opt.step()
                    self.teacher_opt.zero_grad()
                if self.scheduler: self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * input_ids.size(0) * self.grad_accum
            n += input_ids.size(0)
            
            if batch_idx % 100 == 0:
                logging.info(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item()*self.grad_accum:.4f}")
        
        return total_loss / n
    
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        
        for texts, input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(input_ids).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return correct / total
    
    def train(self, train_loader, val_loader):
        # FIXED: Calculate steps_per_epoch dynamically
        steps_per_epoch = len(train_loader) // self.grad_accum
        self.setup_scheduler(steps_per_epoch)
        
        logging.info("="*70)
        logging.info(f"TRAINING: {self.epochs} epochs, batch={train_loader.batch_size}, accum={self.grad_accum}")
        logging.info(f"Steps per epoch: {steps_per_epoch}")
        logging.info("="*70)
        
        for epoch in range(self.epochs):
            t0 = time.time()
            GPUOptimizer.clear_cache()
            
            train_loss = self.train_epoch(train_loader)
            val_acc = self.validate(val_loader)
            self.adjust_params(val_acc)
            
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['alpha'].append(self.alpha)
            self.history['temp'].append(self.temperature)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_state, 'best_model.pth')
                logging.info(f"  * Best model saved: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            logging.info("="*70)
            logging.info(f"Epoch {epoch+1}/{self.epochs} | {time.time()-t0:.1f}s")
            logging.info(f"  Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} (Best: {self.best_acc:.4f})")
            logging.info(f"  Alpha: {self.alpha:.3f} | Temp: {self.temperature:.2f}")
            logging.info(f"  Early stop: {self.patience_counter}/{self.patience}")
            logging.info("="*70)
            
            if self.patience_counter >= self.patience:
                logging.info("Early stopping triggered!")
                break
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        return {'best_accuracy': self.best_acc, 'epochs': len(self.history['epoch']), 'history': self.history}


def main():
    parser = argparse.ArgumentParser(description='Multi-Teacher Knowledge Distillation')
    parser.add_argument('--params', type=float, default=10.5, help='Target params in millions')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--train-samples', type=int, default=None)
    parser.add_argument('--val-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teachers', nargs='+', default=['gpt2', 'distilbert', 'roberta'])
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('training.log')])
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "="*70)
    print("MULTI-TEACHER KNOWLEDGE DISTILLATION - SST-2")
    print(f"Target: ~{args.params}M parameters | Teachers: {args.teachers}")
    print("="*70 + "\n")
    
    device = GPUOptimizer.get_device()
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        info = GPUOptimizer.memory_info()
        logging.info(f"  GPU: {info['device']} ({info['total_gb']:.1f} GB)")
    
    config = ModelConfig.from_param_count(args.params)
    logging.info(f"Config: d={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    
    model = StudentModel(config).to(device)
    teachers = TeacherEnsemble(device, args.teachers)
    
    train_texts, train_ids, train_labels = load_sst2_data('train', args.train_samples)
    val_texts, val_ids, val_labels = load_sst2_data('validation', args.val_samples)
    
    train_loader = DataLoader(SST2Dataset(train_texts, train_ids, train_labels), 
                              batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(SST2Dataset(val_texts, val_ids, val_labels),
                            batch_size=args.batch_size, num_workers=2, pin_memory=True)
    
    trainer = Trainer(model, teachers, device, lr=args.lr, epochs=args.epochs)
    results = trainer.train(train_loader, val_loader)
    
    with open('history.json', 'w') as f:
        json.dump(results['history'], f, indent=2)
    
    print("\n" + "="*70)
    print(f"COMPLETED! Best Accuracy: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)")
    print(f"Model saved: best_model.pth | History: history.json")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    if not HAS_HF:
        print("Install dependencies: pip install transformers datasets")
        sys.exit(1)
    main()
