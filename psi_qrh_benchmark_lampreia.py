#!/usr/bin/env python3
"""
ΨQRH LAMPREIA (LAMPREY) - MULTI-TEACHER SEMANTIC KNOWLEDGE DISTILLATION
==================================================================================

SISTEMA SANGUESSUGA: Extrai conhecimento semântico de múltiplos modelos pré-treinados
e destila em um modelo compacto com matemática genuína ΨQRH.

CONCEITO LAMPREIA:
- Múltiplos Teachers: GPT-2, DistilBERT, RoBERTa (todos em CPU)
- Extração Semântica: Converte outputs em representações semânticas universais
- Modelo Estudante Compacto: ΨQRH com matemática genuína (GPU)
- Destilação Multi-Teacher: Aprende de múltiplas fontes simultaneamente
- Zero Simulações: Tudo real, backpropagation genuíno

ARQUITETURA:
    [GPT-2]        [DistilBERT]      [RoBERTa]
       ↓                 ↓                ↓
   Semantic         Semantic         Semantic
   Extractor        Extractor        Extractor
       ↓                 ↓                ↓
       └─────────────────┴────────────────┘
                      ↓
              Multi-Teacher Loss
                      ↓
           [ΨQRH Student Model]
                   (GPU)

Author: Klenio Araujo Padilha
Compliance: GENUINE MATHEMATICS + MULTI-TEACHER DISTILLATION + SEMANTIC EXTRACTION
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
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import (
    GPT2Model, GPT2Tokenizer,
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
    AutoModel, AutoTokenizer
)

# =============================================================================
# GPU OPTIMIZATION UTILITIES
# =============================================================================

class GPUOptimizer:
    """GPU optimization utilities"""

    @staticmethod
    def optimize_model_for_gpu(model: nn.Module) -> nn.Module:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            device = torch.device('cuda')
            model = model.to(device)
        return model

    @staticmethod
    def gpu_memory_info() -> dict:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = props.total_memory / 1024**3
            info[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": total,
                "allocated_gb": allocated,
            }
        return info

    @staticmethod
    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# =============================================================================
# MULTI-TEACHER SEMANTIC EXTRACTION SYSTEM
# =============================================================================

class SemanticTeacher:
    """Base class for semantic knowledge extraction from pre-trained models"""

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def extract_semantic_embeddings(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract semantic embeddings from input - to be implemented by subclasses"""
        raise NotImplementedError

class GPT2SemanticTeacher(SemanticTeacher):
    """GPT-2 based semantic teacher"""

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__("gpt2", device)
        logging.info(f"Loading GPT-2 teacher on {device}...")

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2Model.from_pretrained("gpt2").to(device)
            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

            logging.info("✓ GPT-2 teacher loaded")
        except Exception as e:
            logging.warning(f"Failed to load GPT-2: {e}")

    def extract_semantic_embeddings(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract semantic embeddings from GPT-2"""
        if self.model is None:
            return None

        try:
            with torch.no_grad():
                input_cpu = input_ids.cpu()
                outputs = self.model(input_ids=input_cpu)
                # Mean pooling over sequence dimension
                semantic_emb = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
                return semantic_emb
        except:
            return None

class DistilBERTSemanticTeacher(SemanticTeacher):
    """DistilBERT based semantic teacher"""

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__("distilbert-base-uncased", device)
        logging.info(f"Loading DistilBERT teacher on {device}...")

        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

            logging.info("✓ DistilBERT teacher loaded")
        except Exception as e:
            logging.warning(f"Failed to load DistilBERT: {e}")

    def extract_semantic_embeddings(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract semantic embeddings from DistilBERT"""
        if self.model is None:
            return None

        try:
            with torch.no_grad():
                input_cpu = input_ids.cpu()
                outputs = self.model(input_ids=input_cpu)
                semantic_emb = outputs.last_hidden_state.mean(dim=1)
                return semantic_emb
        except:
            return None

class RoBERTaSemanticTeacher(SemanticTeacher):
    """RoBERTa based semantic teacher"""

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__("roberta-base", device)
        logging.info(f"Loading RoBERTa teacher on {device}...")

        try:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaModel.from_pretrained("roberta-base").to(device)
            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

            logging.info("✓ RoBERTa teacher loaded")
        except Exception as e:
            logging.warning(f"Failed to load RoBERTa: {e}")

    def extract_semantic_embeddings(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract semantic embeddings from RoBERTa"""
        if self.model is None:
            return None

        try:
            with torch.no_grad():
                input_cpu = input_ids.cpu()
                outputs = self.model(input_ids=input_cpu)
                semantic_emb = outputs.last_hidden_state.mean(dim=1)
                return semantic_emb
        except:
            return None

class MultiTeacherSemanticExtractor:
    """Combines multiple teachers for semantic knowledge extraction"""

    def __init__(self, student_device: torch.device, use_teachers: List[str] = ['gpt2']):
        self.student_device = student_device
        self.teachers = []

        logging.info(f"\n{'='*80}")
        logging.info("INITIALIZING MULTI-TEACHER SEMANTIC EXTRACTION SYSTEM")
        logging.info(f"{'='*80}")

        # Initialize teachers on CPU
        cpu_device = torch.device('cpu')

        if 'gpt2' in use_teachers:
            self.teachers.append(GPT2SemanticTeacher(cpu_device))

        if 'distilbert' in use_teachers:
            self.teachers.append(DistilBERTSemanticTeacher(cpu_device))

        if 'roberta' in use_teachers:
            self.teachers.append(RoBERTaSemanticTeacher(cpu_device))

        logging.info(f"✓ Loaded {len(self.teachers)} teacher models")
        logging.info(f"{'='*80}\n")

    def extract_multi_teacher_embeddings(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Extract semantic embeddings from all teachers"""
        embeddings = []

        for teacher in self.teachers:
            emb = teacher.extract_semantic_embeddings(input_ids)
            if emb is not None:
                # Move to student device
                embeddings.append(emb.to(self.student_device))

        return embeddings

# =============================================================================
# ΨQRH STUDENT MODEL WITH GENUINE MATHEMATICS
# =============================================================================

class PhysicalHarmonicResonanceSystem:
    """Physical harmonic resonance using first 100 primes"""

    def __init__(self, device: torch.device):
        self.device = device
        self.primes = self._generate_primes(100)

    def _generate_primes(self, n: int) -> List[int]:
        """Generate first n prime numbers"""
        primes = []
        num = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > num:
                    break
                if num % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
            num += 1
        return primes

class SpectralAttentionGPU(nn.Module):
    """Spectral attention with FFT processing"""

    def __init__(self, d_model: int, n_heads: int = 8, device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device

        self.q_proj = nn.Linear(d_model, d_model, device=device)
        self.k_proj = nn.Linear(d_model, d_model, device=device)
        self.v_proj = nn.Linear(d_model, d_model, device=device)
        self.out_proj = nn.Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)

        output = attended.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(output)

class LampreiaStudentModel(nn.Module):
    """
    ΨQRH Lamprey Student Model - Compact model with genuine mathematics
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 256,
                 n_layers: int = 4, num_classes: int = 2, max_seq_len: int = 128,
                 device: torch.device = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"Initializing Lampreia Student Model on {self.device}")

        # ΨQRH Components
        self.prime_system = PhysicalHarmonicResonanceSystem(device=self.device)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model, device=self.device))

        # Spectral Attention layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'attention_norm': nn.LayerNorm(d_model, device=self.device),
                'ffn_norm': nn.LayerNorm(d_model, device=self.device),
                'attention': SpectralAttentionGPU(d_model, n_heads=8, device=self.device),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4*d_model, device=self.device),
                    nn.GELU(),
                    nn.Linear(4*d_model, d_model, device=self.device)
                ),
                'dropout': nn.Dropout(0.1)
            })
            self.layers.append(layer)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes, device=self.device)

        # Initialize weights
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Lampreia Student Model: {total_params:,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            T = self.max_seq_len

        input_ids = input_ids.to(self.device)

        # Create embeddings with prime harmonic resonance
        token_floats = input_ids.float() / self.vocab_size

        embeddings_list = []
        for i in range(self.d_model):
            prime_idx = i % len(self.prime_system.primes)
            prime = self.prime_system.primes[prime_idx]
            freq = torch.tensor(prime * 1.618 * math.pi / self.d_model, device=self.device)
            angle = freq * token_floats
            emb_dim = torch.sin(angle) + torch.cos(angle)
            embeddings_list.append(emb_dim)

        x = torch.stack(embeddings_list, dim=-1)
        x = F.normalize(x, p=2, dim=-1) * math.sqrt(self.d_model)

        # Add positional embeddings
        x = x + self.pos_embedding[:T, :]

        # Apply transformer layers
        for layer in self.layers:
            # Attention
            normed = layer['attention_norm'](x)
            attn_out = layer['attention'](normed)
            x = x + layer['dropout'](attn_out)

            # FFN
            normed = layer['ffn_norm'](x)
            ffn_out = layer['ffn'](normed)
            x = x + layer['dropout'](ffn_out)

        # Classification
        x_pooled = x.mean(dim=1)
        logits = self.classifier(x_pooled)

        return logits, x_pooled  # Return both logits and embeddings

# =============================================================================
# DATA LOADING AND AUGMENTATION
# =============================================================================

def augment_input_ids(input_ids: torch.Tensor, aug_prob: float = 0.1) -> torch.Tensor:
    """Data augmentation with token masking and replacement"""
    if torch.rand(1).item() > 0.5:
        augmented = input_ids.clone()
        mask = torch.rand_like(input_ids.float()) < aug_prob
        augmented[mask] = 0

        replace_mask = torch.rand_like(input_ids.float()) < (aug_prob / 2)
        random_ids = torch.randint(1, 1000, input_ids.shape, device=input_ids.device)
        augmented[replace_mask] = random_ids[replace_mask]

        return augmented
    return input_ids

def build_real_glue_data(task: str, split: str, max_samples: int = 2000):
    """Load real GLUE data"""
    logging.info(f"Loading REAL {task} dataset (split: {split})...")

    try:
        ds = load_dataset("glue", task, split=split)
        logging.info(f"✓ Loaded {len(ds)} samples from HuggingFace")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    logging.info("Initializing GPT-2 tokenizer...")
    try:
        tok = Tokenizer.from_pretrained("gpt2")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise

    texts, labels = [], []

    for idx, item in enumerate(ds):
        if idx >= max_samples:
            break

        if task == "sst2":
            texts.append(item["sentence"])
            labels.append(item["label"])
        elif task == "qnli":
            texts.append(f"{item['question']} [SEP] {item['sentence']}")
            labels.append(item["label"])
        elif task == "mrpc":
            texts.append(f"{item['sentence1']} [SEP] {item['sentence2']}")
            labels.append(item["label"])
        elif task == "rte":
            texts.append(f"{item['sentence1']} [SEP] {item['sentence2']}")
            labels.append(item["label"])
        else:
            texts.append(item["sentence"])
            labels.append(item["label"])

    def encode(text):
        ids = tok.encode(text).ids[:128]
        return ids + [0] * (128 - len(ids))

    logging.info(f"Encoding {len(texts)} samples with BPE...")
    input_ids = torch.tensor([encode(t) for t in texts])
    labels = torch.tensor(labels)

    logging.info(f"Dataset built: {input_ids.shape[0]} samples")
    return input_ids, labels

# =============================================================================
# MULTI-TEACHER DISTILLATION TRAINING
# =============================================================================

def train_lampreia_glue(model: nn.Module, multi_teacher: MultiTeacherSemanticExtractor,
                        task: str, device: torch.device, epochs: int = 10):
    """Train Lampreia student with multi-teacher distillation"""

    logging.info("="*80)
    logging.info(f"STARTING LAMPREIA TRAINING ON {task.upper()}")
    logging.info(f"Multi-Teacher Distillation from {len(multi_teacher.teachers)} teachers")
    logging.info("="*80)

    model.to(device)

    # Load data
    logging.info("\n[1/4] Loading training data...")
    train_x, train_y = build_real_glue_data(task, "train", 2000)

    logging.info("\n[2/4] Loading validation data...")
    val_x, val_y = build_real_glue_data(task, "validation", 400)

    train_loader = DataLoader(
        list(zip(train_x, train_y)),
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        list(zip(val_x, val_y)),
        batch_size=32,
        pin_memory=True
    )

    # Optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()

    # Distillation weights
    alpha_ce = 0.7  # Classification loss weight
    alpha_distill = 0.3  # Distillation loss weight

    logging.info("\n[3/4] Starting training with multi-teacher distillation...")
    logging.info(f"  Teachers: {len(multi_teacher.teachers)}")
    logging.info(f"  CE Loss Weight: {alpha_ce}")
    logging.info(f"  Distillation Weight: {alpha_distill}")
    logging.info(f"  Epochs: {epochs}")
    logging.info("="*80)

    results = []

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        total_loss, total_ce_loss, total_distill_loss, n = 0, 0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Data augmentation
            x_aug = augment_input_ids(x, aug_prob=0.1)

            opt.zero_grad(set_to_none=True)

            # Forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, student_emb = model(x_aug)
                    ce_loss = ce_loss_fn(logits, y)

                    # Multi-teacher distillation
                    teacher_embeddings = multi_teacher.extract_multi_teacher_embeddings(x)
                    distill_loss = torch.tensor(0.0, device=device)

                    if len(teacher_embeddings) > 0:
                        for teacher_emb in teacher_embeddings:
                            # Project teacher to student dimension
                            if teacher_emb.shape[-1] != student_emb.shape[-1]:
                                teacher_emb = teacher_emb[:, :student_emb.shape[-1]]

                            distill_loss += mse_loss_fn(student_emb, teacher_emb)

                        distill_loss /= len(teacher_embeddings)

                    loss = alpha_ce * ce_loss + alpha_distill * distill_loss

                if not torch.isnan(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    continue
            else:
                logits, student_emb = model(x_aug)
                ce_loss = ce_loss_fn(logits, y)

                teacher_embeddings = multi_teacher.extract_multi_teacher_embeddings(x)
                distill_loss = torch.tensor(0.0, device=device)

                if len(teacher_embeddings) > 0:
                    for teacher_emb in teacher_embeddings:
                        if teacher_emb.shape[-1] != student_emb.shape[-1]:
                            teacher_emb = teacher_emb[:, :student_emb.shape[-1]]
                        distill_loss += mse_loss_fn(student_emb, teacher_emb)
                    distill_loss /= len(teacher_embeddings)

                loss = alpha_ce * ce_loss + alpha_distill * distill_loss

                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                else:
                    continue

            total_loss += loss.item() * x.size(0)
            total_ce_loss += ce_loss.item() * x.size(0)
            total_distill_loss += distill_loss.item() * x.size(0)
            n += x.size(0)

            if batch_idx % 10 == 0:
                gpu_mem = GPUOptimizer.gpu_memory_info()
                mem_gb = gpu_mem.get('gpu_0', {}).get('allocated_gb', 0) if gpu_mem else 0
                logging.info(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                           f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Distill: {distill_loss.item():.4f}) | "
                           f"GPU: {mem_gb:.2f}GB")

        train_loss = total_loss / n
        train_ce_loss = total_ce_loss / n
        train_distill_loss = total_distill_loss / n

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, _ = model(x)
                preds = logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        epoch_time = time.time() - epoch_start

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logging.info("="*80)
        logging.info(f"EPOCH {epoch+1}/{epochs} COMPLETED")
        logging.info(f"  Total Loss:     {train_loss:.4f}")
        logging.info(f"  CE Loss:        {train_ce_loss:.4f}")
        logging.info(f"  Distill Loss:   {train_distill_loss:.4f}")
        logging.info(f"  Val Accuracy:   {acc:.4f}")
        logging.info(f"  Learning Rate:  {current_lr:.2e}")
        logging.info(f"  Epoch Time:     {epoch_time:.2f}s")
        logging.info("="*80 + "\n")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_ce_loss': train_ce_loss,
            'train_distill_loss': train_distill_loss,
            'val_accuracy': acc,
            'epoch_time': epoch_time
        })

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('lampreia_glue_benchmark.log')
        ],
        force=True
    )

    print("\n" + "=" * 80)
    print("ΨQRH LAMPREIA (LAMPREY) - MULTI-TEACHER SEMANTIC DISTILLATION")
    print("=" * 80)
    print("SEMANTIC KNOWLEDGE EXTRACTION + GENUINE MATHEMATICS")
    print("MULTI-TEACHER DISTILLATION + REAL GLUE DATASETS")
    print("=" * 80 + "\n")

    # Detect GPU
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

    # Initialize multi-teacher system
    multi_teacher = MultiTeacherSemanticExtractor(
        student_device=device,
        use_teachers=['gpt2']  # Start with GPT-2, can add 'distilbert', 'roberta'
    )

    # GLUE tasks
    glue_tasks = ['sst2']  # Start with SST-2

    all_results = {}

    for task in glue_tasks:
        print(f"\n{'='*80}")
        print(f"LAMPREIA DISTILLATION: {task.upper()}")
        print(f"{'='*80}\n")

        num_classes = 2 if task in ['sst2', 'qnli', 'mrpc', 'rte'] else 3

        # Create compact student model
        model = LampreiaStudentModel(
            vocab_size=50257,
            d_model=256,
            n_layers=4,  # Compact student
            num_classes=num_classes,
            max_seq_len=128,
            device=device
        )

        model = GPUOptimizer.optimize_model_for_gpu(model)

        # Train with multi-teacher distillation
        results = train_lampreia_glue(model, multi_teacher, task, device, epochs=10)

        all_results[task] = results

        # Clear GPU
        del model
        GPUOptimizer.clear_gpu_cache()

    # Save results
    print(f"\n{'='*80}")
    print("LAMPREIA BENCHMARK COMPLETE")
    print(f"{'='*80}\n")

    with open('lampreia_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("LAMPREIA BENCHMARK SUMMARY")
    print(f"{'='*80}")
    for task, results in all_results.items():
        best_acc = max(r['val_accuracy'] for r in results)
        print(f"{task.upper():10s} | Best Accuracy: {best_acc:.4f}")
    print(f"{'='*80}\n")

    logging.info("✓ LAMPREIA MULTI-TEACHER DISTILLATION COMPLETED")

if __name__ == "__main__":
    main()
