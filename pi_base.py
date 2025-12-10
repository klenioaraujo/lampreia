#!/usr/bin/env python3
"""
AIMO3 Micro-Model: Full PSIQRH with Enhanced PiBaseSystem
==========================================================

Modelo completo <1M parâmetros com PiBaseSystem totalmente integrado:
- PiAwareAttention: Bias posicional π-harmônico nos scores de atenção
- AdaptivePiPositional: Positional encoding com frequência adaptativa
- TokenWiseGate: Gate por dimensão (não escalar)
- CurriculumTrainer: Treinamento com dificuldade progressiva
- Problem Augmentation: Variações de problemas matemáticos

Complexidades mantidas em O(T² × d) máximo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import logging
import random
import re
from typing import Optional, Tuple, List, Dict
from collections import Counter
from dataclasses import dataclass
from enum import Enum

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import sympy as sp
    from sympy import symbols, Eq, solve, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from mpmath import mp, mpf, pi as MP_PI, power as mp_power
    mp.dps = 30
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

try:
    from einops import rearrange
except ImportError:
    def rearrange(x, pattern, **kwargs):
        if pattern == 'b t (h d) -> b h t d':
            b, t, _ = x.shape
            h = kwargs['h']
            return x.view(b, t, h, -1).transpose(1, 2)
        elif pattern == 'b h t d -> b t (h d)':
            b, h, t, d = x.shape
            return x.transpose(1, 2).contiguous().view(b, t, h * d)
        return x


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# =============================================================================
# 1. PIBASESYSTEM - NÚCLEO NUMÉRICO COMPLETO
# =============================================================================

class PiBaseSystem:
    """
    Sistema de aritmética em base-π.
    
    Usado para:
    1. Inicialização de embeddings (padrões π-harmônicos)
    2. Frequências de atenção (primos × π)
    3. Bias posicional adaptativo
    4. Pesos espectrais
    """
    
    _pi_powers: Dict[int, float] = {}
    _primes: List[int] = []
    _initialized: bool = False
    
    @classmethod
    def initialize(cls, max_power: int = 20, max_prime_count: int = 256):
        if cls._initialized:
            return
        
        # Cache de potências de π
        if MPMATH_AVAILABLE:
            for p in range(-max_power, max_power + 1):
                cls._pi_powers[p] = float(mp_power(MP_PI, p))
        else:
            for p in range(-max_power, max_power + 1):
                cls._pi_powers[p] = math.pi ** p
        
        # Cache de primos
        limit = max_prime_count * 15
        sieve = [True] * limit
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit, i):
                    sieve[j] = False
        cls._primes = [i for i in range(limit) if sieve[i]][:max_prime_count]
        
        cls._initialized = True
    
    @classmethod
    def to_pi_base(cls, value: float, precision: int = 10) -> List[Tuple[int, int]]:
        """Converte decimal para base-π."""
        cls.initialize()
        
        if value == 0:
            return [(0, 0)]
        
        sign = 1 if value >= 0 else -1
        remaining = abs(value)
        digits = []
        
        if remaining >= 1:
            max_pow = int(math.log(remaining) / math.log(math.pi)) + 1
        else:
            max_pow = 0
        
        for p in range(max_pow, -precision - 1, -1):
            pi_p = cls._pi_powers.get(p, math.pi ** p)
            if pi_p > remaining * 1.001:
                continue
            
            digit = min(int(remaining / pi_p), 3)
            if digit > 0:
                digits.append((p, digit * sign))
                remaining -= digit * pi_p
            
            if remaining < 1e-10:
                break
        
        return digits if digits else [(0, 0)]
    
    @classmethod
    def from_pi_base(cls, digits: List[Tuple[int, int]]) -> float:
        """Converte base-π para decimal."""
        cls.initialize()
        result = 0.0
        for pos, digit in digits:
            result += digit * cls._pi_powers.get(pos, math.pi ** pos)
        return result
    
    @classmethod
    def get_pi_prime_frequencies(cls, n: int) -> torch.Tensor:
        """Retorna n frequências = primo[i] × π."""
        cls.initialize()
        freqs = [cls._primes[i] * math.pi for i in range(min(n, len(cls._primes)))]
        while len(freqs) < n:
            freqs.append(freqs[-1] + math.pi)
        return torch.tensor(freqs, dtype=torch.float32)
    
    @classmethod
    def get_pi_attention_bias(cls, n_heads: int, max_seq_len: int) -> torch.Tensor:
        """
        Gera bias de atenção baseado em π-harmonics.
        Shape: [n_heads, max_seq_len, max_seq_len]
        """
        cls.initialize()
        
        freqs = cls.get_pi_prime_frequencies(n_heads)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        
        # Diferença de posições
        pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [T, T]
        
        # Bias π-harmônico por head
        bias = torch.zeros(n_heads, max_seq_len, max_seq_len)
        for h in range(n_heads):
            bias[h] = torch.sin(freqs[h] * pos_diff / max_seq_len)
        
        return bias * 0.1  # Escala pequena para não dominar
    
    @classmethod
    def pi_harmonic_init(cls, shape: Tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
        """Inicialização π-harmônica para pesos."""
        cls.initialize()
        tensor = torch.zeros(shape)
        
        if len(shape) == 2:
            rows, cols = shape
            for i in range(rows):
                for j in range(cols):
                    prime = cls._primes[j % len(cls._primes)]
                    tensor[i, j] = math.sin(math.pi * i * prime / rows)
        elif len(shape) == 1:
            for i in range(shape[0]):
                prime = cls._primes[i % len(cls._primes)]
                tensor[i] = math.sin(math.pi * prime / shape[0])
        
        return tensor * scale


# =============================================================================
# 2. MATH TOKENIZER
# =============================================================================

class MathTokenizer:
    """Tokenizer para expressões matemáticas LaTeX."""
    
    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self._build_vocab()
    
    def _build_vocab(self):
        vocab = self.special_tokens.copy()
        
        math_tokens = [
            '\\frac', '\\sqrt', '\\sum', '\\int', '\\lim', '\\infty', '\\pi',
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\theta', '\\lambda',
            '\\mu', '\\sigma', '\\tau', '\\phi', '\\omega', '\\epsilon',
            '\\partial', '\\nabla', '\\cdot', '\\times', '\\div', '\\pm',
            '\\cap', '\\cup', '\\subset', '\\supset', '\\in', '\\notin',
            '\\forall', '\\exists', '\\neg', '\\wedge', '\\vee', '\\implies',
            '\\iff', '\\equiv', '\\perp', '\\parallel', '\\angle', '\\triangle',
            '\\binom', '\\mod', '\\pmod', '\\gcd', '\\lcm', '\\min', '\\max',
            '\\log', '\\ln', '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
            '\\arcsin', '\\arccos', '\\arctan', '\\exp', '\\det',
            '\\vec', '\\hat', '\\bar', '\\dot', '\\overline',
            '\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg',
            '\\begin', '\\end', '\\boxed', '\\text', '\\mathrm', '\\mathbf',
            '+', '-', '*', '/', '=', '<', '>', '!', '^', '_', '\'',
            '(', ')', '[', ']', '{', '}', '|', ',', '.', ';', ':',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        vocab.extend(math_tokens)
        
        while len(vocab) < self.vocab_size:
            vocab.append(f"<extra_{len(vocab)}>")
        
        self.vocab = vocab[:self.vocab_size]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
    
    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\\[a-zA-Z]+|\d+|[a-zA-Z]|\S', text)
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token_to_id[self.bos_token]]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.token_to_id[self.unk_token]))
        ids.append(self.token_to_id[self.eos_token])
        
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids.extend([self.token_to_id[self.pad_token]] * (max_length - len(ids)))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, self.unk_token)
            if token not in self.special_tokens:
                tokens.append(token)
        return ''.join(tokens)


# =============================================================================
# 3. ADAPTIVE PI-POSITIONAL ENCODING
# =============================================================================

class AdaptivePiPositional(nn.Module):
    """
    Positional encoding com frequência adaptativa baseada no conteúdo.
    
    Diferente do padrão:
    - Frequência base aprendida do conteúdo
    - Usa potências de π (não 10000)
    """
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Adaptador de frequência baseado no conteúdo
        self.freq_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # Garante frequência positiva
        )
        
        # Frequências π-prime base (registradas como buffer)
        pi_freqs = PiBaseSystem.get_pi_prime_frequencies(d_model // 2)
        self.register_buffer('base_freqs', pi_freqs)
        
        # Positional encoding estático como fallback
        self.register_buffer('static_pe', self._create_static_pe(max_len, d_model))
    
    def _create_static_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(torch.tensor(math.pi), torch.arange(0, d_model, 2).float() / d_model)
        
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term[:d_model // 2])
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            pe: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Frequência adaptativa baseada no conteúdo médio
        content_summary = x.mean(dim=1, keepdim=True)  # [B, 1, d]
        freq_scale = self.freq_adapter(content_summary)  # [B, 1, 1]
        freq_scale = freq_scale.clamp(0.5, 2.0)  # Limitar range
        
        # Posições
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        positions = positions.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        
        # Frequências escaladas
        freqs = self.base_freqs[:d_model // 2].unsqueeze(0).unsqueeze(0)  # [1, 1, d/2]
        scaled_freqs = freqs * freq_scale  # [B, 1, d/2]
        
        # Calcular PE adaptativo
        angles = positions / (scaled_freqs + 1e-6)  # [B, T, d/2]
        
        pe = torch.zeros(batch_size, seq_len, d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        
        return pe


# =============================================================================
# 4. PI-HARMONIC EMBEDDING
# =============================================================================

class PiHarmonicEmbedding(nn.Module):
    """Embedding com inicialização π-harmônica e positional adaptativo."""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.adaptive_positional = AdaptivePiPositional(max_seq_len, d_model)
        
        # Inicialização π-harmônica
        self._init_pi_embeddings(vocab_size, d_model)
    
    def _init_pi_embeddings(self, vocab_size: int, d_model: int):
        with torch.no_grad():
            pi_freqs = PiBaseSystem.get_pi_prime_frequencies(d_model)
            
            for i in range(vocab_size):
                if i < 100:
                    pattern = torch.sin(pi_freqs * i / 100)
                else:
                    pattern = torch.sin(pi_freqs * (i - 100) / vocab_size) * 0.5
                self.token_embedding.weight[i] = pattern * 0.02
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        tok_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        pos_emb = self.adaptive_positional(tok_emb)
        return tok_emb + pos_emb


# =============================================================================
# 5. PI-AWARE ATTENTION (COM BIAS π-HARMÔNICO)
# =============================================================================

class PiAwareAttention(nn.Module):
    """
    Multi-head attention com bias posicional π-harmônico.
    
    Diferença do padrão:
    - Bias de atenção baseado em sin(π × prime × pos_diff)
    - Filtro espectral opcional
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, 
                 max_seq_len: int = 256, use_spectral: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_spectral = use_spectral
        self.max_seq_len = max_seq_len
        
        assert d_model % n_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Bias π-harmônico (registrado como buffer, não parâmetro)
        pi_bias = PiBaseSystem.get_pi_attention_bias(n_heads, max_seq_len)
        self.register_buffer('pi_bias', pi_bias)
        
        # Peso learnable para o bias
        self.bias_scale = nn.Parameter(torch.tensor(0.1))
        
        # Parâmetro espectral
        self.spectral_alpha = nn.Parameter(torch.tensor(1.0))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Projeções
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape para multi-head
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)
        
        # Filtro espectral opcional
        if self.use_spectral and seq_len > 4:
            q_fft = torch.fft.fft(q, dim=2, norm='ortho')
            k_fft = torch.fft.fft(k, dim=2, norm='ortho')
            
            freqs = torch.fft.fftfreq(seq_len, device=x.device)
            filter_response = torch.exp(-self.spectral_alpha.abs() * freqs.abs())
            filter_response = filter_response.view(1, 1, seq_len, 1)
            
            q = torch.fft.ifft(q_fft * filter_response, dim=2, norm='ortho').real
            k = torch.fft.ifft(k_fft * filter_response, dim=2, norm='ortho').real
        
        # Scores de atenção
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Adicionar bias π-harmônico
        pi_bias_slice = self.pi_bias[:, :seq_len, :seq_len]  # [n_heads, T, T]
        attn_scores = attn_scores + self.bias_scale * pi_bias_slice.unsqueeze(0)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = rearrange(output, 'b h t d -> b t (h d)')
        
        return self.out_proj(output)


# =============================================================================
# 6. TOKEN-WISE GATE (GATE POR DIMENSÃO)
# =============================================================================

class TokenWiseGate(nn.Module):
    """
    Gate por dimensão, não escalar.
    Permite controle mais fino sobre quais features vêm de qual path.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x_attn: torch.Tensor, x_ffn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_attn: Output do path de atenção [B, T, d]
            x_ffn: Output do path FFN [B, T, d]
        Returns:
            Combinação gated [B, T, d]
        """
        gate = self.gate_net(x_attn)  # [B, T, d] - gate por dimensão
        return gate * x_ffn + (1 - gate) * x_attn


# =============================================================================
# 7. GATED NEURO-SYMBOLIC LAYER (MELHORADO)
# =============================================================================

class GatedNeuroSymbolicLayer(nn.Module):
    """
    Camada neuro-simbólica com:
    - PiAwareAttention (bias π-harmônico)
    - TokenWiseGate (gate por dimensão)
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, 
                 max_seq_len: int = 256, ff_mult: int = 4):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.attention = PiAwareAttention(d_model, n_heads, dropout, max_seq_len)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout)
        )
        
        # Token-wise gate
        self.gate = TokenWiseGate(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm
        x_norm = self.ln1(x)
        
        # Paths paralelos
        x_attn = self.attention(x_norm, mask)
        x_ffn = self.ffn(x_norm)
        
        # Gate por dimensão
        x_combined = self.gate(x_attn, x_ffn)
        
        # Residual
        x = x + self.dropout(x_combined)
        
        # Segunda sub-camada
        x = x + self.dropout(self.ffn(self.ln2(x)))
        
        return x


# =============================================================================
# 8. MODELO COMPLETO
# =============================================================================

class AIMO3PiModel(nn.Module):
    """
    Modelo AIMO3 completo com todas as melhorias PiBaseSystem.
    """
    
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        
        PiBaseSystem.initialize()
        
        self.embedding = PiHarmonicEmbedding(vocab_size, d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            GatedNeuroSymbolicLayer(d_model, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        return self.lm_head(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 9. SYMPY TOOL
# =============================================================================

class SymPyTool:
    def __init__(self):
        self.available = SYMPY_AVAILABLE
        if self.available:
            self.symbols = symbols('x y z a b c n k m')
    
    def solve_equation(self, eq_str: str, var: str = 'x') -> str:
        if not self.available:
            return "SymPy não disponível"
        try:
            if '=' in eq_str:
                left, right = eq_str.split('=', 1)
                eq = Eq(sp.sympify(left), sp.sympify(right))
            else:
                eq = Eq(sp.sympify(eq_str), 0)
            return str(solve(eq, symbols(var)))
        except Exception as e:
            return f"Erro: {e}"
    
    def verify_answer(self, problem: str, answer: int) -> bool:
        if not self.available:
            return True
        try:
            if answer < 0 or answer > 99999:
                return False
            problem_lower = problem.lower()
            if 'positive' in problem_lower and answer <= 0:
                return False
            if 'even' in problem_lower and answer % 2 != 0:
                return False
            if 'odd' in problem_lower and answer % 2 == 0:
                return False
            return True
        except:
            return True


# =============================================================================
# 10. MATH GENERATOR
# =============================================================================

class MathGenerator:
    def __init__(self, model: AIMO3PiModel, tokenizer: MathTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sympy = SymPyTool()
    
    def sample(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-8)
        
        idx = torch.multinomial(sorted_probs, 1)
        return sorted_idx[idx].item()
    
    def generate_one(self, problem: str, max_tokens: int = 50) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device
        
        input_ids = self.tokenizer.encode(problem, max_length=self.model.max_seq_len - max_tokens)
        input_ids = torch.tensor([input_ids], device=device)
        
        generated = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if input_ids.shape[1] >= self.model.max_seq_len:
                    break
                
                logits = self.model(input_ids)
                next_logits = logits[0, -1, :]
                next_token = self.sample(next_logits)
                generated.append(next_token)
                
                if next_token == self.tokenizer.token_to_id[self.tokenizer.eos_token]:
                    break
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=device)
                ], dim=1)
        
        return self.tokenizer.decode(generated)
    
    def extract_answer(self, text: str) -> Optional[int]:
        boxed = re.search(r'\\boxed\{(\d+)\}', text)
        if boxed:
            return int(boxed.group(1))
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])
        return None
    
    def solve(self, problem: str, num_samples: int = 8) -> int:
        answers = []
        for _ in range(num_samples):
            try:
                response = self.generate_one(problem)
                ans = self.extract_answer(response)
                if ans is not None:
                    answers.append(ans)
            except:
                continue
        
        if not answers:
            return 0
        
        counter = Counter(answers)
        best = counter.most_common(1)[0][0]
        
        if self.sympy.verify_answer(problem, best):
            return best
        return best


# =============================================================================
# 11. DIFFICULTY SCHEDULER (CURRICULUM LEARNING)
# =============================================================================

class DifficultyLevel(Enum):
    BASIC = 1        # Soma, subtração simples
    INTERMEDIATE = 2  # Multiplicação, divisão
    ADVANCED = 3      # Frações, equações lineares
    EXPERT = 4        # Sistemas, quadráticas


@dataclass
class DifficultyScheduler:
    """Gerencia progressão de dificuldade."""
    current_level: DifficultyLevel = DifficultyLevel.BASIC
    steps_at_level: int = 0
    steps_to_advance: int = 100
    loss_threshold: float = 0.5
    
    def should_advance(self, loss: float) -> bool:
        self.steps_at_level += 1
        
        if loss < self.loss_threshold and self.steps_at_level >= self.steps_to_advance:
            return True
        return False
    
    def advance(self):
        if self.current_level.value < DifficultyLevel.EXPERT.value:
            self.current_level = DifficultyLevel(self.current_level.value + 1)
            self.steps_at_level = 0
            self.loss_threshold *= 0.9  # Threshold mais rígido
    
    def get_level(self) -> DifficultyLevel:
        return self.current_level


# =============================================================================
# 12. PROBLEM AUGMENTATION
# =============================================================================

class ProblemAugmenter:
    """Gera variações de problemas matemáticos."""
    
    @staticmethod
    def augment_numbers(problem: str, answer: int) -> Tuple[str, int]:
        """Substitui números por variantes."""
        numbers = re.findall(r'\d+', problem)
        if len(numbers) < 2:
            return problem, answer
        
        # Escalar todos os números
        scale = random.choice([2, 3, 5])
        new_problem = problem
        
        for num in numbers:
            new_num = int(num) * scale
            new_problem = new_problem.replace(num, str(new_num), 1)
        
        return new_problem, answer * scale
    
    @staticmethod
    def swap_operands(problem: str, answer: int) -> Tuple[str, int]:
        """Troca ordem de operandos (para operações comutativas)."""
        if '+' in problem or '*' in problem or '\\times' in problem:
            numbers = re.findall(r'\d+', problem)
            if len(numbers) == 2:
                new_problem = problem.replace(numbers[0], 'TEMP')
                new_problem = new_problem.replace(numbers[1], numbers[0])
                new_problem = new_problem.replace('TEMP', numbers[1])
                return new_problem, answer
        return problem, answer
    
    @staticmethod
    def add_context(problem: str, answer: int) -> Tuple[str, int]:
        """Adiciona contexto textual."""
        contexts = [
            "Calculate: ",
            "Find the value of ",
            "Compute ",
            "What is ",
            "Determine ",
        ]
        return random.choice(contexts) + problem, answer


# =============================================================================
# 13. CURRICULUM TRAINER
# =============================================================================

class CurriculumTrainer:
    """Trainer com curriculum learning e augmentação."""
    
    def __init__(self, model: AIMO3PiModel, tokenizer: MathTokenizer, lr: float = 1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.token_to_id[tokenizer.pad_token],
            label_smoothing=0.1  # Label smoothing
        )
        
        self.difficulty = DifficultyScheduler()
        self.augmenter = ProblemAugmenter()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        batch = batch.to(self.device)
        
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits = self.model(input_ids)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train_epoch(self, loader: DataLoader) -> float:
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Verificar se deve avançar nível
            if self.difficulty.should_advance(loss):
                self.difficulty.advance()
                logging.info(f"Avançando para nível: {self.difficulty.get_level().name}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                logits = self.model(input_ids)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss


# =============================================================================
# 14. DATASET COM CURRICULUM
# =============================================================================

class CurriculumMathDataset(Dataset):
    """Dataset que gera problemas baseado no nível de dificuldade."""
    
    def __init__(self, tokenizer: MathTokenizer, n_samples: int = 500, 
                 difficulty: DifficultyLevel = DifficultyLevel.BASIC,
                 max_len: int = 256, augment: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmenter = ProblemAugmenter()
        
        self.problems, self.answers = self._generate_problems(n_samples, difficulty)
    
    def _generate_problems(self, n: int, level: DifficultyLevel) -> Tuple[List[str], List[int]]:
        problems, answers = [], []
        
        if level == DifficultyLevel.BASIC:
            for _ in range(n):
                a, b = np.random.randint(1, 20, 2)
                if random.random() < 0.5:
                    problems.append(f"What is ${a} + {b}$?")
                    answers.append(a + b)
                else:
                    problems.append(f"What is ${max(a,b)} - {min(a,b)}$?")
                    answers.append(abs(a - b))
        
        elif level == DifficultyLevel.INTERMEDIATE:
            for _ in range(n):
                a, b = np.random.randint(1, 15, 2)
                if random.random() < 0.5:
                    problems.append(f"What is ${a} \\times {b}$?")
                    answers.append(a * b)
                else:
                    c = a * b
                    problems.append(f"What is ${c} \\div {a}$?")
                    answers.append(b)
        
        elif level == DifficultyLevel.ADVANCED:
            for _ in range(n):
                a, b = np.random.randint(1, 10, 2)
                c, d = np.random.randint(1, 10, 2)
                if b != 0 and d != 0:
                    num = a * d + c * b
                    den = b * d
                    gcd = math.gcd(num, den)
                    num //= gcd
                    den //= gcd
                    if den == 1:
                        problems.append(f"Compute $\\frac{{{a}}}{{{b}}} + \\frac{{{c}}}{{{d}}}$")
                        answers.append(num)
                    else:
                        # Fallback para soma simples
                        problems.append(f"What is ${a} + {b} + {c} + {d}$?")
                        answers.append(a + b + c + d)
        
        elif level == DifficultyLevel.EXPERT:
            for _ in range(n):
                a = np.random.randint(1, 10)
                b = np.random.randint(1, 20)
                # ax + c = b, x = (b-c)/a
                c = np.random.randint(1, 10)
                if (b - c) % a == 0:
                    x = (b - c) // a
                    problems.append(f"Find $x$ if ${a}x + {c} = {b}$")
                    answers.append(x)
                else:
                    problems.append(f"What is ${a} + {b} + {c}$?")
                    answers.append(a + b + c)
        
        return problems, answers
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        answer = self.answers[idx]
        
        # Augmentação
        if self.augment and random.random() < 0.3:
            aug_fn = random.choice([
                self.augmenter.add_context,
                self.augmenter.swap_operands
            ])
            problem, answer = aug_fn(problem, answer)
        
        text = f"{problem} Answer: {answer}"
        ids = self.tokenizer.encode(text, max_length=self.max_len)
        return torch.tensor(ids, dtype=torch.long)


# =============================================================================
# 15. FACTORY E MAIN
# =============================================================================

def create_model(
    vocab_size: int = 2048,
    d_model: int = 64,
    n_layers: int = 3,
    n_heads: int = 4
) -> Tuple[AIMO3PiModel, MathTokenizer, MathGenerator]:
    tokenizer = MathTokenizer(vocab_size)
    model = AIMO3PiModel(vocab_size, d_model, n_layers, n_heads)
    generator = MathGenerator(model, tokenizer)
    return model, tokenizer, generator


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Criar modelo
    model, tokenizer, generator = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    param_count = model.count_parameters()
    logging.info(f"Parâmetros: {param_count:,}")
    logging.info(f"Limite 1M: {'✓' if param_count < 1_000_000 else '✗'}")
    logging.info(f"Device: {device}")
    
    # Curriculum training
    trainer = CurriculumTrainer(model, tokenizer)
    
    logging.info("\n=== Curriculum Training ===")
    
    for level in DifficultyLevel:
        logging.info(f"\nNível: {level.name}")
        
        dataset = CurriculumMathDataset(tokenizer, n_samples=300, difficulty=level)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for epoch in range(2):
            train_loss = trainer.train_epoch(loader)
            logging.info(f"  Epoch {epoch}: Loss = {train_loss:.4f}")
    
    # Testar
    test_problems = [
        ("What is $7 + 8$?", 15),
        ("What is $10 - 4$?", 6),
        ("What is $3 \\times 5$?", 15),
        ("What is $12 \\div 4$?", 3),
    ]
    
    logging.info("\n=== Testes ===")
    for problem, true_ans in test_problems:
        pred = generator.solve(problem, num_samples=4)
        status = "✓" if pred == true_ans else "✗"
        logging.info(f"{status} {problem} -> Pred: {pred}, True: {true_ans}")
    
    # Demo PiBaseSystem
    logging.info("\n=== PiBaseSystem Demo ===")
    for val in [math.pi, 10.0, 100.0]:
        pi_repr = PiBaseSystem.to_pi_base(val)
        recovered = PiBaseSystem.from_pi_base(pi_repr)
        logging.info(f"  {val:.4f} -> π-base -> {recovered:.4f}")
    
    # Demo attention bias
    logging.info("\n=== Pi Attention Bias Demo ===")
    bias = PiBaseSystem.get_pi_attention_bias(4, 16)
    logging.info(f"  Shape: {bias.shape}")
    logging.info(f"  Range: [{bias.min():.3f}, {bias.max():.3f}]")


def kaggle_submission(model_path: Optional[str] = None):
    if not PANDAS_AVAILABLE:
        print("Pandas necessário")
        return
    
    model, tokenizer, generator = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    test_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv')
    
    submissions = []
    for _, row in test_df.iterrows():
        try:
            answer = generator.solve(row['problem'], num_samples=32)
            answer = max(0, min(99999, answer))
        except:
            answer = 0
        submissions.append({'id': row['id'], 'answer': answer})
    
    pd.DataFrame(submissions).to_csv('submission.csv', index=False)
    print("Submission salvo")


if __name__ == "__main__":
    main()
