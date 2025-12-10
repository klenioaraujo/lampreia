import torch
import torch.nn as nn
import math

class PiAwareModel(nn.Module):
    """Versão simplificada com as melhores ideias do PiBaseSystem"""
    
    def __init__(self, vocab_size=5000, d_model=128):
        super().__init__()
        
        # Embedding com inicialização π-aware
        self.embedding = nn.Embedding(vocab_size, d_model)
        self._init_pi_aware()
        
        # Atenção padrão (rápida)
        self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        
        # Gate neuro-simbólico simplificado
        self.symbolic_gate = nn.Linear(d_model, 1)
        
        # Camada π-simbólica (aproximação diferenciável)
        self.pi_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def _init_pi_aware(self):
        """Inicializa embeddings com padrões π-harmônicos"""
        with torch.no_grad():
            for i in range(self.embedding.num_embeddings):
                # Padrão baseado em π para tokens numéricos
                if i < 1000:  # Assumindo primeiros tokens são números
                    pattern = torch.sin(
                        torch.arange(self.embedding.embedding_dim) * 
                        math.pi * i / 1000
                    )
                    self.embedding.weight[i] = pattern
    
    def forward(self, x):
        x = self.embedding(x)
        
        # Atenção neural
        x_attn, _ = self.attention(x, x, x)
        
        # Transformação π-simbólica
        x_pi = self.pi_layer(x)
        
        # Gate adaptativo
        gate = torch.sigmoid(self.symbolic_gate(x))
        x = gate * x_pi + (1 - gate) * x_attn
        
        return x
