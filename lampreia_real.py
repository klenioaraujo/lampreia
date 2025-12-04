# ==========================================================
# ΨQRH HAMILTONIAN MONTE CARLO - CORRIGIDO (COM AUTOGRAD)
# ==========================================================

import torch
import torch.nn as nn
import logging
import sys
import numpy as np
from typing import Tuple, Callable, List
import matplotlib.pyplot as plt

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, force=True)

# ---------- 1. SISTEMA HAMILTONIANO (CORREÇÃO CRÍTICA APLICADA) ----------
class HamiltonianSystem(nn.Module):
    """Sistema Hamiltoniano que usa Autograd para dinâmica correta"""
    
    def __init__(self, potential_energy_fn: Callable, input_dim: int = 2):
        super().__init__()
        self.potential_energy_fn = potential_energy_fn
        self.input_dim = input_dim
        
        # Matriz de massa (M) e sua inversa (M⁻¹)
        # Usamos nn.Parameter caso queiramos aprender a métrica no futuro
        self.mass_matrix = nn.Parameter(torch.eye(input_dim))
        # Pré-calculamos a inversa para eficiência (assumindo matriz diagonal/identidade)
        self.mass_matrix_inv = torch.inverse(self.mass_matrix)
    
    def potential_energy(self, position: torch.Tensor) -> torch.Tensor:
        """Energia potencial U(q)"""
        # Nota: Não usamos torch.no_grad() aqui porque este método
        # será chamado dentro de um contexto que precisa de gradientes
        # nas equações de Hamilton.
        return self.potential_energy_fn(position)
    
    def kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
        """Energia cinética K(p) = ½ pᵀ M⁻¹ p"""
        if momentum.dim() == 1:
            momentum = momentum.unsqueeze(0)
        # Cálculo eficiente para batch de momentos
        return 0.5 * torch.sum(momentum * torch.matmul(self.mass_matrix_inv, momentum.T).T, dim=-1)
    
    def hamiltonian(self, position: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """Hamiltoniano total H(q, p) = U(q) + K(p)"""
        # Usado apenas para calcular probabilidades de aceitação, não precisa de grad
        with torch.no_grad():
            return self.potential_energy(position) + self.kinetic_energy(momentum)
    
    def hamiltonian_equations(self, position: torch.Tensor, momentum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Equações de Hamilton usando PyTorch Autograd (CORREÇÃO CRÍTICA).
        Retorna (dq/dt, dp/dt).
        """
        # Garantir dimensões de batch
        if position.dim() == 1: position = position.unsqueeze(0)
        if momentum.dim() == 1: momentum = momentum.unsqueeze(0)
            
        # --- 1. Cálculo de dq/dt = ∂H/∂p = M⁻¹ p ---
        # Como K(p) é quadrático, o gradiente é analítico e simples.
        dq_dt = torch.matmul(self.mass_matrix_inv, momentum.T).T
        
        # --- 2. Cálculo de dp/dt = -∂H/∂q = -∇U(q) ---
        # Habilitamos o rastreamento de gradiente na posição temporariamente
        pos_with_grad = position.detach().requires_grad_(True)
        
        # Calculamos a energia potencial neste ponto
        u_at_pos = self.potential_energy(pos_with_grad)
        
        # Usamos torch.autograd.grad para obter o gradiente exato ∇U(q).
        # .sum() é necessário para agregar o loss se tivermos um batch, 
        # mas o gradiente resultante mantém a forma do batch.
        grad_u = torch.autograd.grad(
            outputs=u_at_pos.sum(), 
            inputs=pos_with_grad,
            create_graph=False, # Não precisamos de derivadas de segunda ordem aqui
            retain_graph=False,
            only_inputs=True
        )[0]
        
        # As equações de Hamilton definem dp/dt como o negativo do gradiente do potencial
        dp_dt = -grad_u.detach()
        
        return dq_dt, dp_dt

# ---------- 2. INTEGRADOR LEAPFROG (Funcional e Correto) ----------
class LeapfrogIntegrator:
    """Integrador simplético Leapfrog"""
    
    def __init__(self, hamiltonian_system: HamiltonianSystem, step_size: float = 0.1):
        self.hamiltonian_system = hamiltonian_system
        self.step_size = step_size
    
    def single_step(self, position: torch.Tensor, momentum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Um passo do integrador Leapfrog"""
        # Não precisamos de gradientes sendo rastreados através dos passos do integrador
        with torch.no_grad():
            # Garantir dimensões
            if position.dim() == 1: position = position.unsqueeze(0)
            if momentum.dim() == 1: momentum = momentum.unsqueeze(0)
                
            # 1. Meio passo no momento (p -> p_{t+e/2})
            # Usamos a posição atual para calcular o gradiente da força
            _, dp_dt_initial = self.hamiltonian_system.hamiltonian_equations(position, momentum)
            momentum_half = momentum + 0.5 * self.step_size * dp_dt_initial
            
            # 2. Passo completo na posição (q -> q_{t+e})
            # Usamos o momento de meio passo
            dq_dt_half, _ = self.hamiltonian_system.hamiltonian_equations(position, momentum_half)
            position_new = position + self.step_size * dq_dt_half
            
            # 3. Meio passo restante no momento (p_{t+e/2} -> p_{t+e})
            # Usamos a nova posição para calcular a força final
            _, dp_dt_final = self.hamiltonian_system.hamiltonian_equations(position_new, momentum_half)
            momentum_new = momentum_half + 0.5 * self.step_size * dp_dt_final
            
            return position_new, momentum_new
    
    def integrate(self, position: torch.Tensor, momentum: torch.Tensor, 
                  num_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integração por múltiplos passos para gerar uma trajetória"""
        if position.dim() == 1: position = position.unsqueeze(0)
        if momentum.dim() == 1: momentum = momentum.unsqueeze(0)
            
        trajectory_pos = [position.detach().clone()]
        trajectory_mom = [momentum.detach().clone()]
        
        current_pos, current_mom = position.detach().clone(), momentum.detach().clone()
        
        for _ in range(num_steps):
            current_pos, current_mom = self.single_step(current_pos, current_mom)
            trajectory_pos.append(current_pos.detach().clone())
            trajectory_mom.append(current_mom.detach().clone())
        
        return torch.stack(trajectory_pos), torch.stack(trajectory_mom)

# ---------- 3. HAMILTONIAN MONTE CARLO (Amostrador) ----------
class HamiltonianMonteCarlo:
    """Amostrador HMC"""
    
    def __init__(self, potential_energy_fn: Callable, input_dim: int = 2,
                 step_size: float = 0.1, num_steps: int = 10):
        self.input_dim = input_dim
        # Instancia o sistema corrigido
        self.hamiltonian_system = HamiltonianSystem(potential_energy_fn, input_dim)
        self.integrator = LeapfrogIntegrator(self.hamiltonian_system, step_size)
        self.num_steps = num_steps
        
    def sample(self, initial_position: torch.Tensor, num_samples: int = 1000, 
               burn_in: int = 100) -> Tuple[torch.Tensor, List[float], List[float]]:
        """Gera amostras usando HMC"""
        
        if initial_position.dim() == 1:
            initial_position = initial_position.unsqueeze(0)
            
        current_position = initial_position.detach().clone()
        samples = []
        acceptance_rates = []
        hamiltonian_values = []
        
        for i in range(num_samples + burn_in):
            # 1. Amostragem de Gibbs do momento (p ~ N(0, M))
            # Como M é identidade, p ~ N(0, I)
            momentum = torch.randn_like(current_position)
            
            # Calcular Hamiltoniano inicial H(q_init, p_init)
            H_initial = self.hamiltonian_system.hamiltonian(current_position, momentum)
            
            # 2. Integração da trajetória Hamiltoniana
            position_proposal, momentum_proposal = self.integrator.integrate(
                current_position, momentum, self.num_steps
            )
            
            # Pegar o estado final da trajetória
            final_pos = position_proposal[-1]
            final_mom = momentum_proposal[-1]
            
            # Calcular Hamiltoniano final H(q_final, p_final)
            H_final = self.hamiltonian_system.hamiltonian(final_pos, final_mom)
            
            # 3. Critério de Metropolis
            # log(alpha) = H_initial - H_final (devido ao sinal negativo na def. de probabilidade)
            log_accept_ratio = H_initial - H_final
            # Usamos clamp para evitar overflow na exponencial
            accept_prob = torch.exp(torch.clamp(log_accept_ratio, max=0.0))
            
            # Aceitar ou rejeitar
            if torch.rand(1, device=current_position.device) < accept_prob:
                current_position = final_pos.detach().clone()
                acceptance_rates.append(1.0)
            else:
                # Se rejeitado, mantém a posição atual
                acceptance_rates.append(0.0)
            
            # Guardar amostra após período de aquecimento (burn-in)
            if i >= burn_in:
                samples.append(current_position.detach().clone())
                # Guardamos o H_final da amostra aceita (ou H_initial se rejeitada, 
                # mas H deve ser conservado, então H_final é uma boa proxy)
                hamiltonian_values.append(H_final.item())
        
        return torch.stack(samples), acceptance_rates, hamiltonian_values

# ---------- 4. POTENCIAIS (Funções U(q)) ----------
class ComplexPotentials:
    """
    Funções de energia potencial.
    IMPORTANTE: Não precisam de decoradores @torch.no_grad() ou tratamento
    manual de dimensões complexo, pois o sistema Hamiltoniano e o Autograd
    lidam com isso. Mantive a estrutura básica para compatibilidade.
    """
    
    @staticmethod
    def multimodal_2d(position: torch.Tensor) -> torch.Tensor:
        """Potencial multimodal 2D (Mistura de poços gaussianos)"""
        # Garante que temos [batch, dim]
        if position.dim() == 1: position = position.unsqueeze(0)
        x, y = position[:, 0], position[:, 1]
            
        potential = (
            -torch.exp(-((x - 2.0)**2 + (y - 2.0)**2)) 
            - torch.exp(-((x + 2.0)**2 + (y + 2.0)**2))
            - torch.exp(-((x - 2.0)**2 + (y + 2.0)**2))
            - torch.exp(-((x + 2.0)**2 + (y - 2.0)**2))
            + 0.1 * (x**2 + y**2) # Termo quadrático suave para confinar as amostras
        )
        return potential
    
    @staticmethod
    def rosenbrock(position: torch.Tensor) -> torch.Tensor:
        """Função de Rosenbrock 2D (O 'vale da banana')"""
        if position.dim() == 1: position = position.unsqueeze(0)
        x, y = position[:, 0], position[:, 1]
        # Mínimo global em (1, 1)
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    @staticmethod
    def double_well_1d(position: torch.Tensor) -> torch.Tensor:
        """Poço duplo 1D simples"""
        if position.dim() == 1: position = position.unsqueeze(0)
        x = position[:, 0]
        # Mínimos em -1 e +1
        return (x**2 - 1)**2

# ---------- 5. VISUALIZAÇÃO (Mantida do original) ----------
def visualize_hamiltonian_dynamics(hmc_sampler: HamiltonianMonteCarlo, 
                                 initial_position: torch.Tensor,
                                 num_steps: int = 100):
    """Visualiza trajetórias Hamiltonianas no espaço de fase"""
    
    # Amostrar momento inicial
    momentum = torch.randn_like(initial_position)
    
    # Integrar trajetória
    positions, momenta = hmc_sampler.integrator.integrate(
        initial_position, momentum, num_steps
    )
    
    # Calcular energias ao longo da trajetória para verificar conservação
    # Precisamos de no_grad aqui pois estamos apenas extraindo valores para plotar
    with torch.no_grad():
        potential_energies = torch.stack([
            hmc_sampler.hamiltonian_system.potential_energy(pos) 
            for pos in positions
        ]).cpu().numpy().squeeze()
        
        kinetic_energies = torch.stack([
            hmc_sampler.hamiltonian_system.kinetic_energy(mom) 
            for mom in momenta
        ]).cpu().numpy().squeeze()
    
    total_energies = potential_energies + kinetic_energies
    
    # Converter para numpy para plotting
    positions_np = positions.detach().cpu().numpy()
    momenta_np = momenta.detach().cpu().numpy()
    
    input_dim = positions_np.shape[-1]
    
    # Plot baseado na dimensionalidade
    if input_dim == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trajetória 1D (Tempo vs Posição)
        time_steps = np.arange(len(positions_np))
        axes[0, 0].plot(time_steps, positions_np[:, 0, 0], 'b-', linewidth=2)
        axes[0, 0].scatter(time_steps[0], positions_np[0, 0, 0], color='green', s=100, label='Start')
        axes[0, 0].scatter(time_steps[-1], positions_np[-1, 0, 0], color='red', s=100, label='End')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Position q')
        axes[0, 0].set_title('1D Hamiltonian Trajectory (q vs t)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
    elif input_dim >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trajetória 2D (q1 vs q2)
        axes[0, 0].plot(positions_np[:, 0, 0], positions_np[:, 0, 1], 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].scatter(positions_np[0, 0, 0], positions_np[0, 0, 1], color='green', s=100, label='Start', zorder=5)
        axes[0, 0].scatter(positions_np[-1, 0, 0], positions_np[-1, 0, 1], color='red', s=100, label='End', zorder=5)
        axes[0, 0].set_xlabel('Position q₁')
        axes[0, 0].set_ylabel('Position q₂')
        axes[0, 0].set_title(f'{input_dim}D Hamiltonian Trajectory (Space)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Energias (Diagnóstico de conservação H = U + K)
    time_steps = np.arange(len(total_energies))
    axes[0, 1].plot(time_steps, potential_energies, 'r-', label='Potential U(q)', linewidth=2, alpha=0.6)
    axes[0, 1].plot(time_steps, kinetic_energies, 'g-', label='Kinetic K(p)', linewidth=2, alpha=0.6)
    axes[0, 1].plot(time_steps, total_energies, 'b-', label='Total H(q,p)', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Energy')
    # O Hamiltoniano total deve permanecer constante (linha azul plana)
    axes[0, 1].set_title('Energy Conservation (Blue line should be flat)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribuição de momentos (Diagnóstico)
    axes[1, 0].hist(momenta_np[:, 0, 0], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('Momentum p₁')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Momentum Distribution along trajectory')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Espaço de fase (q vs p)
    if input_dim == 1:
        axes[1, 1].plot(positions_np[:, 0, 0], momenta_np[:, 0, 0], 'purple', linewidth=2)
    else:
        # Para >1D, plotamos a primeira dimensão
        axes[1, 1].plot(positions_np[:, 0, 0], momenta_np[:, 0, 0], 'purple', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Position q₁')
    axes[1, 1].set_ylabel('Momentum p₁')
    axes[1, 1].set_title('Phase Space projected on dim 1 (q₁ vs p₁)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Estatísticas de conservação de energia
    # Quanto menor a variação, melhor o integrador
    energy_variation = np.std(total_energies) / np.mean(np.abs(total_energies))
    print(f"📊 Energy Conservation Metric (CV): {energy_variation:.6f} (Closer to 0 is better)")
    
    return positions, momenta, total_energies

# ---------- 6. DEMONSTRAÇÃO ----------
def demonstrate_hamiltonian_mc():
    print("=" * 80)
    print("ΨQRH HAMILTONIAN MONTE CARLO - VERSÃO CORRIGIDA (AUTOGRAD)")
    print("=" * 80)
    
    # Definir dispositivo (GPU se disponível, o que agora faz diferença!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Running on device: {device}")
    torch.manual_seed(42) # Para reprodutibilidade
    
    try:
        # --- CASO 1: Potencial Multimodal 2D ---
        print("\n🧪 1. Testing 2D Multimodal Potential...")
        # Nota: Podemos usar passos maiores e menos passos de integração
        # agora que os gradientes são precisos.
        multimodal_sampler = HamiltonianMonteCarlo(
            potential_energy_fn=ComplexPotentials.multimodal_2d,
            input_dim=2,
            step_size=0.15,  # Passo ajustado
            num_steps=20     # Trajetória mais longa
        )
        
        # Movendo o amostrador para o dispositivo correto
        multimodal_sampler.hamiltonian_system.to(device)

        initial_position = torch.tensor([[0.0, 0.0]], device=device)
        
        print("  - Visualizing dynamics first...")
        visualize_hamiltonian_dynamics(multimodal_sampler, initial_position, num_steps=50)

        print("  - Sampling...")
        samples, acc_rates, _ = multimodal_sampler.sample(
            initial_position, num_samples=800, burn_in=200
        )
        print(f"✅ 2D Multimodal Acceptance Rate: {np.mean(acc_rates):.3f}")

        # Visualizar amostras
        samples_np = samples.detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(samples_np[:, 0, 0], samples_np[:, 0, 1], alpha=0.5, s=15, label='HMC Samples')
        # Plotar os centros teóricos dos modos para comparação
        modes = np.array([[2,2], [-2,-2], [2,-2], [-2,2]])
        plt.scatter(modes[:,0], modes[:,1], c='red', marker='x', s=100, label='True Modes')
        plt.title("HMC Samples - 2D Multimodal")
        plt.xlabel("q1"); plt.ylabel("q2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # --- CASO 2: Rosenbrock 2D (Difícil) ---
        print("\n🔬 2. Testing 2D Rosenbrock Potential (The 'Banana')...")
        rosenbrock_sampler = HamiltonianMonteCarlo(
            potential_energy_fn=ComplexPotentials.rosenbrock,
            input_dim=2,
            step_size=0.02, # Rosenbrock precisa de passos menores
            num_steps=50
        )
        rosenbrock_sampler.hamiltonian_system.to(device)
        
        # Começando longe do ótimo (1,1)
        init_rosen = torch.tensor([[-1.0, 0.0]], device=device)
        
        print("  - Sampling Rosenbrock...")
        rosen_samples, rosen_acc, _ = rosenbrock_sampler.sample(
            init_rosen, num_samples=1000, burn_in=500
        )
        print(f"✅ Rosenbrock Acceptance Rate: {np.mean(rosen_acc):.3f}")
        
        rosen_np = rosen_samples.detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(rosen_np[:, 0, 0], rosen_np[:, 0, 1], alpha=0.5, s=10, c='orange')
        plt.scatter(1, 1, c='red', marker='*', s=200, label='Global Min (1,1)')
        plt.title("HMC Samples - Rosenbrock (Exploring the banana)")
        plt.xlim(-2, 2); plt.ylim(-1, 3)
        plt.xlabel("q1"); plt.ylabel("q2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print("\n✅ Demonstration complete. The system is now using true autograd gradients.")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_hamiltonian_mc()
