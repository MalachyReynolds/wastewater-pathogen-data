"""
Object-oriented version of the SPINNS training framework.
Makes it easy to add/remove neural network features and experiment with different architectures.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
from itertools import product
import copy


# ──────────────────── FEATURE MODULES ─────────────────────────────────────────
class RMSELoss(nn.Module):
    """Root Mean Square Error Loss"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)

class FeatureModule(nn.Module, ABC):
    """Base class for input feature transformations."""
    
    @abstractmethod
    def get_output_dim(self, input_dim: int) -> int:
        """Return the output dimension given an input dimension."""
        pass


class IdentityFeature(FeatureModule):
    """Pass through input without transformation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    
    def get_output_dim(self, input_dim: int) -> int:
        return input_dim


class FourierFeature(FeatureModule):
    """Random Fourier Features for neural network input encoding."""
    
    def __init__(self, in_dim: int = 1, num_bands: int = 32, 
                 sigma: float = 5.0, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        self.in_dim = in_dim
        self.num_bands = num_bands
        
        # Random Gaussian projection matrix (fixed)
        B = torch.randn(num_bands, in_dim) * sigma
        self.register_buffer('B', B)
    
    def forward(self, x):
        # Project and apply sine/cosine
        x_proj = 2 * torch.pi * x @ self.B.T
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        if self.include_input:
            return torch.cat([x, emb], dim=-1)
        return emb
    
    def get_output_dim(self, input_dim: int) -> int:
        base_dim = 2 * self.num_bands
        return base_dim + input_dim if self.include_input else base_dim
    
class RWFLinear(nn.Module):
    """Random Weight Factorization Linear Layer: W = diag(exp(s)) @ V"""
    def __init__(self, in_features, out_features, bias=True, mu=1.0, sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Standard weight matrix V
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        # Scale vector s (learnable)
        self.s = nn.Parameter(torch.empty(out_features))
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        # Initialization
        nn.init.xavier_uniform_(self.V)
        nn.init.normal_(self.s, mean=mu, std=sigma)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # W = diag(exp(s)) @ V
        weight = torch.exp(self.s).unsqueeze(1) * self.V
        return nn.functional.linear(input, weight, self.bias)


# ──────────────────── NETWORK ARCHITECTURES ───────────────────────────────────

class Network(nn.Module):
    """
    Flexible free energy neural network with configurable features and architecture.
    """
    
    def __init__(self, 
                 input_dim: int = 30,
                 hidden_dims: List[int] = [64, 64, 64, 64],
                 activation: nn.Module = nn.Tanh,
                 feature_modules: Optional[List[FeatureModule]] = None,
                 use_rwf: bool = False,
                 rwf_mu: float = 0.5,
                 rwf_sigma: float = 0.1):
        super().__init__()
        if feature_modules is None:
            feature_modules = [FourierFeature(in_dim=input_dim, num_bands=2, sigma=0.5, include_input=True)]
        self.feature_modules = nn.ModuleList(feature_modules)
        feature_dim = sum([fm.get_output_dim(input_dim) for fm in self.feature_modules])
        layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            if use_rwf:
                layers.append(RWFLinear(prev_dim, hidden_dim, mu=rwf_mu, sigma=rwf_sigma))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        if use_rwf:
            layers.append(RWFLinear(prev_dim, 1, mu=rwf_mu, sigma=rwf_sigma))
        else:
            layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for standard Linear layers
        if not use_rwf:
            self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for standard linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = [fm(x) for fm in self.feature_modules]
        features = torch.cat(feats, dim=-1)
        return self.network(features)


# ──────────────────── CONFIGURATION ───────────────────────────────────────────

class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    def __init__(self,
                 # Data parameters
                 data_file: str = "filtered_data.csv",
                 
                 # Training parameters
                 batch_size: int = 20,
                 n_steps: int = 1,
                 epochs: int = 100_000,
                 learning_rate: Optional[float] = None,  # None = use Adam default
                 loss_scale: float = 1e10,
                 
                 # Network architecture
                 hidden_dims: List[int] = [64, 64, 64, 64],
                 activation: str = 'Tanh',
                 feature_type: Optional[str] = None,  # Feature type
                 fourier_bands: int = 2,
                 fourier_sigma: float = 0.5,
                 use_rwf: bool = False,  # Enable Random Weight Factorization
                 rwf_mu: float = 1.0,
                 rwf_sigma: float = 0.1,
                 
                 # L-BFGS parameters
                 use_lbfgs: bool = True,
                 lbfgs_max_iter: int = 15000,
                 lbfgs_max_iter_per_step: int = 10,
                 lbfgs_history_size: int = 100,
                 
                 # Output parameters
                 save_outputs: bool = True,
                 save_numpy_outputs: bool = True,  # If False, only write CSV (no .npy arrays or plots)
                 output_base_dir: str = ".",
                 free_energy_output_dir: Optional[str] = None,  # Custom directory for free energy outputs
                 epsilon_output_dir: Optional[str] = None,  # Custom directory for epsilon outputs
                 plot_output_dir: Optional[str] = None,  # Custom directory for plots
                 csv_filename: str = "test_results.csv",  # CSV file for results
                 plot_free_energy: bool = False,  # Toggle to plot predicted vs actual free energy
                 show_plots: bool = False,  # Whether to display plots interactively
                 
                 # Other
                 seed: int = 42,
                 device_id: int = 0,
                 
                 # Learning rate exponential decay
                 use_lr_decay: bool = False,  # Enable exponential LR decay for Adam
                 lr_decay_gamma: float = 0.99998,  # Multiplicative factor per decay step
                 lr_decay_step: int = 1,
                 
                 # Learning rate cosine annealing
                 use_cosine_annealing: bool = False,  # Enable cosine annealing LR schedule
                 cosine_T_max: int = 50_000,  # Maximum number of iterations for cosine annealing
                 cosine_eta_min: float = 1e-5,  # Minimum learning rate for cosine annealing
                 
                 # gamma parameters
                 gamma_lb: float = 1e-6,
                 gamma_ub: float = 0.1):

        self.data_file = data_file
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_scale = loss_scale
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.feature_type = feature_type if feature_type is not None else 'identity'
        self.fourier_bands = fourier_bands
        self.fourier_sigma = fourier_sigma
        self.use_rwf = use_rwf
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        self.use_lbfgs = use_lbfgs
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_iter_per_step = lbfgs_max_iter_per_step
        self.lbfgs_history_size = lbfgs_history_size
        self.save_outputs = save_outputs
        self.save_numpy_outputs = save_numpy_outputs
        self.output_base_dir = output_base_dir
        self.free_energy_output_dir = free_energy_output_dir
        self.epsilon_output_dir = epsilon_output_dir
        self.plot_output_dir = plot_output_dir
        self.csv_filename = csv_filename
        self.plot_free_energy = plot_free_energy
        self.show_plots = show_plots
        self.seed = seed
        self.device_id = device_id
        # LR decay settings
        self.use_lr_decay = use_lr_decay
        self.lr_decay_gamma = lr_decay_gamma
        self.lr_decay_step = lr_decay_step
        # Cosine annealing settings
        self.use_cosine_annealing = use_cosine_annealing
        self.cosine_T_max = cosine_T_max
        self.cosine_eta_min = cosine_eta_min
        self.gamma_lb = gamma_lb
        self.gamma_ub = gamma_ub
    
    def get_activation_class(self) -> nn.Module:
        return getattr(nn, self.activation)

    def get_feature_modules(self) -> List[FeatureModule]:
        """Create a list of feature modules based on configuration."""
        modules = []
        if self.feature_type == 'fourier':
            modules.append(FourierFeature(in_dim=1, num_bands=self.fourier_bands, sigma=self.fourier_sigma, include_input=True))
        elif self.feature_type == 'identity':
            modules.append(IdentityFeature())
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        return modules


# ──────────────────── TRAINER ─────────────────────────────────────────────────

class SpinnsTrainer:
    """Main training class for learning free energy from Cahn-Hilliard data."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Setup device
        self.device = torch.device(
            f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize placeholders
        self.network_model = None
        self.optimizer = None
        
        self.data = None
        self.nn_output = None
        self.true_output= None
        
        # Training history
        self.loss_history = []
        self.start_time = None
        
        # Loss scale from config
        self.loss_scale = config.loss_scale
        
    def load_and_prepare_data(self):
        """Load and normalize training data."""
        # Load data
        data = np.load(self.config.data_file).astype(np.float64)
        x_vals = np.linspace(0, 1, data.shape[1])
        
        # Convert to torch
        self.data = torch.from_numpy(data).double()
        T, N = self.data.shape

        return
    
    def build_models(self, initial_gamma: float):
        """Build neural network models."""
        # Free energy model with optional RWF
        self.network_model = Network(
            input_dim=30,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.get_activation_class(),
            feature_modules=self.config.get_feature_modules(),
            use_rwf=self.config.use_rwf,
            rwf_mu=self.config.rwf_mu,
            rwf_sigma=self.config.rwf_sigma
        ).to(self.device).double()

        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.network_model = nn.DataParallel(self.network_model)
    
    def setup_optimizer(self):
        """Setup Adam optimizer."""
        params = [
            {'params': self.network_model.parameters()},
        ]
        self.mse = nn.MSELoss()
        self.rmse = RMSELoss()
        
        if self.config.learning_rate is not None:
            self.optimizer = optim.Adam(params, lr=self.config.learning_rate)
        else:
            self.optimizer = optim.Adam(params)
        
        # Optionally add LR scheduler
        self.scheduler = None
        if getattr(self.config, 'use_lr_decay', False):
            # Create ExponentialLR scheduler; will be stepped once per epoch in train_adam
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.lr_decay_gamma)
            print(f"Exponential LR decay enabled: gamma={self.config.lr_decay_gamma}, step_every={self.config.lr_decay_step} epochs")
        elif getattr(self.config, 'use_cosine_annealing', False):
            # Create CosineAnnealingLR scheduler; will be stepped once per epoch in train_adam
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.cosine_T_max, 
                eta_min=self.config.cosine_eta_min
            )
            print(f"Cosine annealing LR enabled: T_max={self.config.cosine_T_max}, eta_min={self.config.cosine_eta_min}")
    

    
    def train_adam(self):
        """Train with Adam optimizer"""
        print("Starting Adam training...")
        
        for epoch in range(self.config.epochs):
            # print(f"Epoch {epoch+1}/{self.config.epochs}", end='\r')
            # Forward pass
            phi_pred = self.rollout_dynamics(self.fixed_phi_1)
            # Loss between phi_1 and phi_2
            phi_loss = self.mse(phi_pred, self.fixed_phi_true) * self.loss_scale
            # Epsilon loss (relative error to true epsilon)
            #gamma_pred = self.gamma_model(self.gamma_lb, self.gamma_ub).squeeze()
            #loss_eps = torch.abs(torch.sqrt(gamma_pred) - self.config.true_epsilon)
            
            # Total loss (currently only using phi_loss for backprop)
            total_loss = phi_loss  # + loss_eps if you want to include epsilon loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Track loss history
            self.loss_history.append(total_loss.item())

            # Step LR scheduler if configured
            if getattr(self, 'scheduler', None) is not None:
                if getattr(self.config, 'use_lr_decay', False):
                    # Step exponential decay once every `lr_decay_step` epochs (default 1 -> every epoch)
                    if ((epoch + 1) % max(1, self.config.lr_decay_step)) == 0:
                        self.scheduler.step()
                elif getattr(self.config, 'use_cosine_annealing', False):
                    # Step cosine annealing every epoch
                    self.scheduler.step()
                
                # Optional: record or print new lr (show the lr of the first param group)
                cur_lr = self.optimizer.param_groups[0]['lr']
                '''if (epoch + 1) % 1000 == 0:
                    print(f"Epoch {epoch+1}: current learning rate = {cur_lr:.3e}")'''
            
            '''if (epoch + 1) % 100 == 0 or epoch == 0:
                # evaluate cloned model after bias correction, does not change training model
                f_l2_err_bc, eps_l2_err_bc = self._eval_clone_with_bias()

                # store the clone-only metrics: [epoch, f_L2_biascorr, eps_biascorr]
                self.l2_errors.append([epoch + 1, f_l2_err_bc, eps_l2_err_bc])'''

                #print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss.item():.3e}, f_L2_err_bc: {f_l2_err_bc:.4e}, eps_L2_err_bc: {eps_l2_err_bc:.4e}")
    
    def train_lbfgs(self):
        """Fine-tune with L-BFGS optimizer."""
        if not self.config.use_lbfgs:
            return
        
        print("Starting L-BFGS fine-tuning...")
        
        # Compute initial loss
        with torch.no_grad():
            phi_pred_init = self.rollout_dynamics(self.fixed_phi_1)
            initial_loss = (self.mse(phi_pred_init, self.fixed_phi_true) * self.loss_scale).item()
            initial_epsilon = math.sqrt(self.gamma_model(self.gamma_lb, self.gamma_ub).squeeze().item())
        
        # Setup L-BFGS
        params = list(self.free_energy_model.parameters()) + list(self.gamma_model.parameters())
        lbfgs = optim.LBFGS(
            params,
            lr=1.0,
            max_iter=self.config.lbfgs_max_iter_per_step,
            history_size=self.config.lbfgs_history_size,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-15,
            tolerance_change=1e-15
        )
        
        self.free_energy_model.train()
        self.gamma_model.train()
        
        iteration_count = [0]
        
        def closure():
            lbfgs.zero_grad()
            phi_pred = self.rollout_dynamics(self.fixed_phi_1)
            loss = self.mse(phi_pred, self.fixed_phi_true) * self.loss_scale
            iteration_count[0] += 1
            loss.backward()
            return loss
        
        # Run L-BFGS steps
        num_lbfgs_steps = self.config.lbfgs_max_iter // self.config.lbfgs_max_iter_per_step
        final_loss = None
        
        # Track starting epoch for L-BFGS (after Adam training)
        lbfgs_start_epoch = self.config.epochs
        
        for step_idx in range(num_lbfgs_steps):
            final_loss = lbfgs.step(closure)

            # evaluate a clone with bias correction after this L-BFGS step
            try:
                f_l2_err_bc, eps_l2_err_bc = self._eval_clone_with_bias()
            except Exception as e:
                # fail-safe: if clone evaluation fails, record NaNs and continue
                f_l2_err_bc, eps_l2_err_bc = float('nan'), float('nan')

            # compute an epoch-like index for reporting. lbfgs_start_epoch is defined earlier.
            current_epoch = lbfgs_start_epoch + step_idx + 1

            # store clone metrics in the same structure you use for Adam checkpoints
            self.l2_errors.append([current_epoch, f_l2_err_bc, eps_l2_err_bc])

            #print(f"[LBFGS step {step_idx+1}/{num_lbfgs_steps}] final_loss: {float(final_loss):.3e}, f_L2_err_bc: {f_l2_err_bc:.4e}, eps_L2_err_bc: {eps_l2_err_bc:.4e}")
        
        # Report results
        final_epsilon = math.sqrt(self.gamma_model(self.gamma_lb, self.gamma_ub).squeeze().item())
        loss_improvement = initial_loss - float(final_loss)
        loss_reduction_percent = (loss_improvement / initial_loss) * 100 if initial_loss > 0 else 0
        
        print(f"[LBFGS] Initial loss:  {initial_loss:.3e}")
        print(f"[LBFGS] Final loss:    {float(final_loss):.3e}")
        print(f"[LBFGS] Loss improvement: {loss_improvement:.3e} ({loss_reduction_percent:.2f}% reduction)")
        print(f"[LBFGS] Initial epsilon: {initial_epsilon:.4e}")
        print(f"[LBFGS] Final epsilon:   {final_epsilon:.4e}")
    
    def apply_bias_correction(self):
        """Translate bias so that f(0) = 0."""
        with torch.no_grad():
            zero = torch.tensor([[self.zero_term]], device=self.device).double()
            f0 = self.free_energy_model(zero).item()
            
            # Find output layer and subtract bias
            for m in self.free_energy_model.modules():
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    m.bias.sub_(f0)
                    break
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate final model and compute metrics."""
        with torch.no_grad():
            # Training loss
            phi_pred_final = self.rollout_dynamics(self.fixed_phi_1)
            training_loss = (self.mse(phi_pred_final, self.fixed_phi_true) * self.loss_scale).item()
            
            # Evaluate free energy function
            phi_vis = torch.linspace(self.min_phi, self.max_phi, 100).unsqueeze(-1).to(self.device).double()
            f_pred = self.free_energy_model(phi_vis).squeeze().detach().cpu().numpy()
            
            # Get epsilon
            gamma_pred = self.gamma_model(self.gamma_lb, self.gamma_ub).squeeze().detach().cpu().numpy()
            eps = np.sqrt(gamma_pred)
            
            # Map back to original domain
            phi_final = np.linspace(self.min_phi, self.max_phi, 100)
            f_final = f_pred 
            
            # Get true free energy
            if 'poly' in self.config.data_file:
                f_true = PhysicsUtils.free_energy_poly(phi_final)
            else:
                f_true = PhysicsUtils.free_energy_log(phi_final, chi=2.5)
            
            # Compute relative L² error for free energy function
            d_phi = phi_final[1] - phi_final[0]
            num = np.sqrt(d_phi * np.sum((f_final - f_true)**2))
            den = np.sqrt(d_phi * np.sum(f_true**2))
            relative_l2_error = num / den
            f_percentage_error = 100 * relative_l2_error

            giulia_num = np.sqrt(d_phi * np.sum(((f_final/eps**2) - (f_true/0.05**2))**2))
            giulia_den = np.sqrt(d_phi * np.sum(((f_true/0.05**2))**2))
            giulia_error = giulia_num / giulia_den


            
            return {
                'training_loss': training_loss,
                'f_relative_l2_error': relative_l2_error,
                'f_percentage_error': f_percentage_error,
                'predicted_epsilon': eps,
                'true_epsilon': 0.05,
                'epsilon_relative_error': abs(eps - 0.05) / 0.05,
                'epsilon_error_percent': 100 * abs(eps - 0.05) / 0.05,
                'phi_final': phi_final,
                'f_final': f_final,
                'f_true': f_true,
                'giulia_error': giulia_error
            }
    
    def save_results(self, metrics: Dict[str, Any]):
        """Save training results to files."""
        if not self.config.save_outputs:
            return
        
        # Optionally save numpy outputs (free energy arrays and epsilon)
        if self.config.save_numpy_outputs:
            # Create output directories
            layers_str = 'x'.join(map(str, self.config.hidden_dims))
            
            # Use custom directories if provided, otherwise use default naming
            if self.config.free_energy_output_dir:
                free_energy_dir = self.config.free_energy_output_dir
            else:
                free_energy_dir = os.path.join(
                    self.config.output_base_dir,
                    f"f_poly/pairs_{self.config.batch_size}_{layers_str}_{self.config.activation}/noise_{self.config.noise}"
                )
            
            if self.config.epsilon_output_dir:
                eps_dir = self.config.epsilon_output_dir
            else:
                eps_dir = os.path.join(
                    self.config.output_base_dir,
                    f"eps_poly/pairs_{self.config.batch_size}_{layers_str}_{self.config.activation}/noise_{self.config.noise}"
                )
            
            os.makedirs(free_energy_dir, exist_ok=True)
            os.makedirs(eps_dir, exist_ok=True)
            
            # Save free energy function
            final_array = np.column_stack((metrics['phi_final'], metrics['f_final']))
            np.save(os.path.join(free_energy_dir, f"seed_{self.config.seed}.npy"), final_array)
            
            # Save epsilon
            np.save(os.path.join(eps_dir, f"seed_{self.config.seed}.npy"), metrics['predicted_epsilon'])
        
        # Save metrics to CSV
        elapsed_time = time.time() - self.start_time
        
        df = pd.DataFrame({
            'data_file': [self.config.data_file],
            'loss_scale': [self.loss_scale],
            'activation': [self.config.activation],
            'feature_type': [self.config.feature_type],
            'fourier_bands': [self.config.fourier_bands],
            'fourier_sigma': [self.config.fourier_sigma],
            'noise': [self.config.noise],
            'seed': [self.config.seed],
            'batch_size': [self.config.batch_size],
            'learning_rate': [self.config.learning_rate],
            'hidden_dims': [str(self.config.hidden_dims)],
            'use_rwf': [self.config.use_rwf],
            'rwf_mu': [self.config.rwf_mu if self.config.use_rwf else None],
            'rwf_sigma': [self.config.rwf_sigma if self.config.use_rwf else None],
            'use_lr_decay': [self.config.use_lr_decay],
            'lr_decay_gamma': [self.config.lr_decay_gamma if self.config.use_lr_decay else None],
            'lr_decay_step': [self.config.lr_decay_step if self.config.use_lr_decay else None],
            'use_cosine_annealing': [self.config.use_cosine_annealing],
            'cosine_T_max': [self.config.cosine_T_max if self.config.use_cosine_annealing else None],
            'cosine_eta_min': [self.config.cosine_eta_min if self.config.use_cosine_annealing else None],
            'Training Loss (MSE)': [metrics['training_loss']],
            'f Relative L2 Error': [metrics['f_relative_l2_error']],
            'f Percentage Error (%)': [metrics['f_percentage_error']],
            'initial_Epsilon': [math.sqrt(self.initial_gamma)],
            'gamma_lb': [self.config.gamma_lb],
            'gamma_ub': [self.config.gamma_ub],
            'Predicted Epsilon': [metrics['predicted_epsilon']],
            'True Epsilon': [metrics['true_epsilon']],
            'Epsilon Relative Error': [metrics['epsilon_relative_error']],
            'Epsilon Error (%)': [metrics['epsilon_error_percent']],
            'Giulia Error': [metrics['giulia_error']],
            'time_seconds': [elapsed_time]
        })
        
        results_file = os.path.join(self.config.output_base_dir, self.config.csv_filename)
        if not os.path.exists(results_file):
            df.to_csv(results_file, index=False)
        else:
            df.to_csv(results_file, mode='a', header=False, index=False)
        
        # Save epoch-wise error tracking
        if self.l2_errors:
            # Debug: print array size
            print(f"L2 errors array size: {len(self.l2_errors)} entries")
            
            # Convert to numpy array - format: [epoch, f_L2_error, eps_L2_error]
            l2_errors_array = np.array(self.l2_errors)  # Shape: (n_recorded_epochs, 3)
            
            error_file = os.path.join(self.config.output_base_dir, f"whatever.npy")
            np.save(error_file, l2_errors_array)
            print(f"Saved L2 errors to: {error_file}")
            print(f"Array shape: {l2_errors_array.shape} - columns: [epoch, f_L2_error, eps_L2_error]")
        else:
            print(f"Warning: L2 errors array is empty: {len(self.l2_errors)} entries")
    
    def print_results(self, metrics: Dict[str, Any]):
        """Print training results."""
        elapsed_time = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"TRAINING RESULTS:")
        print(f"Data file:                   {self.config.data_file}")
        print(f"scaling:                      {self.loss_scale:.3e}")
        print(f"{'='*60}")
        print(f"Training loss (MSE):  {metrics['training_loss']:.3e}")
        print(f"Actual relative L² error:     {metrics['f_relative_l2_error']:.3e}")
        print(f"Actual relative L² error (%): {metrics['f_percentage_error']:.2f}%")
        print(f"Predicted epsilon:            {metrics['predicted_epsilon']:.6e}")
        print(f"True epsilon:                 {metrics['true_epsilon']:.6e}")
        print(f"Epsilon relative error:       {metrics['epsilon_relative_error']:.6e} ({metrics['epsilon_error_percent']:.2f}%)")
        print(f"Total training time:          {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")
    
    def plot_free_energy(self, metrics: Dict[str, Any], save_path: Optional[str] = None, show: bool = True):
        """
        Plot predicted vs actual free energy function.
        
        Args:
            metrics: Dictionary containing evaluation metrics (from evaluate())
            save_path: Optional path to save the figure. If None, uses default naming.
            show: Whether to display the plot interactively (default: True)
        """
        phi = metrics['phi_final']
        f_pred = metrics['f_final']
        f_true = metrics['f_true']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Comparison
        ax1.plot(phi, f_true, 'b-', linewidth=2, label='True Free Energy', alpha=0.8)
        ax1.plot(phi, f_pred, 'r--', linewidth=2, label='Predicted Free Energy', alpha=0.8)
        ax1.set_xlabel('φ', fontsize=12)
        ax1.set_ylabel('f(φ)', fontsize=12)
        ax1.set_title(f'Free Energy Comparison\nRelative L² Error: {metrics["f_relative_l2_error"]:.3e}', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Error
        error = f_pred - f_true
        ax2.plot(phi, error, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('φ', fontsize=12)
        ax2.set_ylabel('Error: f_pred - f_true', fontsize=12)
        ax2.set_title('Pointwise Error', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Add configuration info
        config_text = (
            f"Config: {self.config.activation}, "
            f"RWF={self.config.use_rwf}, "
            f"Seed: {self.config.seed}, "
            f"Noise: {self.config.noise:.2e}, "
            f"ε_pred: {metrics['predicted_epsilon']:.4e}, "
            f"ε_true: {metrics['true_epsilon']:.4e}, "
            f"γ_lb: {self.config.gamma_lb:.4e}, "
            f"γ_ub: {self.config.gamma_ub:.4e}, "
            f"epochs: {self.config.epochs}"
        )
        fig.suptitle(config_text, fontsize=10, y=1.02)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None and self.config.save_outputs:
            # Create default save path
            layers_str = 'x'.join(map(str, self.config.hidden_dims))
            
            # Use custom plot directory if provided
            if self.config.plot_output_dir:
                plot_dir = self.config.plot_output_dir
            else:
                plot_dir = os.path.join(
                    self.config.output_base_dir,
                    f"f_plots_fourier_rwf/pairs_{self.config.batch_size}_{layers_str}_{self.config.activation}/noise_{self.config.noise}"
                )
            
            os.makedirs(plot_dir, exist_ok=True)
            save_path = os.path.join(plot_dir, f"free_energy_{self.config.seed}_rwf_{self.config.use_rwf}_gamma_{self.config.gamma_lb}_{self.config.gamma_ub}.png")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def train(self):
        """Main training loop."""
        self.start_time = time.time()
        
        # Load data and setup
        print("Loading and preparing data...")
        initial_gamma = self.load_and_prepare_data()
        
        print("Building models...")
        self.build_models(initial_gamma)
        
        print("Setting up optimizer...")
        self.setup_optimizer()
        
        # Train with Adam
        self.train_adam()
        
        # Fine-tune with L-BFGS
        self.train_lbfgs()
        
        # Apply bias correction
        self.apply_bias_correction()
        
        # Evaluate and save
        print("Evaluating final model...")
        metrics = self.evaluate()
        
        self.save_results(metrics)
        self.print_results(metrics)
        
        # Plot if requested
        if self.config.plot_free_energy:
            self.plot_free_energy(metrics, show=self.config.show_plots)
        
        return metrics


# ──────────────────── PARALLEL EXECUTION ─────────────────────────────────────

def run_single_experiment(params):
    """
    Run a single experiment with the given parameters.
    This function is designed to be called in parallel.
    
    Args:
        params: Tuple of (seed, activation, use_rwf, rwf_mu, rwf_sigma, 
                         causality_eps, causality_segs, use_gn, gn_alpha, gn_lr, csv_filename)
    
    Returns:
        Dictionary with experiment results
    """
    (seed, activation, use_rwf, rwf_mu, rwf_sigma, 
     csv_filename) = params
    
    try:
        config = TrainingConfig(
            data_file="data_poly.npy",
            noise=0.0,
            learning_rate=1e-4,
            batch_size=20,
            epochs=100_000,
            hidden_dims=[20, 20],
            activation=activation,
            feature_type='fourier',
            use_rwf=use_rwf,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
            csv_filename=csv_filename,
            seed=seed,
            gamma_lb=1e-4,
            gamma_ub=0.1
        )
        
        trainer = SpinnsTrainer(config)
        metrics = trainer.train()
        
        return {
            'status': 'success',
            'seed': seed,
            'activation': activation,
            'metrics': metrics
        }
    except Exception as e:
        return {
            'status': 'failed',
            'seed': seed,
            'activation': activation,
            'error': str(e)
        }


# ──────────────────── MAIN ────────────────────────────────────────────────────
def main(noise, seed, activation, loss_scale, data_file, RWF_use, RWF_mu, RWF_sigma, gamma_lb, gamma_ub, batch_size, cuda_device, use_lr_decay=False, lr_decay_gamma=0.99998, lr_decay_step=1, use_cosine_annealing=False,
          cosine_T_max=50_000, cosine_eta_min=1e-5 ,learning_rate=1e-2, hidden_dims=[20, 20], feature_type=['fourier'], fourier_bands=2, fourier_sigma=0.5):
    """Example usage of the OO framework."""
    if 'poly' in data_file:
        dir_f = 'poly'
    else:
        dir_f = 'log'
    
    # Example: use multiple features (Fourier + Polynomial + Identity)
    config = TrainingConfig(
        data_file = data_file,
        noise = noise,
        learning_rate = learning_rate,
        batch_size = batch_size,
        epochs = 20_000,
        loss_scale = loss_scale,
        hidden_dims = hidden_dims,
        activation = activation,
        feature_type = feature_type,
        fourier_bands = fourier_bands,
        fourier_sigma = fourier_sigma,
        use_rwf = RWF_use,  # Set to True to enable Random Weight Factorization
        rwf_mu = RWF_mu,
        rwf_sigma = RWF_sigma,
        use_lbfgs = True,
        lbfgs_max_iter = 30_000,
        lbfgs_max_iter_per_step = 10,
        save_outputs = True,
        save_numpy_outputs=True,
        output_base_dir = ".",  # Base directory for outputs
        free_energy_output_dir = f"f_{dir_f}_v2/batch_size_{batch_size}",  # Custom directory for free energy (None = use default)
        epsilon_output_dir = f"epsilon_{dir_f}_v2/batch_size_{batch_size}",  # Custom directory for epsilon (None = use default)
        plot_output_dir = f"plots_{dir_f}_v2/batch_size_{batch_size}",  # Custom directory for plots (None = use default)
        # csv_filename = f"{dir_f}_results.csv",  # Output CSV filename
        csv_filename=f"arch_sens_results.csv",
        plot_free_energy = False,  # Set to True to generate plots
        show_plots = False,  # Set to True to display plots interactively
        seed = seed,
        use_lr_decay = use_lr_decay,  # Enable exponential LR decay
        lr_decay_gamma = lr_decay_gamma,  # Exponential decay factor
        lr_decay_step = lr_decay_step,  # Steps between decay
        use_cosine_annealing = use_cosine_annealing,  # Enable cosine annealing
        cosine_T_max = cosine_T_max,  # T_max for cosine annealing
        cosine_eta_min = cosine_eta_min,  # Minimum LR for cosine annealing
        gamma_lb = gamma_lb,
        gamma_ub = gamma_ub
    )

    # Create trainer and run
    trainer = SpinnsTrainer(config)
    metrics = trainer.train()

    return trainer, metrics

# ─────RUN──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(1e-3, 0, 'SiLU', 1e4, "data_log_chi_2.5.npy", False, 0.5, 0.15, 1e-4, 0.1, 20, False, 
        use_lr_decay=True, lr_decay_gamma=0.9998, lr_decay_step=1, 
        use_cosine_annealing=False, cosine_T_max=50_000, cosine_eta_min=1e-5 , learning_rate=1e-2,
            hidden_dims=[20, 20], feature_type='identity', fourier_bands=2, fourier_sigma=0.5)