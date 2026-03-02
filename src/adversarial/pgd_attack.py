"""
PGD Attack for SafetyGAT.

Implements Projected Gradient Descent to perturb node features and test robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack:
    """
    Projected Gradient Descent Attack on Graph Node Features.
    """
    def __init__(self, model, epsilon=0.1, alpha=0.01, num_steps=10, device='cpu'):
        """
        Args:
            model: The SafetyGAT model to attack.
            epsilon: Maximum perturbation norm (L-infinity).
            alpha: Step size.
            num_steps: Number of PGD steps.
            device: 'cpu' or 'cuda'.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def attack(self, x, edge_index, batch, target):
        """
        Generate adversarial examples.
        
        Args:
            x: Node features (N, D)
            edge_index: Edge list
            batch: Batch vector
            target: Target labels (Batch, ) - We want to maximize loss against this.
            
        Returns:
            perturbed_x: Adversarial node features
        """
        self.model.eval() # Ensure dropout/batchnorm are in eval mode
        
        # Detach original x and create a copy for perturbation
        x_adv = x.clone().detach().to(self.device).requires_grad_(True)
        x_orig = x.clone().detach().to(self.device)
        
        for t in range(self.num_steps):
            # Forward pass
            outputs = self.model(x_adv, edge_index, batch=batch)
            pred = outputs['safety_pred']
            
            # Calculate loss (we want to MAXIMIZE this loss to find adversarial example)
            # Actually, standard PGD maximizes loss. 
            # If target is "Safe" (1) and we want to fool it to "Unsafe" (0), maximizing BCE(1) does that.
            loss = self.bce_loss(pred, target)
            
            # Zero gradients
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
                
            # Backward pass (compute gradient w.r.t input features)
            loss.backward()
            
            # Update x_adv (Gradient Ascent)
            data_grad = x_adv.grad.data
            x_adv.data = x_adv.data + self.alpha * data_grad.sign()
            
            # Projection (clip perturbation to epsilon ball around x_orig)
            eta = torch.clamp(x_adv.data - x_orig.data, -self.epsilon, self.epsilon)
            x_adv.data = x_orig.data + eta
            
            # Reset gradient requirement for next step (except last)
            if t < self.num_steps - 1:
                x_adv = x_adv.clone().detach().requires_grad_(True)
                
        return x_adv.detach()

if __name__ == "__main__":
    print("PGD Attack module ready.")
