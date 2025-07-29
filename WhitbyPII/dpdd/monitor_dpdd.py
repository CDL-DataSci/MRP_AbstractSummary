# utils/monitor_dpdd.py

import numpy as np
import torch

class DPDDMonitor:
    def __init__(self, noise_multiplier=1.0, batch_size=4, sample_rate=0.01):
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.grad_norms = []

    def record_gradients(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = np.sqrt(total_norm)
        self.grad_norms.append(total_norm)

    def compute_dpdd_score(self):
        # We define DPDD as sensitivity * noise_multiplier
        if not self.grad_norms:
            return None
        avg_norm = np.mean(self.grad_norms)
        dpdd_score = avg_norm * self.noise_multiplier
        return dpdd_score

    def reset(self):
        self.grad_norms = []
