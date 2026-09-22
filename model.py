import torch
import torch.nn as nn


class HeatPINN(nn.Module):
    def __init__(self, alpha_adim):
        super(HeatPINN, self).__init__()
        self.alpha = alpha_adim  # Diffusivité thermique adimensionnée (lambda)

        # Architecture du Perceptron Multicouche (MLP)
        # Entrée : 3 neurones (x, y, t) -> Sortie : 1 neurone (Température T)
        self.network = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, t):
        """Prédiction classique du réseau T = f(x, y, t)"""
        # Concaténation des entrées (N, 1) en un tenseur (N, 3)
        inputs = torch.cat([x, y, t], dim=1)
        return self.network(inputs)

    def compute_gradients(self, y_pred, x_input):
        """
        Calcule la dérivée de y_pred par rapport à x_input via dérivation automatique.
        create_graph=True est obligatoire pour permettre les dérivées d'ordre supérieur (Hessien).
        """
        grads = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_input,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        return grads

    def compute_pde_residual(self, x, y, t):
        """
        Évalue le respect de l'équation de la chaleur 2D instationnaire.
        Retourne le résidu de l'EDP qui devrait tendre vers 0.
        """
        # On s'assure que les tenseurs requièrent un gradient pour autograd
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        # 1. Prédiction de la température
        T = self.forward(x, y, t)

        # 2. Calcul des dérivées premières
        T_t = self.compute_gradients(T, t)
        T_x = self.compute_gradients(T, x)
        T_y = self.compute_gradients(T, y)

        # 3. Calcul des dérivées secondes spatiales
        T_xx = self.compute_gradients(T_x, x)
        T_yy = self.compute_gradients(T_y, y)

        # 4. Résidu de l'équation de la chaleur (EDP)
        # L'objectif du PINN est de minimiser cette valeur absolue
        residual = T_t - self.alpha * (T_xx + T_yy)

        return residual

    def compute_loss(self, ic_data, bc_data, res_data, weights=(1.0, 1.0, 1.0)):
        """
        Calcule la perte multi-objectif globale (IC, BC, EDP).
        """
        w_ic, w_bc, w_res = weights
        mse = nn.MSELoss()

        # 1. Perte de la Condition Initiale (IC)
        x_ic, y_ic, t_ic, T_ic_target = ic_data
        T_ic_pred = self.forward(x_ic, y_ic, t_ic)
        loss_ic = mse(T_ic_pred, T_ic_target)

        # 2. Perte des Conditions aux Limites (BC)
        x_bc, y_bc, t_bc, T_bc_target = bc_data
        T_bc_pred = self.forward(x_bc, y_bc, t_bc)
        loss_bc = mse(T_bc_pred, T_bc_target)

        # 3. Perte du Résidu Physique (EDP)
        x_res, y_res, t_res = res_data
        residual = self.compute_pde_residual(x_res, y_res, t_res)
        # Le résidu doit être égal à 0
        loss_res = mse(residual, torch.zeros_like(residual))

        # Perte globale pondérée
        total_loss = w_ic * loss_ic + w_bc * loss_bc + w_res * loss_res

        return total_loss, (loss_ic.item(), loss_bc.item(), loss_res.item())