import torch
import numpy as np
import matplotlib.pyplot as plt
from model import HeatPINN
from dataset import ThermalDataset
from solver import FiniteDifferenceSolver


def evaluate_model(model_path="heat_pinn_model.pth", eval_time=1800.0):
    """
    Compare les prédictions du PINN avec la méthode des Différences Finies.
    eval_time : temps en secondes où l'on souhaite observer la diffusion (ex: 1800s = 30min).
    """
    # 1. Instanciation des objets de base
    dataset = ThermalDataset()
    device = torch.device("cpu")  # L'évaluation se fait très bien sur CPU

    model = HeatPINN(alpha_adim=dataset.lambda_adim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Mode inférence

    # 2. Obtention de la Vérité Terrain (FDM)
    print(f"Calcul FDM jusqu'à t = {eval_time} s...")
    fdm = FiniteDifferenceSolver(nx=100, ny=100)  # Grille de 100x100
    X_fdm, Y_fdm, T_fdm = fdm.solve(eval_time)

    # 3. Prédiction par le PINN sur la même grille
    print("Prédiction PINN...")
    # Aplatissement des grilles pour le réseau
    x_tensor = torch.tensor(X_fdm.flatten()[:, None], dtype=torch.float32)
    y_tensor = torch.tensor(Y_fdm.flatten()[:, None], dtype=torch.float32)

    # Adimensionnement du temps cible pour le réseau (t* = t / t_max)
    t_adim = eval_time / dataset.t_max
    t_tensor = torch.full_like(x_tensor, t_adim)

    # Inférence
    with torch.no_grad():
        T_pred_flat = model(x_tensor, y_tensor, t_tensor).numpy()

    T_pinn = T_pred_flat.reshape(X_fdm.shape)

    # 4. Calcul de l'erreur
    # Erreur relative L2 (norme standard d'évaluation physique)
    l2_error = np.linalg.norm(T_pinn - T_fdm) / np.linalg.norm(T_fdm)
    mse = np.mean((T_pinn - T_fdm) ** 2)

    print(f"--- Résultats à t={eval_time}s ---")
    print(f"MSE : {mse:.4e}")
    print(f"Erreur L2 relative : {l2_error * 100:.2f}%")

    # 5. Visualisation comparative
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmax = 1.0  # Echelle adimensionnée max

    c1 = axes[0].contourf(X_fdm, Y_fdm, T_fdm, levels=50, cmap='inferno', vmin=0, vmax=vmax)
    axes[0].set_title("Vérité Terrain (FDM)")
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(X_fdm, Y_fdm, T_pinn, levels=50, cmap='inferno', vmin=0, vmax=vmax)
    axes[1].set_title("Prédiction PINN")
    plt.colorbar(c2, ax=axes[1])

    # Carte d'erreur absolue
    error_map = np.abs(T_pinn - T_fdm)
    c3 = axes[2].contourf(X_fdm, Y_fdm, error_map, levels=50, cmap='viridis')
    axes[2].set_title(f"Erreur Absolue (L2: {l2_error * 100:.1f}%)")
    plt.colorbar(c3, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()