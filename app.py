import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import HeatPINN
from dataset import ThermalDataset
from solver import FiniteDifferenceSolver

# ==========================================
# 1. Initialisation globale (chargée une seule fois)
# ==========================================
dataset = ThermalDataset()
device = torch.device("cpu")  # L'inférence interactive se fait très bien sur CPU

# Instanciation et chargement du modèle
model = HeatPINN(alpha_adim=dataset.lambda_adim).to(device)
try:
    model.load_state_dict(torch.load("heat_pinn_model.pth", map_location=device))
    print("Modèle PINN chargé avec succès.")
except FileNotFoundError:
    print("Attention : 'heat_pinn_model.pth' introuvable. Le PINN utilisera des poids aléatoires.")
model.eval()


# ==========================================
# 2. Fonction de mise à jour du Dashboard
# ==========================================
def update_dashboard(t_real):
    """
    Génère la figure Matplotlib à chaque mouvement du slider temporel.
    t_real : Temps en secondes.
    """
    # Résolution FDM (Vérité terrain)
    # Grille réduite (ex: 60x60) pour garantir la fluidité de l'interface Gradio
    fdm = FiniteDifferenceSolver(nx=60, ny=60)
    X_fdm, Y_fdm, T_fdm = fdm.solve(t_real)

    # Prédiction PINN
    x_tensor = torch.tensor(X_fdm.flatten()[:, None], dtype=torch.float32)
    y_tensor = torch.tensor(Y_fdm.flatten()[:, None], dtype=torch.float32)
    t_adim = t_real / dataset.t_max
    t_tensor = torch.full_like(x_tensor, t_adim)

    with torch.no_grad():
        T_pred = model(x_tensor, y_tensor, t_tensor).numpy()
    T_pinn = T_pred.reshape(X_fdm.shape)

    # Calcul de l'erreur
    error_map = np.abs(T_pinn - T_fdm)

    # Génération de la figure (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = 1.0  # Echelle max (1.0 = 80°C)

    c1 = axes[0].contourf(X_fdm, Y_fdm, T_fdm, levels=50, cmap='inferno', vmin=0, vmax=vmax)
    axes[0].set_title(f"Vérité Terrain (FDM) à t={t_real}s")
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(X_fdm, Y_fdm, T_pinn, levels=50, cmap='inferno', vmin=0, vmax=vmax)
    axes[1].set_title(f"Prédiction PINN à t={t_real}s")
    fig.colorbar(c2, ax=axes[1])

    c3 = axes[2].contourf(X_fdm, Y_fdm, error_map, levels=50, cmap='viridis')
    axes[2].set_title("Carte d'Erreur Absolue")
    fig.colorbar(c3, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    return fig


# ==========================================
# 3. Interface Gradio
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌡️ Démonstrateur Interactif : PINN vs Différences Finies")
    gr.Markdown(
        "Ce tableau de bord compare la diffusion thermique modélisée par des méthodes numériques classiques (FDM) et prédite par un réseau de neurones informé par la physique (PINN).")

    with gr.Row():
        # Slider temporel : de 0 à 1 heure (3600s) avec des pas de 1 minute
        time_slider = gr.Slider(minimum=0, maximum=3600, step=60, value=0, label="Curseur temporel (secondes)")

    plot_output = gr.Plot()

    # Interactivité : mise à jour du graphe quand le slider bouge
    time_slider.change(fn=update_dashboard, inputs=time_slider, outputs=plot_output)

    # Affichage initial au lancement
    demo.load(fn=update_dashboard, inputs=time_slider, outputs=plot_output)

if __name__ == "__main__":
    # Lancement du serveur local
    demo.launch(server_name="0.0.0.0", share=False)