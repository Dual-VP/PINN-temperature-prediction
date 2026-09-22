import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_thermal_heatmap(x, y, T, title="Carte Thermique (Caméra infrarouge)"):
    """
    Affiche une carte de chaleur continue (style caméra thermique) à partir de points de collocation.
    Générique : s'adapte aux limites min/max des points fournis et masque l'extérieur.

    Args:
        x (torch.Tensor ou np.ndarray): Coordonnées x des points
        y (torch.Tensor ou np.ndarray): Coordonnées y des points
        T (torch.Tensor ou np.ndarray): Températures associées aux points
        title (str): Titre du graphique
    """
    # 1. Conversion en tableaux numpy 1D
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        T_np = T.detach().cpu().numpy().flatten()
    else:
        x_np, y_np, T_np = x.flatten(), y.flatten(), T.flatten()

    # 2. Création d'une grille fine s'adaptant à la géométrie (min/max des coordonnées)
    # Une résolution de 300x300 donne un bel effet lissé "caméra thermique"
    grid_res = 300
    xi = np.linspace(x_np.min(), x_np.max(), grid_res)
    yi = np.linspace(y_np.min(), y_np.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)

    # 3. Interpolation des données non structurées vers la grille
    # method='cubic' permet un lissage très réaliste des gradients de température.
    # Les zones hors de la forme (si géométrie complexe) vaudront NaN et seront invisibles.
    Ti = griddata((x_np, y_np), T_np, (X, Y), method='cubic')

    # 4. Affichage avec Matplotlib
    plt.figure(figsize=(8, 6))

    # contourf génère des zones de chaleur continues. cmap='inferno' simule la vision thermique.
    contour = plt.contourf(X, Y, Ti, levels=100, cmap='inferno', vmin=0.0, vmax=1.0)

    # Configuration esthétique
    cbar = plt.colorbar(contour)
    cbar.set_label("Température (adimensionnée)", rotation=270, labelpad=15)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Position x")
    plt.ylabel("Position y")

    # Garantir que les proportions de la pièce ne sont pas déformées
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()