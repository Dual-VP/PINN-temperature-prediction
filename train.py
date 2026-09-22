import torch
import torch.optim as optim
from dataset import ThermalDataset
from model import HeatPINN


def train_pinn():
    # 1. Configuration du device (essentiel pour le cluster SLURM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {device}")

    # 2. Génération des données
    print("Génération des points de collocation...")
    dataset = ThermalDataset(N_ic=2000, N_bc=2000, N_res=10000)
    ic_data_cpu, bc_data_cpu, res_data_cpu = dataset.generate_data()

    # Fonction utilitaire pour envoyer les tuples de tenseurs sur le GPU/CPU
    def to_device(data_tuple):
        return tuple(d.to(device) for d in data_tuple)

    ic_data = to_device(ic_data_cpu)
    bc_data = to_device(bc_data_cpu)
    res_data = to_device(res_data_cpu)

    # 3. Initialisation du modèle
    model = HeatPINN(alpha_adim=dataset.lambda_adim).to(device)

    # Poids initiaux de la fonction de perte (peuvent être ajustés plus tard)
    loss_weights = (1.0, 1.0, 1.0)  # (w_ic, w_bc, w_res)

    # ==========================================
    # PHASE 1 : Optimiseur ADAM
    # ==========================================
    epochs_adam = 3
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n--- Début de la phase 1 : ADAM ({epochs_adam} epochs) ---")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()

        # Calcul de la perte
        loss, (l_ic, l_bc, l_res) = model.compute_loss(ic_data, bc_data, res_data, loss_weights)

        # Rétropropagation
        loss.backward()
        optimizer_adam.step()

        if epoch % 500 == 0:
            print(
                f"Adam Epoch {epoch:04d} | Loss Globale: {loss.item():.4e} | IC: {l_ic:.4e} | BC: {l_bc:.4e} | EDP: {l_res:.4e}")

    # ==========================================
    # PHASE 2 : Optimiseur L-BFGS
    # ==========================================
    # L-BFGS nécessite une fonction "closure" qui recalcule la perte à chaque itération interne
    epochs_lbfgs = 5
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50
    )

    print(f"\n--- Début de la phase 2 : L-BFGS ({epochs_lbfgs} itérations max) ---")

    # Compteur pour l'affichage dans la closure
    lbfgs_iter = 0

    def closure():
        nonlocal lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss, (l_ic, l_bc, l_res) = model.compute_loss(ic_data, bc_data, res_data, loss_weights)
        loss.backward()

        if lbfgs_iter % 100 == 0:
            print(
                f"L-BFGS Iter {lbfgs_iter:04d} | Loss Globale: {loss.item():.4e} | IC: {l_ic:.4e} | BC: {l_bc:.4e} | EDP: {l_res:.4e}")
        lbfgs_iter += 1
        return loss

    for epoch in range(epochs_lbfgs):
        optimizer_lbfgs.step(closure)

    # 4. Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), "heat_pinn_model.pth")
    print("\nEntraînement terminé. Modèle sauvegardé sous 'heat_pinn_model.pth'.")