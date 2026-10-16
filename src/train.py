import torch
import torch.optim as optim
from dataset import ThermalDataset
from model import HeatPINN
from utils import plot_loss


def train_pinn():
    # 1. Configuration du device (essentiel pour le cluster SLURM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {torch.cuda.get_device_name(device)}")

    # 2. Génération des données
    N_ic, N_bc, N_res = 2000, 2000, 10000
    dataset = ThermalDataset(N_ic=N_ic, N_bc=N_bc, N_res=N_res)

    train_ic, train_bc, train_res = dataset.get_train_data()
    val_ic, val_bc, val_res = dataset.get_val_data()

    # Fonction utilitaire pour envoyer les tuples de tenseurs sur le GPU/CPU
    def to_device(data_tuple):
        return tuple(d.to(device) for d in data_tuple)

    ic_data_train = to_device(train_ic)
    bc_data_train = to_device(train_bc)
    res_data_train = to_device(train_res)

    ic_data_val = to_device(val_ic)
    bc_data_val = to_device(val_bc)
    res_data_val = to_device(val_res)

    # 3. Initialisation du modèle
    model = HeatPINN(alpha_adim=dataset.lambda_adim).to(device)

    # Poids initiaux de la fonction de perte
    loss_weights = (1.0, 1.0, 1.0)  # (w_ic, w_bc, w_res)

    # ==========================================
    # PHASE 1 : Optimiseur ADAM
    # ==========================================
    epochs_adam = 2000
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    train_loss_history = []
    val_loss_history = []
    print(f"\n--- Début de la phase 1 : ADAM ({epochs_adam} epochs) ---")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()

        # Calcul de la perte
        loss, (l_ic, l_bc, l_res) = model.compute_loss(ic_data_train, bc_data_train, res_data_train, loss_weights)

        # Rétropropagation
        loss.backward()
        optimizer_adam.step()

        val_loss, (val_l_ic, val_l_bc, val_l_res) = model.compute_loss(ic_data_val, bc_data_val, res_data_val, loss_weights)

        if epoch % 500 == 0:
            print(
                f"Adam Epoch {epoch:04d} | Loss Globale: {loss.item():.4e} | IC: {l_ic:.4e} | BC: {l_bc:.4e} | EDP: {l_res:.4e}")
            print(
                f"Validation | Loss Globale: {val_loss.item():.4e} | IC: {val_l_ic:.4e} | BC: {val_l_bc:.4e} | EDP: {val_l_res:.4e}")

        val_loss_history.append(val_loss.item())
        train_loss_history.append(loss.item())

    plot_loss(train_loss_history, val_loss_history)

    # ==========================================
    # PHASE 2 : Optimiseur L-BFGS
    # ==========================================
    epochs_lbfgs = 1000

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1e-3,
        max_iter=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100
    )

    print(f"\n--- Début de la phase 2 : L-BFGS ({epochs_lbfgs} itérations max) ---")
    lbfgs_iter = 0

    def closure():
        # iter = lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss, (l_ic, l_bc, l_res) = model.compute_loss(ic_data_train, bc_data_train, res_data_train, loss_weights)
        # print(f"L-BFGS Iter {iter:04d} | Loss Globale: {loss.item():.4e} | IC: {l_ic:.4e} | BC: {l_bc:.4e} | EDP: {l_res:.4e}")
        loss.backward()
        return loss

    for epoch in range(epochs_lbfgs):
        loss = optimizer_lbfgs.step(closure)
        val_loss, (val_l_ic, val_l_bc, val_l_res) = model.compute_loss(ic_data_val, bc_data_val, res_data_val,
                                                                       loss_weights)
        if lbfgs_iter % 50 == 0:
            print(f"L-BFGS Iter {lbfgs_iter:04d} | Loss Globale: {loss.item():.4e}")
            print(f"Validation | Loss Globale: {val_loss.item():.4e} | IC: {val_l_ic:.4e} | BC: {val_l_bc:.4e} | EDP: {val_l_res:.4e}")
            
        lbfgs_iter += 1

        val_loss_history.append(val_loss.item())
        train_loss_history.append(loss.item())

    plot_loss(train_loss_history, val_loss_history)

    # 4. Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), "../heat_pinn_model.pth")
    print("\nEntraînement terminé. Modèle sauvegardé sous 'heat_pinn_model.pth'.")