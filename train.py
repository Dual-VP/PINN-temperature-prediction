import torch
import torch.optim as optim
from dataset import ThermalDataset
from model import HeatPINN


def _batch(data, indices):
    return tuple(value[indices] for value in data)


def train_pinn(
        epochs_adam=3,
        batch_size_ic=2048,
        batch_size_bc=2048,
        batch_size_res=16384,
        use_lbfgs=True,
):
    # 1. Configuration du device (essentiel pour le cluster SLURM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {device}")

    # 2. Génération des données
    print("Génération des points de collocation...")
    dataset = ThermalDataset(N_ic=8192, N_bc=8192, N_res=65536)
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
    # Pour Adam, la règle en racine carrée est plus stable que la règle
    # linéaire lorsque la taille des mini-batchs augmente.
    reference_batch_size = 10000
    adam_lr = 1e-3 * (batch_size_res / reference_batch_size) ** 0.5
    optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr)
    print(
        f"Mini-batchs IC/BC/résidu : {batch_size_ic}/{batch_size_bc}/{batch_size_res}"
    )
    print(f"Learning rate Adam ajusté : {adam_lr:.2e}")

    print(f"\n--- Début de la phase 1 : ADAM ({epochs_adam} epochs) ---")
    for epoch in range(epochs_adam):
        permutations = [
            torch.randperm(data[0].shape[0], device=device)
            for data in (ic_data, bc_data, res_data)
        ]
        n_steps = max(
            (ic_data[0].shape[0] + batch_size_ic - 1) // batch_size_ic,
            (bc_data[0].shape[0] + batch_size_bc - 1) // batch_size_bc,
            (res_data[0].shape[0] + batch_size_res - 1) // batch_size_res,
        )
        losses = []

        for step in range(n_steps):
            optimizer_adam.zero_grad(set_to_none=True)
            ic_start = (step * batch_size_ic) % ic_data[0].shape[0]
            bc_start = (step * batch_size_bc) % bc_data[0].shape[0]
            res_start = (step * batch_size_res) % res_data[0].shape[0]

            ic_indices = permutations[0][ic_start:ic_start + batch_size_ic]
            bc_indices = permutations[1][bc_start:bc_start + batch_size_bc]
            res_indices = permutations[2][res_start:res_start + batch_size_res]
            loss, (l_ic, l_bc, l_res) = model.compute_loss(
                _batch(ic_data, ic_indices),
                _batch(bc_data, bc_indices),
                _batch(res_data, res_indices),
                loss_weights,
            )
            loss.backward()
            optimizer_adam.step()
            losses.append(loss.item())

        if epoch % 500 == 0:
            print(
                f"Adam Epoch {epoch:04d} | Loss Globale: {sum(losses) / len(losses):.4e} "
                f"| IC: {l_ic:.4e} | BC: {l_bc:.4e} | EDP: {l_res:.4e}"
            )

    # ==========================================
    # PHASE 2 : Optimiseur L-BFGS
    # ==========================================
    if not use_lbfgs:
        torch.save(model.state_dict(), "heat_pinn_model.pth")
        print("\nEntraînement terminé. Modèle sauvegardé sous 'heat_pinn_model.pth'.")
        return

    # Une fermeture L-BFGS doit rester déterministe. On utilise donc un
    # mini-batch fixe afin d'éviter de conserver tout le résidu en mémoire GPU.
    lbfgs_batch_size = min(batch_size_res, res_data[0].shape[0])
    lbfgs_res_data = _batch(res_data, torch.arange(lbfgs_batch_size, device=device))
    lbfgs_ic_data = _batch(
        ic_data, torch.arange(min(batch_size_ic, ic_data[0].shape[0]), device=device)
    )
    lbfgs_bc_data = _batch(
        bc_data, torch.arange(min(batch_size_bc, bc_data[0].shape[0]), device=device)
    )
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
        loss, (l_ic, l_bc, l_res) = model.compute_loss(
            lbfgs_ic_data, lbfgs_bc_data, lbfgs_res_data, loss_weights
        )
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