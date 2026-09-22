import numpy as np


class FiniteDifferenceSolver:
    def __init__(self, nx=50, ny=50, t_max=3600.0):
        # Paramètres physiques identiques au PINN
        self.L = 1.0
        self.alpha = 2e-5
        self.t_max = t_max

        # Maillage spatial
        self.nx = nx
        self.ny = ny
        self.dx = self.L / (nx - 1)
        self.dy = self.L / (ny - 1)

        # Le schéma FTCS nécessite un dt critique pour la stabilité
        # dt <= (dx^2 * dy^2) / (2 * alpha * (dx^2 + dy^2))
        dt_critique = (self.dx ** 2 * self.dy ** 2) / (2 * self.alpha * (self.dx ** 2 + self.dy ** 2))
        self.dt = dt_critique * 0.9  # Marge de sécurité

        # Nombre de pas de temps
        self.nt = int(self.t_max / self.dt) + 1

        # Variables adimensionnées (pour correspondre au PINN)
        self.T_amb = 0.0
        self.T_obj = 1.0

    def solve(self, target_time):
        """
        Calcule la diffusion thermique jusqu'à un temps cible (target_time en secondes).
        Retourne la grille spatiale et la matrice de température correspondante.
        """
        # Initialisation de la grille (nx, ny) à T_amb
        T = np.full((self.nx, self.ny), self.T_amb)

        # Grilles de coordonnées
        x = np.linspace(0, self.L, self.nx)
        y = np.linspace(0, self.L, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Application de la condition initiale (objet à 80°C / 1.0)
        obj_mask = (X >= 0.4) & (X <= 0.6) & (Y >= 0.4) & (Y <= 0.6)
        T[obj_mask] = self.T_obj

        # Condition de fin
        target_step = min(int(target_time / self.dt), self.nt)

        # Boucle temporelle (Schéma explicite)
        for n in range(target_step):
            T_new = T.copy()
            # Mise à jour des points internes (hors frontières)
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    # Laplacien discret
                    d2T_dx2 = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / self.dx ** 2
                    d2T_dy2 = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / self.dy ** 2

                    # Mise à jour temporelle
                    T_new[i, j] = T[i, j] + self.alpha * self.dt * (d2T_dx2 + d2T_dy2)

            # Les bords restent à T_amb (conditions de Dirichlet) - déjà assuré par T_new.copy()
            T = T_new

        return X, Y, T