import torch
import numpy as np
from pydoe import lhs

class ThermalDataset:
    """
    Classe pour générer des points d'entraînement pour un problème de diffusion thermique 2D.
    Le domaine est un carré [0,1]x[0,1] avec un objet chaud au centre. La température est adimensionnée entre 0 et 1.
    :arg:
    - L : longueur du domaine (m)
    - alpha_phys : diffusivité thermique physique (m^2/s)
    - t_max : temps maximum pour la simulation (s)
    - lambda_adim : diffusivité thermique adimensionnée
    - N_ic : nombre de points pour les conditions initiales
    - N_bc : nombre de points pour les conditions aux limites
    - N_res : nombre de points pour le résidu interne
    - obj_x_min, obj_x_max, obj_y_min, obj_y_max : coordonnées de l'objet chaud au centre du domaine
    """

    def __init__(self, N_ic=2000, N_bc=2000, N_res=10000, t_max=3600.0):
        # Paramètres physiques
        self.L = 1.0  # m (domaine 1x1)
        self.alpha_phys = 2e-5  # m^2/s
        self.t_max = t_max

        # Paramètre de l'EDP adimensionnée
        self.lambda_adim = (self.alpha_phys * self.t_max) / (self.L ** 2)

        # Nombre de points
        self.N_ic = N_ic
        self.N_bc = N_bc
        self.N_res = N_res

        # Définition de l'objet chaud au centre (un pavé par exemple)
        self.obj_x_min, self.obj_x_max = 0.4, 0.6
        self.obj_y_min, self.obj_y_max = 0.4, 0.6

    def is_inside_object(self, x, y):
        """
        Vérifie si les coordonnées (x,y) sont dans l'objet chaud
        :param: x: coordonnées x (tensor)
        :param: y: coordonnées y (tensor)
        :return: booléen indiquant si le point est à l'intérieur de l'objet
        """
        return (x >= self.obj_x_min) & (x <= self.obj_x_max) & \
            (y >= self.obj_y_min) & (y <= self.obj_y_max)

    def generate_data(self):
        """
        Génère les points d'entraînement N_ic, N_bc, et N_res.
        :return: tuples de points pour les conditions initiales, aux limites et le résidu interne
        """

        # 1. Résidu interne (N_res) dans le domaine spatio-temporel [0,1]x[0,1]x[0,1]
        # x, y, t
        res_points = lhs(3, self.N_res)
        x_res = torch.tensor(res_points[:, 0:1], dtype=torch.float32)
        y_res = torch.tensor(res_points[:, 1:2], dtype=torch.float32)
        t_res = torch.tensor(res_points[:, 2:3], dtype=torch.float32)

        # 2. Conditions Initiales (N_ic) à t = 0
        ic_points = lhs(2, self.N_ic)
        x_ic = torch.tensor(ic_points[:, 0:1], dtype=torch.float32)
        y_ic = torch.tensor(ic_points[:, 1:2], dtype=torch.float32)
        t_ic = torch.zeros_like(x_ic)  # t=0

        # Assignation de la température adimensionnée (1.0 dans l'objet, 0.0 sinon)
        inside = self.is_inside_object(x_ic, y_ic)
        T_ic = torch.where(inside, torch.tensor(1.0), torch.tensor(0.0))

        # 3. Conditions aux limites de Dirichlet (N_bc)
        # On divise N_bc par 4 pour les 4 murs (x=0, x=1, y=0, y=1)
        N_murs = self.N_bc // 4

        # Echantillonnage spatial 1D et temporel 1D pour les murs
        bc_1 = lhs(2, N_murs)  # mur x=0
        bc_2 = lhs(2, N_murs)  # mur x=1
        bc_3 = lhs(2, N_murs)  # mur y=0
        bc_4 = lhs(2, N_murs)  # mur y=1

        # Concaténation des 4 frontières (x, y, t)
        x_bc = torch.tensor(np.concatenate([np.zeros((N_murs, 1)), np.ones((N_murs, 1)), bc_3[:, 0:1], bc_4[:, 0:1]]),
                            dtype=torch.float32)
        y_bc = torch.tensor(np.concatenate([bc_1[:, 0:1], bc_2[:, 0:1], np.zeros((N_murs, 1)), np.ones((N_murs, 1))]),
                            dtype=torch.float32)
        t_bc = torch.tensor(np.concatenate([bc_1[:, 1:2], bc_2[:, 1:2], bc_3[:, 1:2], bc_4[:, 1:2]]),
                            dtype=torch.float32)

        # La température sur les murs est toujours T_amb (adimensionnée à 0.0)
        T_bc = torch.zeros_like(x_bc)

        return (x_ic, y_ic, t_ic, T_ic), (x_bc, y_bc, t_bc, T_bc), (x_res, y_res, t_res)