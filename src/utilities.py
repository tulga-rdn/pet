import os
import random
import torch
import numpy as np

from sklearn.linear_model import Ridge
from scipy.spatial.transform import Rotation
import copy

def get_self_contributions(energy_key, train_structures, all_species):
    train_energies = np.array([structure.info[energy_key] for structure in train_structures])
    train_c_feat = get_compositional_features(train_structures, all_species)
    rgr = Ridge(alpha = 1e-10, fit_intercept = False)
    rgr.fit(train_c_feat, train_energies)
    return rgr.coef_

def get_corrected_energies(energy_key, structures, all_species, self_contributions):
    energies = np.array([structure.info[energy_key] for structure in structures])

    compositional_features = get_compositional_features(structures, all_species)
    self_contributions_energies = []
    for i in range(len(structures)):
        self_contributions_energies.append(np.dot(compositional_features[i], self_contributions))
    self_contributions_energies = np.array(self_contributions_energies)
    return energies - self_contributions_energies


def get_calc_names(all_completed_calcs, current_name):
    name_to_load = None
    name_of_calculation = current_name
    if name_of_calculation in all_completed_calcs:
        name_to_load = name_of_calculation
        for i in range(100000):
            name_now = name_of_calculation + f"_continuation_{i}"
            if name_now not in all_completed_calcs:
                name_to_save = name_now
                break
            name_to_load = name_now
        name_of_calculation = name_to_save
    return name_to_load, name_of_calculation


def set_reproducibility(random_seed, cuda_deterministic):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic and torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_all_species(structures):
    all_species = []
    for structure in structures:
        all_species.append(np.array(structure.get_atomic_numbers()))
    all_species = np.concatenate(all_species, axis=0)
    all_species = np.sort(np.unique(all_species))
    return all_species


def get_compositional_features(structures, all_species):
    result = np.zeros([len(structures), len(all_species)])
    for i, structure in enumerate(structures):
        species_now = structure.get_atomic_numbers()
        for j, specie in enumerate(all_species):
            num = np.sum(species_now == specie)
            result[i, j] = num
    return result


def get_length(delta):
    return np.sqrt(np.sum(delta * delta))


class ModelKeeper:
    def __init__(self):
        self.best_model = None
        self.best_error = None
        self.best_epoch = None
        self.additional_info = None

    def update(self, model_now, error_now, epoch_now, additional_info=None):
        if (self.best_error is None) or (error_now < self.best_error):
            self.best_error = error_now
            self.best_model = copy.deepcopy(model_now)
            self.best_epoch = epoch_now
            self.additional_info = additional_info


class Logger:
    def __init__(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions_now, targets_now):
        self.predictions.append(predictions_now.data.cpu().numpy())
        self.targets.append(targets_now.data.cpu().numpy())

    def flush(self):
        self.predictions = np.concatenate(self.predictions, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        output = {}
        output["rmse"] = get_rmse(self.predictions, self.targets)
        output["mae"] = get_mae(self.predictions, self.targets)
        output["relative rmse"] = get_relative_rmse(self.predictions, self.targets)

        self.predictions = []
        self.targets = []
        return output


class FullLogger:
    def __init__(self):
        self.train_logger = Logger()
        self.val_logger = Logger()

    def flush(self):
        return {"train": self.train_logger.flush(), "val": self.val_logger.flush()}


def get_rotations(indices, global_aug=False):
    if global_aug:
        num = np.max(indices) + 1
    else:
        num = indices.shape[0]

    rotations = Rotation.random(num).as_matrix()
    rotations[np.random.randn(rotations.shape[0]) >= 0] *= -1

    if global_aug:
        return rotations[indices]
    else:
        return rotations


def get_loss(predictions, targets):
    delta = predictions - targets
    return torch.mean(delta * delta)


def get_rmse(first, second):
    delta = first - second
    return np.sqrt(np.mean(delta * delta))


def get_mae(first, second):
    delta = first - second
    return np.mean(np.abs(delta))


def get_relative_rmse(predictions, targets):
    rmse = get_rmse(predictions, targets)
    return rmse / get_rmse(np.mean(targets), targets)
