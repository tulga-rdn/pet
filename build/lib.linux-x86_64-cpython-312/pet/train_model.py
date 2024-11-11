from .data_preparation import get_all_species
import os

import torch
import ase.io
import numpy as np
import logging
from .utilities import ModelKeeper
import time
import datetime
import pickle
from torch_geometric.nn import DataParallel

from .hypers import save_hypers, set_hypers_from_files, Hypers, hypers_to_dict
from .pet import PET, PETMLIPWrapper, PETUtilityWrapper
from .utilities import (
    FullLogger,
    get_scheduler,
    load_checkpoint,
    get_data_loaders,
    log_epoch_stats,
)
from .utilities import get_rmse, get_loss, set_reproducibility, get_calc_names
from .utilities import get_optimizer
from .analysis import adapt_hypers
from .data_preparation import get_self_contributions, get_corrected_energies
import argparse
from .data_preparation import get_pyg_graphs, update_pyg_graphs, get_forces
from .utilities import dtype2string, string2dtype
from .pet import FlagsWrapper
import sys

logger = logging.getLogger(__name__)

format = "[{asctime}][{levelname}]" + " - {message}"
formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S", style="{")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
handlers = [stream_handler]

logging.basicConfig(format=format, handlers=handlers, level=logging.INFO, style="{")
logging.captureWarnings(True)

for handler in handlers:
    logger.addHandler(handler)


def fit_pet(
    train_structures,
    val_structures,
    hypers_dict,
    name_of_calculation,
    device,
    output_dir,
    checkpoint_path=None,
):
    logging.info("Initializing PET training...")

    TIME_SCRIPT_STARTED = time.time()
    value = datetime.datetime.fromtimestamp(TIME_SCRIPT_STARTED)
    logging.info(f"Starting training at: {value.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Training configuration:")

    print(f"Output directory: {output_dir}")
    print(f"Training using device: {device }")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    hypers = Hypers(hypers_dict)
    dtype = string2dtype(hypers.ARCHITECTURAL_HYPERS.DTYPE)
    torch.set_default_dtype(dtype)

    FITTING_SCHEME = hypers.FITTING_SCHEME
    MLIP_SETTINGS = hypers.MLIP_SETTINGS
    ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS

    if FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS:
        raise ValueError(
            "shift agnostic loss is intended only for general target training"
        )

    ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"  # energy is structural property
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = (
        "sum"  # energy is a sum of atomic energies
    )
    print(f"Output dimensionality: {ARCHITECTURAL_HYPERS.D_OUTPUT}")
    print(f"Target type: {ARCHITECTURAL_HYPERS.TARGET_TYPE}")
    print(f"Target aggregation: {ARCHITECTURAL_HYPERS.TARGET_AGGREGATION}")

    set_reproducibility(FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC)

    print(f"Random seed: {FITTING_SCHEME.RANDOM_SEED}")
    print(f"CUDA is deterministic: {FITTING_SCHEME.CUDA_DETERMINISTIC}")

    adapt_hypers(FITTING_SCHEME, train_structures)
    structures = train_structures + val_structures

    all_dataset_species = get_all_species(structures)

    if FITTING_SCHEME.ALL_SPECIES_PATH is not None:
        logging.info(f"Loading all species from: {FITTING_SCHEME.ALL_SPECIES_PATH}")
        all_species = np.load(FITTING_SCHEME.ALL_SPECIES_PATH)
        if not np.all(np.isin(all_dataset_species, all_species)):
            raise ValueError(
                "For the model fine-tuning, the set of species in the dataset "
                "must be a subset of the set of species in the pre-trained model. "
                "Please check, if the ALL_SPECIES_PATH is file contains all the elements "
                "from the fine-tuning dataset."
            )
    else:
        all_species = all_dataset_species

    name_to_load, NAME_OF_CALCULATION = get_calc_names(
        os.listdir(output_dir), name_of_calculation
    )

    os.mkdir(f"{output_dir}/{NAME_OF_CALCULATION}")
    np.save(f"{output_dir}/{NAME_OF_CALCULATION}/all_species.npy", all_species)
    hypers.UTILITY_FLAGS.CALCULATION_TYPE = "mlip"
    save_hypers(hypers, f"{output_dir}/{NAME_OF_CALCULATION}/hypers_used.yaml")

    logging.info("Convering structures to PyG graphs...")

    train_graphs = get_pyg_graphs(
        train_structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
    )
    val_graphs = get_pyg_graphs(
        val_structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
    )

    logging.info("Pre-processing training data...")
    if MLIP_SETTINGS.USE_ENERGIES:
        if FITTING_SCHEME.SELF_CONTRIBUTIONS_PATH is not None:
            logging.info(
                f"Loading self contributions from: {FITTING_SCHEME.SELF_CONTRIBUTIONS_PATH}"
            )
            self_contributions = np.load(FITTING_SCHEME.SELF_CONTRIBUTIONS_PATH)
        else:
            self_contributions = get_self_contributions(
                MLIP_SETTINGS.ENERGY_KEY, train_structures, all_species
            )

        np.save(
            f"{output_dir}/{NAME_OF_CALCULATION}/self_contributions.npy",
            self_contributions,
        )

        train_energies = get_corrected_energies(
            MLIP_SETTINGS.ENERGY_KEY, train_structures, all_species, self_contributions
        )
        val_energies = get_corrected_energies(
            MLIP_SETTINGS.ENERGY_KEY, val_structures, all_species, self_contributions
        )

        update_pyg_graphs(train_graphs, "y", train_energies)
        update_pyg_graphs(val_graphs, "y", val_energies)

    if MLIP_SETTINGS.USE_FORCES:
        train_forces = get_forces(train_structures, MLIP_SETTINGS.FORCES_KEY)
        val_forces = get_forces(val_structures, MLIP_SETTINGS.FORCES_KEY)

        update_pyg_graphs(train_graphs, "forces", train_forces)
        update_pyg_graphs(val_graphs, "forces", val_forces)

    train_loader, val_loader = get_data_loaders(
        train_graphs, val_graphs, FITTING_SCHEME
    )

    logging.info("Initializing the model...")
    model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
    model = PETUtilityWrapper(model, FITTING_SCHEME.GLOBAL_AUG)

    model = PETMLIPWrapper(model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES)

    if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
        logging.info(f"Using multi-GPU training on {torch.cuda.device_count()} GPUs")
        model = DataParallel(FlagsWrapper(model))
        model = model.to(torch.device("cuda:0"))

    if FITTING_SCHEME.MODEL_TO_START_WITH is not None:
        logging.info(f"Loading model from: {FITTING_SCHEME.MODEL_TO_START_WITH}")
        model.load_state_dict(
            torch.load(FITTING_SCHEME.MODEL_TO_START_WITH, weights_only=True)
        )
        model = model.to(dtype=dtype)

    optim = get_optimizer(model, FITTING_SCHEME)
    scheduler = get_scheduler(optim, FITTING_SCHEME)

    if checkpoint_path is not None:
        logging.info(f"Loading model and checkpoint from: {checkpoint_path}\n")
        load_checkpoint(model, optim, scheduler, checkpoint_path)
    elif name_to_load is not None:
        logging.info(
            f"Loading model and checkpoint from: {output_dir}/{name_to_load}/checkpoint\n"
        )
        load_checkpoint(
            model, optim, scheduler, f"{output_dir}/{name_to_load}/checkpoint"
        )

    history = []
    if MLIP_SETTINGS.USE_ENERGIES:
        energies_logger = FullLogger(
            FITTING_SCHEME.SUPPORT_MISSING_VALUES,
            FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
            device,
        )

    if MLIP_SETTINGS.USE_FORCES:
        forces_logger = FullLogger(
            FITTING_SCHEME.SUPPORT_MISSING_VALUES,
            FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
            device,
        )

    if MLIP_SETTINGS.USE_FORCES:
        val_forces = torch.cat(val_forces, dim=0)

        sliding_forces_rmse = get_rmse(
            val_forces.data.cpu().to(dtype=torch.float32).numpy(), 0.0
        )

        forces_rmse_model_keeper = ModelKeeper()
        forces_mae_model_keeper = ModelKeeper()

    if MLIP_SETTINGS.USE_ENERGIES:
        if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
            sliding_energies_rmse = get_rmse(val_energies, np.mean(val_energies))
        else:
            val_n_atoms = np.array([len(struc.positions) for struc in val_structures])
            val_energies_per_atom = val_energies / val_n_atoms
            sliding_energies_rmse = get_rmse(
                val_energies_per_atom, np.mean(val_energies_per_atom)
            )

        energies_rmse_model_keeper = ModelKeeper()
        energies_mae_model_keeper = ModelKeeper()

    if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
        multiplication_rmse_model_keeper = ModelKeeper()
        multiplication_mae_model_keeper = ModelKeeper()

    logging.info(f"Starting training for {FITTING_SCHEME.EPOCH_NUM} epochs")
    if FITTING_SCHEME.EPOCHS_WARMUP > 0:
        logging.info(f"Performing {FITTING_SCHEME.EPOCHS_WARMUP} epochs of LR warmup")
    TIME_TRAINING_STARTED = time.time()
    last_elapsed_time = 0
    print("=" * 50)
    for epoch in range(1, FITTING_SCHEME.EPOCH_NUM + 1):
        model.train(True)
        for batch in train_loader:
            print('positions: ', batch.positions.shape, flush=True)
            print('batch_y: ', batch.y.shape, flush=True)
            predictions_dict = model.get_predictions(batch, augmentation=True)
            print('last_layer_features: ', predictions_dict["last_layer_features"].shape, flush=True)
            print("cell list len", len(batch.cell), flush=True)
            print("cell list [0] shape", np.shape(batch.cell[0]), flush=True)
            print("neighbors index: ", [t.unsqueeze(0) for t in batch.neighbors_index], flush=True)
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            if FITTING_SCHEME.MULTI_GPU:
                model.module.augmentation = True
                model.module.create_graph = True
                predictions_energies, predictions_forces, last_layer_features = model(batch)
            else:
                predictions_energies, predictions_forces, last_layer_features = model(
                    batch, augmentation=True, create_graph=True
                )
                print('predictions_energies: ', predictions_energies.shape, flush=True)
                print('last_layer_features: ', last_layer_features.shape, flush=True)

            if FITTING_SCHEME.MULTI_GPU:
                y_list = [el.y for el in batch]
                batch_y = torch.tensor(
                    y_list, dtype=torch.get_default_dtype(), device=device
                )

                n_atoms_list = [el.n_atoms for el in batch]
                batch_n_atoms = torch.tensor(
                    n_atoms_list, dtype=torch.get_default_dtype(), device=device
                )
                # print('batch_y: ', batch_y.shape)
                # print('batch_n_atoms: ', batch_n_atoms.shape)

            else:
                batch_y = batch.y
                batch_n_atoms = batch.n_atoms

            if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                predictions_energies = predictions_energies / batch_n_atoms
                ground_truth_energies = batch_y / batch_n_atoms
            else:
                ground_truth_energies = batch_y

            if MLIP_SETTINGS.USE_ENERGIES:
                energies_logger.train_logger.update(
                    predictions_energies, ground_truth_energies
                )
                loss_energies = get_loss(
                    predictions_energies,
                    ground_truth_energies,
                    FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                    FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                )
            if MLIP_SETTINGS.USE_FORCES:

                if FITTING_SCHEME.MULTI_GPU:
                    forces_list = [el.forces for el in batch]
                    batch_forces = torch.cat(forces_list, dim=0).to(device)
                else:
                    batch_forces = batch.forces

                forces_logger.train_logger.update(predictions_forces, batch_forces)
                loss_forces = get_loss(
                    predictions_forces,
                    batch_forces,
                    FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                    FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                )

            if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
                loss = FITTING_SCHEME.ENERGY_WEIGHT * loss_energies / (
                    sliding_energies_rmse**2
                ) + loss_forces / (sliding_forces_rmse**2)
                loss.backward()

            if MLIP_SETTINGS.USE_ENERGIES and (not MLIP_SETTINGS.USE_FORCES):
                loss_energies.backward()
            if MLIP_SETTINGS.USE_FORCES and (not MLIP_SETTINGS.USE_ENERGIES):
                loss_forces.backward()

            if FITTING_SCHEME.DO_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=FITTING_SCHEME.GRADIENT_CLIPPING_MAX_NORM,
                )
            optim.step()
            optim.zero_grad()

        model.train(False)
        for batch in val_loader:
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            if FITTING_SCHEME.MULTI_GPU:
                model.module.augmentation = False
                model.module.create_graph = False
                predictions_energies, predictions_forces, last_layer_features = model(batch)
            else:
                predictions_energies, predictions_forces, last_layer_features = model(
                    batch, augmentation=False, create_graph=False
                )

            if FITTING_SCHEME.MULTI_GPU:
                y_list = [el.y for el in batch]
                batch_y = torch.tensor(
                    y_list, dtype=torch.get_default_dtype(), device=device
                )

                n_atoms_list = [el.n_atoms for el in batch]
                batch_n_atoms = torch.tensor(
                    n_atoms_list, dtype=torch.get_default_dtype(), device=device
                )

                # print('batch_y: ', batch_y.shape)
                # print('batch_n_atoms: ', batch_n_atoms.shape)
            else:
                batch_y = batch.y
                batch_n_atoms = batch.n_atoms

            if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                predictions_energies = predictions_energies / batch_n_atoms
                ground_truth_energies = batch_y / batch_n_atoms
            else:
                ground_truth_energies = batch_y

            if MLIP_SETTINGS.USE_ENERGIES:
                energies_logger.val_logger.update(
                    predictions_energies, ground_truth_energies
                )
            if MLIP_SETTINGS.USE_FORCES:
                if FITTING_SCHEME.MULTI_GPU:
                    forces_list = [el.forces for el in batch]
                    batch_forces = torch.cat(forces_list, dim=0).to(device)
                else:
                    batch_forces = batch.forces
                forces_logger.val_logger.update(predictions_forces, batch_forces)

        now = {}
        if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
            energies_key = "energies per structure"
        else:
            energies_key = "energies per atom"

        if MLIP_SETTINGS.USE_ENERGIES:
            now[energies_key] = energies_logger.flush()
        else:
            energies_key = ""

        if MLIP_SETTINGS.USE_FORCES:
            now["forces"] = forces_logger.flush()
        now["lr"] = scheduler.get_last_lr()
        now["epoch"] = epoch

        now["elapsed_time"] = time.time() - TIME_TRAINING_STARTED
        now["epoch_time"] = now["elapsed_time"] - last_elapsed_time
        now["estimated_remaining_time"] = (now["elapsed_time"] / epoch) * (
            FITTING_SCHEME.EPOCH_NUM - epoch
        )
        last_elapsed_time = now["elapsed_time"]

        if MLIP_SETTINGS.USE_ENERGIES:
            sliding_energies_rmse = (
                FITTING_SCHEME.SLIDING_FACTOR * sliding_energies_rmse
                + (1.0 - FITTING_SCHEME.SLIDING_FACTOR)
                * now[energies_key]["val"]["rmse"]
            )

            energies_mae_model_keeper.update(
                model, now[energies_key]["val"]["mae"], epoch
            )
            energies_rmse_model_keeper.update(
                model, now[energies_key]["val"]["rmse"], epoch
            )

        if MLIP_SETTINGS.USE_FORCES:
            sliding_forces_rmse = (
                FITTING_SCHEME.SLIDING_FACTOR * sliding_forces_rmse
                + (1.0 - FITTING_SCHEME.SLIDING_FACTOR) * now["forces"]["val"]["rmse"]
            )
            forces_mae_model_keeper.update(model, now["forces"]["val"]["mae"], epoch)
            forces_rmse_model_keeper.update(model, now["forces"]["val"]["rmse"], epoch)

        if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
            multiplication_mae_model_keeper.update(
                model,
                now["forces"]["val"]["mae"] * now[energies_key]["val"]["mae"],
                epoch,
                additional_info=[
                    now[energies_key]["val"]["mae"],
                    now["forces"]["val"]["mae"],
                ],
            )
            multiplication_rmse_model_keeper.update(
                model,
                now["forces"]["val"]["rmse"] * now[energies_key]["val"]["rmse"],
                epoch,
                additional_info=[
                    now[energies_key]["val"]["rmse"],
                    now["forces"]["val"]["rmse"],
                ],
            )
        last_lr = scheduler.get_last_lr()[0]
        log_epoch_stats(epoch, FITTING_SCHEME.EPOCH_NUM, now, last_lr, energies_key)

        history.append(now)
        scheduler.step()
        elapsed = time.time() - TIME_SCRIPT_STARTED
        if epoch > 0 and epoch % FITTING_SCHEME.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "dtype_used": dtype2string(dtype),
                },
                f"{output_dir}/{NAME_OF_CALCULATION}/checkpoint_{epoch}",
            )
        if FITTING_SCHEME.MAX_TIME is not None:
            if elapsed > FITTING_SCHEME.MAX_TIME:
                logging.info("Reached maximum time\n")
                break
    logging.info("Training is finished\n")
    logging.info("Saving the model and history...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dtype_used": dtype2string(dtype),
        },
        f"{output_dir}/{NAME_OF_CALCULATION}/checkpoint",
    )
    with open(f"{output_dir}/{NAME_OF_CALCULATION}/history.pickle", "wb") as f:
        pickle.dump(history, f)

    def save_model(model_name, model_keeper):
        torch.save(
            model_keeper.best_model.state_dict(),
            f"{output_dir}/{NAME_OF_CALCULATION}/{model_name}_state_dict",
        )

    summary = ""
    if MLIP_SETTINGS.USE_ENERGIES:
        if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
            postfix = "per structure"
        if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
            postfix = "per atom"
        save_model("best_val_mae_energies_model", energies_mae_model_keeper)
        summary += f"best val mae in energies {postfix}: {energies_mae_model_keeper.best_error} at epoch {energies_mae_model_keeper.best_epoch}\n"

        save_model("best_val_rmse_energies_model", energies_rmse_model_keeper)
        summary += f"best val rmse in energies {postfix}: {energies_rmse_model_keeper.best_error} at epoch {energies_rmse_model_keeper.best_epoch}\n"

    if MLIP_SETTINGS.USE_FORCES:
        save_model("best_val_mae_forces_model", forces_mae_model_keeper)
        summary += f"best val mae in forces: {forces_mae_model_keeper.best_error} at epoch {forces_mae_model_keeper.best_epoch}\n"

        save_model("best_val_rmse_forces_model", forces_rmse_model_keeper)
        summary += f"best val rmse in forces: {forces_rmse_model_keeper.best_error} at epoch {forces_rmse_model_keeper.best_epoch}\n"

    if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
        save_model("best_val_mae_both_model", multiplication_mae_model_keeper)
        summary += f"best both (multiplication) mae in energies {postfix}: {multiplication_mae_model_keeper.additional_info[0]} in forces: {multiplication_mae_model_keeper.additional_info[1]} at epoch {multiplication_mae_model_keeper.best_epoch}\n"

        save_model("best_val_rmse_both_model", multiplication_rmse_model_keeper)
        summary += f"best both (multiplication) rmse in energies {postfix}: {multiplication_rmse_model_keeper.additional_info[0]} in forces: {multiplication_rmse_model_keeper.additional_info[1]} at epoch {multiplication_rmse_model_keeper.best_epoch}\n"

    with open(f"{output_dir}/{NAME_OF_CALCULATION}/summary.txt", "w") as f:
        print(summary, file=f)
    logging.info(f"Total elapsed time: {time.time() - TIME_SCRIPT_STARTED}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_structures_path",
        help="Path to an xyz file with train structures",
        type=str,
    )
    parser.add_argument(
        "val_structures_path",
        help="Path to an xyz file with validation structures",
        type=str,
    )
    parser.add_argument(
        "provided_hypers_path",
        help="Path to a YAML file with provided hypers",
        type=str,
    )
    parser.add_argument(
        "default_hypers_path", help="Path to a YAML file with default hypers", type=str
    )
    parser.add_argument(
        "name_of_calculation", help="Name of this calculation", type=str
    )
    parser.add_argument("--gpu_id", help="ID of the GPU to use", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    train_structures = ase.io.read(args.train_structures_path, index=":")
    val_structures = ase.io.read(args.val_structures_path, index=":")

    hypers = set_hypers_from_files(args.provided_hypers_path, args.default_hypers_path)

    name_of_calculation = args.name_of_calculation

    output_dir = "results"

    hypers_dict = hypers_to_dict(hypers)
    fit_pet(
        train_structures,
        val_structures,
        hypers_dict,
        name_of_calculation,
        device,
        output_dir,
    )


if __name__ == "__main__":
    main()
