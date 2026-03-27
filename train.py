import argparse
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
import optuna

from src import CatVSDogDataModule, SimpleCNN

ARTIFACT_FOLDER = "checkpoints"

def objective(trial: optuna.trial.Trial, data_module: CatVSDogDataModule | None) -> float:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    if data_module is None:
        data_module = CatVSDogDataModule(data_dir="./data", batch_size=32, num_workers=4)

    model = SimpleCNN(learning_rate=lr)

    logger = MLFlowLogger(
        experiment_name="cats-vs-dogs-experiment",
        tracking_uri="sqlite:///mlflow.db",
        artifact_location=ARTIFACT_FOLDER,
        run_name=f"run-{trial.number}-LR-{lr}-{time.strftime('%Y-%m-%d-%H-%M-%S')}",
        log_model=True
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    hyperparameters = {"learning_rate": lr}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val/acc"].item()

def main():
    parser = argparse.ArgumentParser(description="Train a model for cats vs dogs classification")
    parser.add_argument("--optuna", "-o", action="store_true", help="Use optuna to find the best hyperparameters")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    dm = CatVSDogDataModule(data_dir="./data", batch_size=32, num_workers=4)

    if args.optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, dm), n_trials=10)
        return

    model = SimpleCNN(learning_rate=1e-3)

    logger = MLFlowLogger(
        experiment_name="cats-vs-dogs-experiment",
        tracking_uri="sqlite:///mlflow.db",
        artifact_location=ARTIFACT_FOLDER,
        log_model=True
    )
    # mlflow server

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
