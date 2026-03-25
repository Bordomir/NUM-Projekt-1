import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataModule import CatVSDogDataModule
from model import SimpleCNN

def main():
    dm = CatVSDogDataModule(data_dir="./data", batch_size=32, num_workers=4)

    model = SimpleCNN(learning_rate=1e-3)

    logger = MLFlowLogger(
        experiment_name="cats-vs-dogs-experiment",
        save_dir="./mlruns" 
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="best-cat-dog"
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=3)],
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()