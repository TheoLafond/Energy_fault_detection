from Data.Datamodule.Datamodule import MyDataModule
import lightning as L

from Model.LightningModule import Module

if __name__ == "__main__":
    tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir="Logs/")
    model = Module(tb_logger.log_dir)
    datamodule = MyDataModule(batch_size=32,num_workers=4)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename='{epoch}-{val_loss:.2f}-{val_snr:.2f}',
        monitor='val_snr',
        mode="max",
        
    )
    # Init trainer
    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, datamodule)



