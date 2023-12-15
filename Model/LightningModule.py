import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from Model.CNN import CNN
from Model.Tester import tester,valider
from Model.UNet import UNet

# define the LightningModule
class Module(pl.LightningModule):
    def __init__(self, log_dir,save=False):
        super().__init__()
        self.tester = tester(log_path=log_dir, save=save)
        self.model = UNet()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["x"]
        y = batch["y"]
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["x"]
        y = batch["y"]
        y_hat = self.model(x)
        snr = self.tester(batch,y_hat)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.log("val_snr", snr.mean())
        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["x"]
        y_hat = self.model(x)
        snr = self.tester(batch,y_hat)
        # Logging to TensorBoard (if     installed) by default
        self.log("test_snr", snr.mean())
        return snr.mean()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

