import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm
import config

class GarbageClassifier(pl.LightningModule):
    def __init__(self, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(config.MODEL_NAME, pretrained=True, num_classes=config.NUM_CLASSES)
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        return [optimizer], [scheduler]