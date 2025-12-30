import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataset import GarbageDataModule
from model import GarbageClassifier
import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    pl.seed_everything(42)

    data_module = GarbageDataModule()
    data_module.setup()

    model = GarbageClassifier(class_weights=data_module.class_weights)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator='gpu',
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=config.ACCUM_STEPS,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()