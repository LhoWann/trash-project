import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import GarbageDataModule
from model import GarbageClassifier
import config

def evaluate():
    data_module = GarbageDataModule()
    data_module.setup()
    test_loader = data_module.test_dataloader()
    class_names = data_module.class_names

    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.ckpt"
    model = GarbageClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.cuda()
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - ConvNeXt V2 Tiny')
    plt.show()

if __name__ == '__main__':
    evaluate()