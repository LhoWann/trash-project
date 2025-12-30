import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import GarbageDataModule
from model import GarbageClassifier
import config

HIERARCHY_MAP = {
    'battery': 'B3',
    'biological': 'Organik',
    'brown-glass': 'Anorganik',
    'cardboard': 'Anorganik',
    'clothes': 'Anorganik',
    'green-glass': 'Anorganik',
    'metal': 'Anorganik',
    'paper': 'Anorganik',
    'plastic': 'Anorganik',
    'shoes': 'Anorganik',
    'trash': 'Anorganik',
    'white-glass': 'Anorganik'
}

PARENT_CLASSES = ['Organik', 'Anorganik', 'B3']
PARENT_TO_IDX = {name: i for i, name in enumerate(PARENT_CLASSES)}

def evaluate_hierarchy():
    data_module = GarbageDataModule()
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    sub_class_names = data_module.class_names 

    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.ckpt"
    model = GarbageClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()

    raw_preds = []
    raw_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.cuda()
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            raw_preds.extend(preds.cpu().numpy())
            raw_labels.extend(y.numpy())

    # Mapping (12 Kelas -> 3 Kelas)
    
    parent_preds = []
    parent_labels = []

    for i in range(len(raw_preds)):
        pred_sub_name = sub_class_names[raw_preds[i]]
        true_sub_name = sub_class_names[raw_labels[i]]
        
        pred_parent = HIERARCHY_MAP[pred_sub_name]
        true_parent = HIERARCHY_MAP[true_sub_name]
        
        parent_preds.append(PARENT_TO_IDX[pred_parent])
        parent_labels.append(PARENT_TO_IDX[true_parent])

    # 4. Tampilkan Laporan 3 Kelas
    print("\n" + "="*60)
    print("LAPORAN KLASIFIKASI: 3 KELAS BESAR (Organik, Anorganik, B3)")
    print("="*60)
    print(classification_report(parent_labels, parent_preds, target_names=PARENT_CLASSES, digits=4))

    # 5. Confusion Matrix 3 Kelas
    cm = confusion_matrix(parent_labels, parent_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=PARENT_CLASSES, yticklabels=PARENT_CLASSES)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya')
    plt.title('Confusion Matrix - Kategori Utama Sampah')
    plt.show()

if __name__ == '__main__':
    evaluate_hierarchy()    