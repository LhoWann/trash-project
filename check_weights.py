import os
import numpy as np
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight
import config

def print_weight_details():
    train_dir = os.path.join(config.DATA_DIR, 'train')
    
    if not os.path.exists(train_dir):
        print(f"Error: Folder {train_dir} tidak ditemukan.")
        return

    print("Load Dataset (Train)")
    dataset = datasets.ImageFolder(train_dir)
    
    targets = dataset.targets
    classes = dataset.classes
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    
    print("\n" + "="*65)
    print(f"{'KELAS':<20} | {'JUMLAH SAMPEL':<15} | {'BOBOT (WEIGHT)':<15}")
    print("="*65)
    
    sample_counts = np.bincount(targets)
    
    sorted_indices = np.argsort(class_weights)[::-1]
    
    for i in sorted_indices:
        cls_name = classes[i]
        count = sample_counts[i]
        weight = class_weights[i]
        
        print(f"{cls_name:<20} | {count:<15} | {weight:.4f}")
        
    print("="*65)
    print("\nKESIMPULAN:")
    print("1. Kelas dengan sampel SEDIKIT memiliki bobot > 1.0.")
    print("2. Kelas dengan sampel BANYAK memiliki bobot < 1.0.")
    print(f"3. Total Sampel: {len(dataset)}")
    print(f"4. Jumlah Kelas: {len(classes)}")

if __name__ == "__main__":
    print_weight_details()