import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import GarbageDataModule
from model import GarbageClassifier
import config

# Gunakan mapping yang sama
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

def inverse_normalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_hierarchy_predictions(num_images=16):
    data_module = GarbageDataModule()
    data_module.setup()
    test_loader = data_module.test_dataloader()
    class_names = data_module.class_names # 12 kelas
    
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.ckpt"
    model = GarbageClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.cuda()
    
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    plt.figure(figsize=(16, 16))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(rows, cols, i + 1)
        img = inverse_normalize(images[i])
        
        # Ambil nama sub-kelas
        true_sub = class_names[labels[i].item()]
        pred_sub = class_names[preds[i].item()]
        
        # Ambil nama parent-kelas
        true_parent = HIERARCHY_MAP[true_sub]
        pred_parent = HIERARCHY_MAP[pred_sub]
        
        # Logic warna: Hijau jika Sub-Kelas benar, Kuning jika Parent benar tapi Sub salah, Merah jika salah total
        if true_sub == pred_sub:
            color = 'green' # Benar Sempurna
            status = "Correct"
        elif true_parent == pred_parent:
            color = '#FFD700' # Gold/Kuning (Benar Kategori Besar, Salah Jenis)
            status = "Partial"
        else:
            color = 'red' # Salah Total
            status = "Wrong"
        
        plt.imshow(img)
        # Format Judul: PARENT (Sub)
        plt.title(f"T: {true_parent} ({true_sub})\nP: {pred_parent} ({pred_sub})", 
                  color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle("Visualisasi Klasifikasi Bertingkat\n(Hijau: Tepat, Kuning: Kategori Benar/Sub Salah, Merah: Salah Total)", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_hierarchy_predictions()