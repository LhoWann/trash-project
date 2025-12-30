import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import GarbageDataModule
from model import GarbageClassifier
import config

def inverse_normalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_predictions(num_images=16):
    data_module = GarbageDataModule()
    data_module.setup()
    test_loader = data_module.test_dataloader()
    class_names = data_module.class_names
    
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
    
    plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(rows, cols, i + 1)
        img = inverse_normalize(images[i])
        
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_predictions()