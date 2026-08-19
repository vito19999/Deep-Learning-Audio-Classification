# train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# === CONFIGURAZIONE ===
ANIMALS = ['dog', 'cat', 'chirping_birds', 'rooster', 'pig', 
           'cow', 'frog', 'crickets', 'crow', 'hen', 'sheep', 'insects']

BASE_PATH = "/media/neurone-pc5/Volume/AI & ML/Soundscapes/audio_sources/spettrogrammi"

# Semi per riproducibilità
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, split, max_samples=None):
        self.file_paths = []
        self.labels = []
        
        path = f"{BASE_PATH}/{split}"
        files = [f for f in os.listdir(path) if f.startswith('mix_')]
        files = sorted(files)
        
        if max_samples:
            files = files[:max_samples]
        
        print(f"Loading {len(files)} samples from {split}...", end=' ')
        for mix in tqdm(files, desc=f"{split}", ncols=60, leave=False):
            try:
                spec_path = f"{path}/{mix}/spectrogram.npz"
                label_path = f"{path}/{mix}/labels.npz"
                label_data = np.load(label_path, allow_pickle=True)
                label = str(label_data['class_name'])
                
                if label in ANIMALS:
                    self.file_paths.append(spec_path)
                    self.labels.append(ANIMALS.index(label))
            except:
                continue
        print(f"Ready: {len(self.file_paths)} samples")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        spec = np.load(self.file_paths[idx])["spectrogram"]
        spec = (spec + 80) / 80  # normalizzazione
        spec = spec[None, :, :]  # (1, H, W)
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# === TRAINING UTILS ===

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="Training Progress"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Loss durante il training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Accuracy durante il training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def train_phase(model, name, train_loader, val_loader, lr=1e-3, epochs=50, patience=10, device=None):
    """Training con early stopping"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {name}: LR={lr}, Epochs={epochs}, Device={device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Storici
    train_losses_history, val_losses_history = [], []
    train_acc_history, val_acc_history = [], []

    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        # === TRAIN ===
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}", ncols=80, leave=False)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            acc = train_correct / train_total
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{acc:.3f}'})

        # VALIDATION 
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # METRICHE 
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1:2d}: Train={train_acc:.3f} Val={val_acc:.3f} Loss={avg_val_loss:.3f}")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{name}_best.pth")
            print(f"   >>> New best model saved: {val_acc:.3f}")  
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   >>> Early stopping at epoch {epoch+1}")  
                break

    print(f"Best validation accuracy: {best_acc:.3f}")

    # Plot delle curve
    plot_training_curves(train_losses_history, val_losses_history, 
                         train_acc_history, val_acc_history, title=f"{name} - Training Curves")

    return best_acc, train_losses_history, val_losses_history, train_acc_history, val_acc_history

def test_model(model, model_path, test_loader, device=None):
    """Test finale con confusion matrix"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = [0] * len(ANIMALS)
    class_total = [0] * len(ANIMALS)

    all_predictions, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", ncols=60, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.3f}")

    # Per-class accuracy
    per_class = [(ANIMALS[i], class_correct[i]/class_total[i] if class_total[i] > 0 else 0) 
                 for i in range(len(ANIMALS))]
    per_class.sort(key=lambda x: x[1])
    print("Per-class accuracy (worst to best):")
    for animal, acc in per_class:
        count = class_total[ANIMALS.index(animal)]
        if count > 0:
            print(f"   {animal:15s}: {acc:.3f} ({class_correct[ANIMALS.index(animal)]}/{count})")

    # Confusion matrix
    unique_labels = sorted(list(set(all_labels)))
    animal_names = [ANIMALS[i] for i in unique_labels]
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=animal_names)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return test_acc
