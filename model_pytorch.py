#!/usr/bin/env python3
"""
Arquitectura CNN con PyTorch para Clasificación de Hologramas
Proyecto: Clasificación de células sanas vs SCD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.models as models
from torchvision.models import efficientnet_b0, mobilenet_v2, resnet50

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HologramDataset(Dataset):
    """Dataset personalizado para hologramas"""
    
    def __init__(self, images, features, labels, transform=None):
        self.images = torch.FloatTensor(images)
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'features': feature, 'label': label}


class LightweightCNN(nn.Module):
    """CNN ligera con regularización fuerte"""
    
    def __init__(self, dropout_rate=0.5):
        super(LightweightCNN, self).__init__()
        
        # Bloque 1
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque 3
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bloque 4
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Capas densas
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)
        self.dropout3 = nn.Dropout(dropout_rate * 0.8)
        self.dropout4 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Bloque 1
        x = F.relu(self.bn1(self.conv1_1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Bloque 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.bn2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Bloque 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.bn3(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Bloque 4
        x = F.relu(self.bn4(self.conv4_1(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout4(x)
        
        # Capas densas
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class FeatureBasedModel(nn.Module):
    """Modelo basado solo en características extraídas"""
    
    def __init__(self, n_features=88, dropout_rate=0.5):
        super(FeatureBasedModel, self).__init__()
        
        self.fc1 = nn.Linear(n_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 1)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        self.dropout4 = nn.Dropout(dropout_rate * 0.4)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc5(x))
        return x


class HybridCNN(nn.Module):
    """Modelo híbrido que combina CNN y características"""
    
    def __init__(self, n_features=88, dropout_rate=0.5):
        super(HybridCNN, self).__init__()
        
        # Rama CNN (simplificada)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(4, 4)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(4, 4)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Rama de características
        self.feature_fc = nn.Linear(n_features, 64)
        self.feature_bn = nn.BatchNorm1d(64)
        
        # Capas combinadas
        self.fc1 = nn.Linear(64 + 64, 64)  # CNN features + extracted features
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout
        self.dropout_cnn = nn.Dropout(dropout_rate * 0.5)
        self.dropout_features = nn.Dropout(dropout_rate * 0.4)
        self.dropout_final = nn.Dropout(dropout_rate)
        
    def forward(self, image, features):
        # Rama CNN
        x = F.relu(self.bn1(self.conv1(image)))
        x = self.pool1(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        cnn_features = self.dropout_cnn(x)
        
        # Rama de características
        feat = F.relu(self.feature_bn(self.feature_fc(features)))
        feat = self.dropout_features(feat)
        
        # Combinar
        combined = torch.cat([cnn_features, feat], dim=1)
        
        # Capas finales
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout_final(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_final(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class TransferLearningModel(nn.Module):
    """Transfer learning con modelos pre-entrenados"""
    
    def __init__(self, base_model='efficientnet', dropout_rate=0.5):
        super(TransferLearningModel, self).__init__()
        
        # Cargar modelo base
        if base_model == 'efficientnet':
            self.base_model = efficientnet_b0(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif base_model == 'mobilenet':
            self.base_model = mobilenet_v2(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:  # resnet
            self.base_model = resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        
        # Congelar capas iniciales
        for i, param in enumerate(self.base_model.parameters()):
            if i < len(list(self.base_model.parameters())) - 20:
                param.requires_grad = False
        
        # Capas personalizadas
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        
    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class HologramTrainer:
    """Clase para entrenar y evaluar modelos"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.scaler = StandardScaler()
        
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            if isinstance(self.model, HybridCNN):
                images = batch['image'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images, features).squeeze()
            else:
                if isinstance(self.model, FeatureBasedModel):
                    inputs = batch['features'].to(self.device)
                else:
                    inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(self.model, HybridCNN):
                    images = batch['image'].to(self.device)
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(images, features).squeeze()
                else:
                    if isinstance(self.model, FeatureBasedModel):
                        inputs = batch['features'].to(self.device)
                    else:
                        inputs = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(inputs).squeeze()
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(dataloader), correct / total, np.array(all_outputs), np.array(all_labels)
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.0001, patience=25):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                # Guardar mejor modelo
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Cargar mejor modelo
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return history
    
    def evaluate(self, test_loader, model_name):
        _, test_acc, y_pred_proba, y_true = self.validate(test_loader)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Métricas
        report = classification_report(y_true, y_pred, 
                                     target_names=['Healthy', 'SCD'],
                                     output_dict=True)
        
        auc_score = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nResultados del modelo {model_name}:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"Precision (SCD): {report['SCD']['precision']:.4f}")
        print(f"Recall (SCD): {report['SCD']['recall']:.4f}")
        
        # Guardar gráficos
        self._save_evaluation_plots(cm, fpr, tpr, auc_score, model_name)
        
        # Guardar métricas
        metrics = {
            'model_type': model_name,
            'test_accuracy': float(test_acc),
            'test_auc': float(auc_score),
            'classification_report': report
        }
        
        with open(f'{model_name}_metrics_pytorch.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _save_evaluation_plots(self, cm, fpr, tpr, auc_score, model_name):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Healthy', 'SCD'],
                   yticklabels=['Healthy', 'SCD'],
                   ax=axes[0])
        axes[0].set_title(f'Matriz de Confusión - {model_name}')
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Real')
        
        # Curva ROC
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'Curva ROC - {model_name}')
        axes[1].legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_evaluation_pytorch.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Función principal para entrenar modelos con PyTorch"""
    print("ENTRENAMIENTO DE MODELOS CON PYTORCH")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar datos
    print("\nCargando datasets...")
    img_data = np.load('augmented_dataset.npz')
    X_train_img = img_data['X_train'].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    X_test_img = img_data['X_test'].transpose(0, 3, 1, 2)
    y_train = img_data['y_train'].astype(np.float32)
    y_test = img_data['y_test'].astype(np.float32)
    
    feat_data = np.load('extracted_features.npz')
    X_train_feat = feat_data['X_train_features'].astype(np.float32)
    X_test_feat = feat_data['X_test_features'].astype(np.float32)
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)
    
    print(f"Imágenes de entrenamiento: {X_train_img.shape}")
    print(f"Características de entrenamiento: {X_train_feat.shape}")
    
    # Crear datasets
    train_dataset = HologramDataset(X_train_img, X_train_feat_scaled, y_train)
    test_dataset = HologramDataset(X_test_img, X_test_feat_scaled, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Modelos a entrenar
    models_config = [
        ('CNN_Ligera', LightweightCNN(dropout_rate=0.5)),
        ('Feature_Based', FeatureBasedModel(n_features=X_train_feat.shape[1], dropout_rate=0.5)),
        ('Hybrid_CNN', HybridCNN(n_features=X_train_feat.shape[1], dropout_rate=0.5)),
        ('Transfer_EfficientNet', TransferLearningModel(base_model='efficientnet', dropout_rate=0.5))
    ]
    
    results = {}
    
    for model_name, model in models_config:
        print(f"\n{'='*60}")
        print(f"Entrenando modelo: {model_name}")
        print(f"{'='*60}")
        
        trainer = HologramTrainer(model, device)
        
        # Entrenar
        history = trainer.train(train_loader, val_loader, epochs=100, lr=0.0001, patience=25)
        
        # Evaluar
        metrics = trainer.evaluate(test_loader, model_name)
        results[model_name] = metrics
        
        # Guardar modelo
        torch.save(model.state_dict(), f'{model_name}_pytorch.pth')
        print(f"Modelo guardado: {model_name}_pytorch.pth")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  AUC: {metrics['test_auc']:.4f}")
    
    with open('models_summary_pytorch.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEntrenamiento completado!")


if __name__ == "__main__":
    main() 