#!/usr/bin/env python3
"""
Arquitectura Anti-Overfitting para Dataset Pequeño de Hologramas
Proyecto: Clasificación de células sanas vs SCD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class UltraLightCNN(nn.Module):
    """CNN ultra ligera para evitar overfitting"""
    
    def __init__(self, dropout_rate=0.7):
        super(UltraLightCNN, self).__init__()
        
        # Arquitectura muy simple pero con BatchNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(4, 4)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Capa intermedia pequeña
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        
        # Dropout extremo
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class SimpleFeatureNet(nn.Module):
    """Red muy simple para características"""
    
    def __init__(self, n_features=88, dropout_rate=0.7):
        super(SimpleFeatureNet, self).__init__()
        
        self.fc1 = nn.Linear(n_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CrossValidationTrainer:
    """Entrenador con validación cruzada k-fold"""
    
    def __init__(self, n_folds=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_folds = n_folds
        self.device = device
        self.fold_results = []
        
    def train_fold(self, model, train_loader, val_loader, epochs=50, lr=0.001):
        """Entrenar un fold específico"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Detectar tipo de modelo
                if isinstance(model, SimpleFeatureNet):
                    features = batch['features'].to(self.device)
                    outputs = model(features).squeeze()
                else:
                    images = batch['image'].to(self.device)
                    outputs = model(images).squeeze()
                    
                loss = criterion(outputs, labels)
                
                # L2 regularization adicional
                l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + 0.01 * l2_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_outputs = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    labels = batch['label'].to(self.device)
                    
                    # Detectar tipo de modelo
                    if isinstance(model, SimpleFeatureNet):
                        features = batch['features'].to(self.device)
                        outputs = model(features).squeeze()
                    else:
                        images = batch['image'].to(self.device)
                        outputs = model(images).squeeze()
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_outputs.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            train_acc = correct / total
            val_acc = val_correct / val_total
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_outputs = all_outputs
                best_labels = all_labels
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_outputs, best_labels
    
    def cross_validate_pytorch(self, X_images, X_features, y, model_class, model_kwargs):
        """Validación cruzada para modelos PyTorch"""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_predictions = []
        fold_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_images, y)):
            print(f"  Fold {fold + 1}/{self.n_folds}")
            
            # Crear datasets para este fold
            train_dataset = HologramDataset(
                X_images[train_idx], 
                X_features[train_idx], 
                y[train_idx]
            )
            val_dataset = HologramDataset(
                X_images[val_idx], 
                X_features[val_idx], 
                y[val_idx]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
            
            # Crear y entrenar modelo
            model = model_class(**model_kwargs).to(self.device)
            outputs, labels = self.train_fold(model, train_loader, val_loader)
            
            fold_predictions.extend(outputs)
            fold_labels.extend(labels)
        
        return np.array(fold_predictions), np.array(fold_labels)
    
    def cross_validate_sklearn(self, X_features, y, model):
        """Validación cruzada para modelos sklearn"""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_predictions = []
        fold_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
            X_train_fold = X_features[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_features[val_idx]
            y_val_fold = y[val_idx]
            
            # Entrenar modelo
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Predicciones
            if hasattr(model_fold, 'predict_proba'):
                y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
            else:
                y_pred_proba = model_fold.decision_function(X_val_fold)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            
            fold_predictions.extend(y_pred_proba)
            fold_labels.extend(y_val_fold)
        
        return np.array(fold_predictions), np.array(fold_labels)


class HologramDataset(Dataset):
    """Dataset simplificado"""
    
    def __init__(self, images, features, labels):
        self.images = torch.FloatTensor(images)
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'features': self.features[idx],
            'label': self.labels[idx]
        }


def create_ensemble_models():
    """Crear modelos para ensemble"""
    models = {
        'svm_linear': SVC(kernel='linear', C=0.1, probability=True, random_state=42),
        'svm_rbf': SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42),
        'random_forest': RandomForestClassifier(
            n_estimators=50, 
            max_depth=3, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='liblinear',
            random_state=42
        )
    }
    return models


def evaluate_model(y_true, y_pred_proba, model_name):
    """Evaluar modelo con métricas robustas"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Métricas
    report = classification_report(y_true, y_pred, target_names=['Healthy', 'SCD'], output_dict=True)
    
    # AUC-ROC
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    except:
        auc_score = 0.5
        fpr, tpr = [0, 1], [0, 1]
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nResultados {model_name}:")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  Precision (SCD): {report['SCD']['precision']:.4f}")
    print(f"  Recall (SCD): {report['SCD']['recall']:.4f}")
    print(f"  F1-Score (SCD): {report['SCD']['f1-score']:.4f}")
    
    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
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
    plt.savefig(f'{model_name}_cv_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'model': model_name,
        'accuracy': report['accuracy'],
        'auc': auc_score,
        'precision_scd': report['SCD']['precision'],
        'recall_scd': report['SCD']['recall'],
        'f1_scd': report['SCD']['f1-score']
    }


def main():
    """Función principal con validación cruzada"""
    print("ENTRENAMIENTO ROBUSTO CON VALIDACIÓN CRUZADA")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar SOLO los datos originales (sin augmentation)
    print("\nCargando dataset ORIGINAL (sin augmentation)...")
    data = np.load('preprocessed_dataset.npz')
    X_train_img = data['X_train'].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    X_test_img = data['X_test'].transpose(0, 3, 1, 2)
    y_train = data['y_train'].astype(np.float32)
    y_test = data['y_test'].astype(np.float32)
    
    # Cargar características
    feat_data = np.load('extracted_features.npz')
    # Usar solo las características de las imágenes originales
    X_train_feat = feat_data['X_train_features'][:len(y_train)]
    X_test_feat = feat_data['X_test_features']
    
    print(f"Dataset original:")
    print(f"  Train: {len(y_train)} imágenes")
    print(f"  Test: {len(y_test)} imágenes")
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)
    
    # Inicializar validador cruzado
    cv_trainer = CrossValidationTrainer(n_folds=5, device=device)
    
    results = []
    
         # 1. CNN Ultra Ligera
    print("\n" + "="*60)
    print("Modelo: CNN Ultra Ligera")
    y_pred_proba, y_true = cv_trainer.cross_validate_pytorch(
         X_train_img, X_train_feat_scaled, y_train,
         UltraLightCNN, {'dropout_rate': 0.5}
     )
    results.append(evaluate_model(y_true, y_pred_proba, 'CNN_UltraLight'))
     
     # 2. Red de Características Simple
    print("\n" + "="*60)
    print("Modelo: Red de Características Simple")
    y_pred_proba, y_true = cv_trainer.cross_validate_pytorch(
         X_train_img, X_train_feat_scaled, y_train,
         SimpleFeatureNet, {'n_features': X_train_feat.shape[1], 'dropout_rate': 0.5}
     )
    results.append(evaluate_model(y_true, y_pred_proba, 'SimpleFeatureNet'))
     
    # 3. Modelos tradicionales de ML
    sklearn_models = create_ensemble_models()
    
    for model_name, model in sklearn_models.items():
        print("\n" + "="*60)
        print(f"Modelo: {model_name}")
        y_pred_proba, y_true = cv_trainer.cross_validate_sklearn(
            X_train_feat_scaled, y_train, model
        )
        results.append(evaluate_model(y_true, y_pred_proba, model_name))
    
    # 4. Ensemble voting
    print("\n" + "="*60)
    print("Modelo: Ensemble Voting")
    
    # Entrenar ensemble con todos los datos de entrenamiento
    voting_clf = VotingClassifier(
        estimators=list(sklearn_models.items()),
        voting='soft',
        weights=[1, 1, 2, 2, 1]  # Más peso a RF y GB
    )
    voting_clf.fit(X_train_feat_scaled, y_train)
    
    # Evaluar en test real
    y_test_pred_proba = voting_clf.predict_proba(X_test_feat_scaled)[:, 1]
    results.append(evaluate_model(y_test, y_test_pred_proba, 'Ensemble_Voting_Test'))
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS (Validación Cruzada)")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Guardar resultados
    with open('cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r['model'] for r in results]
    aucs = [r['auc'] for r in results]
    
    bars = ax.bar(models, aucs)
    ax.set_ylabel('AUC Score')
    ax.set_title('Comparación de Modelos (Validación Cruzada)')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
    ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Aceptable (AUC=0.7)')
    
    # Colorear barras según rendimiento
    for bar, auc in zip(bars, aucs):
        if auc >= 0.8:
            bar.set_color('green')
        elif auc >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_cv.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nArchivos generados:")
    print("  • cv_results.json")
    print("  • model_comparison_cv.png")
    print("  • [model]_cv_evaluation.png para cada modelo")


if __name__ == "__main__":
    import pandas as pd
    main() 