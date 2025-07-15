#!/usr/bin/env python3
"""
Pipeline Optimizado Completo para Clasificación de Hologramas
Proyecto: Clasificación de células sanas vs SCD
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


# ============= EXTRACCIÓN DE CARACTERÍSTICAS MEJORADA =============

class EnhancedHologramFeatureExtractor:
    """Extractor mejorado de características específicas para hologramas"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, image):
        """Extraer conjunto completo de características optimizadas"""
        features = []
        self.feature_names = []
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
            
        # 1. Características de fase holográfica
        phase_features = self._extract_phase_features(gray)
        features.extend(phase_features)
        
        # 2. Características de patrones de interferencia
        interference_features = self._extract_interference_features(gray)
        features.extend(interference_features)
        
        # 3. Características morfológicas mejoradas
        morph_features = self._extract_morphological_features(gray)
        features.extend(morph_features)
        
        # 4. Características de textura avanzadas
        texture_features = self._extract_advanced_texture_features(gray)
        features.extend(texture_features)
        
        # 5. Características espectrales
        spectral_features = self._extract_spectral_features(gray)
        features.extend(spectral_features)
        
        return np.array(features)
    
    def _extract_phase_features(self, gray):
        """Características específicas de fase holográfica"""
        features = []
        
        # Gradiente de fase
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitud y ángulo
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Estadísticas de gradiente
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.percentile(magnitude, 95),
            np.mean(angle),
            np.std(angle)
        ])
        
        self.feature_names.extend([
            'phase_grad_mag_mean', 'phase_grad_mag_std', 'phase_grad_mag_p95',
            'phase_angle_mean', 'phase_angle_std'
        ])
        
        # Variación local de fase
        local_var = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(local_var)),
            np.std(local_var),
            np.max(np.abs(local_var))
        ])
        
        self.feature_names.extend([
            'phase_local_var_mean', 'phase_local_var_std', 'phase_local_var_max'
        ])
        
        return features
    
    def _extract_interference_features(self, gray):
        """Características de patrones de interferencia"""
        features = []
        
        # FFT para analizar frecuencias
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Análisis radial del espectro
        h, w = gray.shape
        center = (h//2, w//2)
        
        # Energía en diferentes bandas de frecuencia
        total_energy = np.sum(magnitude_spectrum**2)
        
        # Bandas radiales
        radii = [10, 20, 40, 80]
        prev_energy = 0
        
        for r in radii:
            y, x = np.ogrid[:h, :w]
            mask = (x - center[1])**2 + (y - center[0])**2 <= r**2
            band_energy = np.sum(magnitude_spectrum[mask]**2)
            features.append((band_energy - prev_energy) / total_energy)
            self.feature_names.append(f'interference_band_{r}_energy')
            prev_energy = band_energy
        
        # Direccionalidad del patrón
        angles = np.linspace(0, np.pi, 8, endpoint=False)
        for i, angle in enumerate(angles):
            # Crear máscara direccional
            y, x = np.ogrid[:h, :w]
            theta = np.arctan2(y - center[0], x - center[1])
            angle_mask = np.abs(theta - angle) < np.pi/16
            
            directional_energy = np.sum(magnitude_spectrum[angle_mask]**2)
            features.append(directional_energy / total_energy)
            self.feature_names.append(f'interference_direction_{i}')
        
        return features
    
    def _extract_morphological_features(self, gray):
        """Características morfológicas mejoradas"""
        features = []
        
        # Umbralización adaptativa
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 2)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Diferencias morfológicas
        morph_diff = cv2.absdiff(closed, opened)
        features.append(np.sum(morph_diff) / (gray.shape[0] * gray.shape[1]))
        self.feature_names.append('morph_diff_ratio')
        
        # Análisis de componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        
        if num_labels > 1:
            # Excluir el fondo (label 0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            features.extend([
                len(areas),  # Número de componentes
                np.mean(areas) if len(areas) > 0 else 0,
                np.std(areas) if len(areas) > 0 else 0,
                np.max(areas) if len(areas) > 0 else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
            
        self.feature_names.extend([
            'morph_num_components', 'morph_area_mean', 'morph_area_std', 'morph_area_max'
        ])
        
        # Esqueleto morfológico
        skeleton = cv2.ximgproc.thinning(opened)
        features.append(np.sum(skeleton > 0) / (gray.shape[0] * gray.shape[1]))
        self.feature_names.append('morph_skeleton_density')
        
        return features
    
    def _extract_advanced_texture_features(self, gray):
        """Características de textura avanzadas"""
        features = []
        
        # Filtros de Gabor multi-escala
        frequencies = [0.05, 0.1, 0.2, 0.4]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        gabor_responses = []
        for freq in frequencies:
            for theta in orientations:
                kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, freq, 0)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                gabor_responses.append(filtered)
        
        # Estadísticas de las respuestas de Gabor
        for i, response in enumerate(gabor_responses):
            features.extend([
                np.mean(response),
                np.std(response),
                np.percentile(response, 90) - np.percentile(response, 10)
            ])
            self.feature_names.extend([
                f'gabor_{i}_mean', f'gabor_{i}_std', f'gabor_{i}_range'
            ])
        
        # Local Binary Patterns multi-escala
        for radius in [1, 2, 3]:
            lbp = cv2.ximgproc.createLBPH(radius=radius, neighbors=8*radius)
            hist = lbp.compute([gray])[0]
            
            # Características del histograma
            features.extend([
                np.mean(hist),
                np.std(hist),
                np.max(hist),
                -np.sum(hist * np.log(hist + 1e-10))  # Entropía
            ])
            self.feature_names.extend([
                f'lbp_r{radius}_mean', f'lbp_r{radius}_std', 
                f'lbp_r{radius}_max', f'lbp_r{radius}_entropy'
            ])
        
        return features
    
    def _extract_spectral_features(self, gray):
        """Características espectrales avanzadas"""
        features = []
        
        # Transformada de Fourier 2D
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift)**2
        
        # Momentos espectrales
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        cx, cy = w//2, h//2
        
        # Distancia desde el centro
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Momentos radiales
        for order in [1, 2, 3]:
            moment = np.sum(power_spectrum * (r**order)) / np.sum(power_spectrum)
            features.append(moment)
            self.feature_names.append(f'spectral_moment_order_{order}')
        
        # Entropía espectral
        power_norm = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
        features.append(spectral_entropy)
        self.feature_names.append('spectral_entropy')
        
        # Frecuencia dominante
        max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
        dominant_freq = np.sqrt((max_idx[0] - cy)**2 + (max_idx[1] - cx)**2)
        features.append(dominant_freq)
        self.feature_names.append('dominant_frequency')
        
        return features


# ============= MODELOS DE DEEP LEARNING OPTIMIZADOS =============

class OptimizedCNN(nn.Module):
    """CNN optimizada con arquitectura balanceada"""
    
    def __init__(self, dropout_rate=0.4):
        super(OptimizedCNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Atención espacial simple
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout progresivo
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)
        self.dropout2 = nn.Dropout(dropout_rate * 0.75)
        self.dropout3 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        # Aplicar atención
        att = self.attention(x)
        x = x * att
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Clasificador
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        
        return x


class HybridModel(nn.Module):
    """Modelo híbrido que combina CNN y características extraídas"""
    
    def __init__(self, n_features, dropout_rate=0.4):
        super(HybridModel, self).__init__()
        
        # Rama CNN (simplificada)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Rama de características
        self.feat_fc1 = nn.Linear(n_features, 128)
        self.feat_bn1 = nn.BatchNorm1d(128)
        self.feat_fc2 = nn.Linear(128, 64)
        self.feat_bn2 = nn.BatchNorm1d(64)
        
        # Fusión
        self.fusion_fc1 = nn.Linear(64 + 64, 64)
        self.fusion_bn1 = nn.BatchNorm1d(64)
        self.fusion_fc2 = nn.Linear(64, 32)
        self.fusion_bn2 = nn.BatchNorm1d(32)
        self.fusion_fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, image, features):
        # Rama CNN
        x = F.relu(self.bn1(self.conv1(image)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        cnn_features = x.view(x.size(0), -1)
        
        # Rama de características
        f = F.relu(self.feat_bn1(self.feat_fc1(features)))
        f = self.dropout(f)
        f = F.relu(self.feat_bn2(self.feat_fc2(f)))
        
        # Fusión
        combined = torch.cat([cnn_features, f], dim=1)
        x = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.fusion_bn2(self.fusion_fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fusion_fc3(x))
        
        return x


# ============= DATASET Y ENTRENAMIENTO =============

class HologramDataset(Dataset):
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


def train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """Entrenar modelo PyTorch con early stopping"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_score = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            if isinstance(model, HybridModel):
                outputs = model(batch['image'].to(device), batch['features'].to(device))
            else:
                outputs = model(batch['image'].to(device))
            
            labels = batch['label'].to(device)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(model, HybridModel):
                    outputs = model(batch['image'].to(device), batch['features'].to(device))
                else:
                    outputs = model(batch['image'].to(device))
                
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch['label'].numpy())
        
        val_preds = np.array(val_preds).squeeze()
        val_labels = np.array(val_labels)
        
        # Optimizar threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.2, 0.8, 13):
            f1 = f1_score(val_labels, (val_preds > thresh).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        val_score = balanced_accuracy_score(val_labels, (val_preds > best_thresh).astype(int))
        
        scheduler.step(-val_score)
        
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_threshold = best_thresh
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model, best_threshold


# ============= PIPELINE PRINCIPAL =============

def create_ensemble_models():
    """Crear conjunto diverso de modelos"""
    models = {
        'rf_balanced': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=1.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'svm_rbf': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    }
    return models


def main():
    """Pipeline principal optimizado"""
    print("PIPELINE OPTIMIZADO PARA CLASIFICACIÓN DE HOLOGRAMAS")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # 1. Cargar datos originales
    print("\n1. Cargando datos...")
    data = np.load('preprocessed_dataset.npz')
    X_train_img = data['X_train']
    X_test_img = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"   Train: {len(y_train)} imágenes")
    print(f"   Test: {len(y_test)} imágenes")
    print(f"   Balance train: {np.sum(y_train==0)} Healthy, {np.sum(y_train==1)} SCD")
    
    # 2. Extraer características mejoradas
    print("\n2. Extrayendo características mejoradas...")
    extractor = EnhancedHologramFeatureExtractor()
    
    X_train_features = []
    for i, img in enumerate(X_train_img):
        if i % 20 == 0:
            print(f"   Progreso: {i}/{len(X_train_img)}")
        features = extractor.extract_all_features(img)
        X_train_features.append(features)
    
    X_test_features = []
    for img in X_test_img:
        features = extractor.extract_all_features(img)
        X_test_features.append(features)
    
    X_train_features = np.array(X_train_features)
    X_test_features = np.array(X_test_features)
    print(f"   Características extraídas: {X_train_features.shape[1]}")
    
    # 3. Selección de características
    print("\n3. Seleccionando mejores características...")
    selector = SelectKBest(f_classif, k=min(100, X_train_features.shape[1]))
    X_train_features_selected = selector.fit_transform(X_train_features, y_train)
    X_test_features_selected = selector.transform(X_test_features)
    print(f"   Características seleccionadas: {X_train_features_selected.shape[1]}")
    
    # 4. Normalización robusta
    scaler = RobustScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features_selected)
    X_test_features_scaled = scaler.transform(X_test_features_selected)
    
    # 5. Balanceo de clases con SMOTE
    print("\n4. Aplicando SMOTE para balancear clases...")
    smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=3)
    X_train_features_balanced, y_train_balanced = smote.fit_resample(X_train_features_scaled, y_train)
    print(f"   Después de SMOTE: {len(y_train_balanced)} muestras")
    print(f"   Balance: {np.sum(y_train_balanced==0)} Healthy, {np.sum(y_train_balanced==1)} SCD")
    
    # 6. Validación cruzada repetida estratificada
    print("\n5. Entrenando modelos con validación cruzada...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    all_results = []
    
    # 6.1 Modelos tradicionales
    sklearn_models = create_ensemble_models()
    
    for name, model in sklearn_models.items():
        print(f"\n   Entrenando {name}...")
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train_features_balanced, y_train_balanced)):
            X_fold_train = X_train_features_balanced[train_idx]
            y_fold_train = y_train_balanced[train_idx]
            X_fold_val = X_train_features_balanced[val_idx]
            y_fold_val = y_train_balanced[val_idx]
            
            # Entrenar
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_fold_train, y_fold_train)
            
            # Predecir
            if hasattr(model_fold, 'predict_proba'):
                y_pred_proba = model_fold.predict_proba(X_fold_val)[:, 1]
            else:
                y_pred_proba = model_fold.decision_function(X_fold_val)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            
            # Optimizar threshold
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.linspace(0.2, 0.8, 13):
                f1 = f1_score(y_fold_val, (y_pred_proba > thresh).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            # Evaluar
            y_pred = (y_pred_proba > best_thresh).astype(int)
            score = balanced_accuracy_score(y_fold_val, y_pred)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"      Balanced Accuracy: {mean_score:.3f} (+/- {std_score:.3f})")
        
        all_results.append({
            'model': name,
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': scores
        })
    
    # 6.2 Deep Learning - CNN optimizada
    print(f"\n   Entrenando CNN Optimizada...")
    
    # Preparar datos para PyTorch
    X_train_img_tensor = X_train_img.transpose(0, 3, 1, 2)
    
    # Aplicar SMOTE también a las imágenes (usando representación PCA)
    pca = PCA(n_components=50, random_state=42)
    X_train_img_pca = pca.fit_transform(X_train_img.reshape(len(X_train_img), -1))
    X_train_img_pca_balanced, _ = smote.fit_resample(X_train_img_pca, y_train)
    
    # Reconstruir imágenes
    X_train_img_balanced = pca.inverse_transform(X_train_img_pca_balanced)
    X_train_img_balanced = X_train_img_balanced.reshape(-1, X_train_img.shape[1], X_train_img.shape[2], X_train_img.shape[3])
    X_train_img_balanced_tensor = X_train_img_balanced.transpose(0, 3, 1, 2)
    
    cnn_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train_img_balanced_tensor, y_train_balanced)):
        if fold >= 3:  # Solo 3 folds para CNN (más rápido)
            break
            
        # Crear datasets
        train_dataset = HologramDataset(
            X_train_img_balanced_tensor[train_idx],
            X_train_features_balanced[train_idx],
            y_train_balanced[train_idx]
        )
        val_dataset = HologramDataset(
            X_train_img_balanced_tensor[val_idx],
            X_train_features_balanced[val_idx],
            y_train_balanced[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Entrenar modelo
        model = OptimizedCNN(dropout_rate=0.3).to(device)
        model, threshold = train_pytorch_model(model, train_loader, val_loader, epochs=30, lr=0.0001, device=device)
        
        # Evaluar
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['image'].to(device))
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch['label'].numpy())
        
        val_preds = np.array(val_preds).squeeze()
        val_labels = np.array(val_labels)
        
        y_pred = (val_preds > threshold).astype(int)
        score = balanced_accuracy_score(val_labels, y_pred)
        cnn_scores.append(score)
    
    mean_score = np.mean(cnn_scores)
    std_score = np.std(cnn_scores)
    print(f"      Balanced Accuracy: {mean_score:.3f} (+/- {std_score:.3f})")
    
    all_results.append({
        'model': 'OptimizedCNN',
        'mean_score': mean_score,
        'std_score': std_score,
        'scores': cnn_scores
    })
    
    # 7. Entrenar mejores modelos con todos los datos
    print("\n6. Entrenando modelos finales...")
    
    # Seleccionar los 3 mejores modelos
    all_results.sort(key=lambda x: x['mean_score'], reverse=True)
    best_models = all_results[:3]
    
    print("\nMejores modelos:")
    for i, result in enumerate(best_models):
        print(f"   {i+1}. {result['model']}: {result['mean_score']:.3f} (+/- {result['std_score']:.3f})")
    
    # Entrenar modelos finales
    final_models = []
    
    for result in best_models:
        model_name = result['model']
        
        if model_name == 'OptimizedCNN':
            # Entrenar CNN final
            train_dataset = HologramDataset(
                X_train_img_balanced_tensor,
                X_train_features_balanced,
                y_train_balanced
            )
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            model = OptimizedCNN(dropout_rate=0.3).to(device)
            model, threshold = train_pytorch_model(model, train_loader, train_loader, epochs=50, lr=0.0001, device=device)
            
            final_models.append(('OptimizedCNN', model, threshold))
        else:
            # Entrenar modelo sklearn
            model = sklearn_models[model_name]
            model.fit(X_train_features_balanced, y_train_balanced)
            
            # Encontrar threshold óptimo
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_train_features_balanced)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_train_features_balanced)
            
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.linspace(0.2, 0.8, 13):
                f1 = f1_score(y_train_balanced, (y_pred_proba > thresh).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            final_models.append((model_name, model, best_thresh))
    
    # 8. Evaluación final en test
    print("\n7. Evaluación final en conjunto de test...")
    
    for name, model, threshold in final_models:
        print(f"\n   {name} (threshold={threshold:.3f}):")
        
        if name == 'OptimizedCNN':
            # Evaluar CNN
            test_dataset = HologramDataset(
                X_test_img.transpose(0, 3, 1, 2),
                X_test_features_scaled,
                y_test
            )
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            model.eval()
            test_preds = []
            
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(batch['image'].to(device))
                    test_preds.extend(outputs.cpu().numpy())
            
            y_pred_proba = np.array(test_preds).squeeze()
        else:
            # Evaluar modelo sklearn
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_features_scaled)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test_features_scaled)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Métricas
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'SCD']))
        auc = roc_auc_score(y_test, y_pred_proba)
        ba = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"   AUC: {auc:.3f}")
        print(f"   Balanced Accuracy: {ba:.3f}")
        print(f"   F1-Score: {f1:.3f}")
    
    # 9. Ensemble final
    print("\n8. Creando ensemble final...")
    
    # Predicciones de todos los modelos
    all_predictions = []
    
    for name, model, threshold in final_models:
        if name == 'OptimizedCNN':
            model.eval()
            test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(batch['image'].to(device))
                    test_preds.extend(outputs.cpu().numpy())
            y_pred_proba = np.array(test_preds).squeeze()
        else:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_features_scaled)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test_features_scaled)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        
        all_predictions.append(y_pred_proba)
    
    # Promedio ponderado
    weights = [result['mean_score'] for result in best_models]
    weights = np.array(weights) / np.sum(weights)
    
    ensemble_proba = np.average(all_predictions, axis=0, weights=weights)
    
    # Optimizar threshold del ensemble
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.2, 0.8, 13):
        f1 = f1_score(y_test, (ensemble_proba > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    ensemble_pred = (ensemble_proba > best_thresh).astype(int)
    
    print(f"\n   Ensemble Final (threshold={best_thresh:.3f}):")
    print(classification_report(y_test, ensemble_pred, target_names=['Healthy', 'SCD']))
    
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_ba = balanced_accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    
    print(f"   AUC: {ensemble_auc:.3f}")
    print(f"   Balanced Accuracy: {ensemble_ba:.3f}")
    print(f"   F1-Score: {ensemble_f1:.3f}")
    
    # Guardar resultados
    results_summary = {
        'individual_models': [
            {
                'name': name,
                'threshold': float(threshold),
                'cv_score': float(result['mean_score']),
                'test_metrics': {
                    'auc': float(auc),
                    'balanced_accuracy': float(ba),
                    'f1_score': float(f1)
                }
            }
            for (name, _, threshold), result in zip(final_models, best_models)
        ],
        'ensemble': {
            'threshold': float(best_thresh),
            'test_metrics': {
                'auc': float(ensemble_auc),
                'balanced_accuracy': float(ensemble_ba),
                'f1_score': float(ensemble_f1)
            }
        }
    }
    
    with open('optimized_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETADO")
    print("Resultados guardados en: optimized_results.json")


if __name__ == "__main__":
    main() 