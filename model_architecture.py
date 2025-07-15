#!/usr/bin/env python3
"""
Arquitectura CNN Ligera con Regularización Fuerte para Hologramas
Proyecto: Clasificación de células sanas vs SCD
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50V2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HologramClassifier:
    """
    Clasificador híbrido para hologramas con múltiples arquitecturas
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (512, 512, 3),
                 n_features: int = 88,
                 model_type: str = 'hybrid_cnn'):
        """
        Inicializar el clasificador
        
        Args:
            input_shape: Forma de las imágenes de entrada
            n_features: Número de características extraídas
            model_type: Tipo de modelo ('cnn', 'feature_based', 'hybrid_cnn', 'transfer_learning')
        """
        self.input_shape = input_shape
        self.n_features = n_features
        self.model_type = model_type
        self.model = None
        self.history = None
        self.feature_scaler = StandardScaler()
        
    def build_lightweight_cnn(self) -> models.Model:
        """
        Construir CNN ligera con regularización fuerte
        """
        model = models.Sequential([
            # Bloque 1 - Convolución inicial con regularización fuerte
            layers.Conv2D(16, (3, 3), padding='same', 
                         kernel_regularizer=regularizers.l2(0.01),
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 2 - Aumentar filtros gradualmente
            layers.Conv2D(32, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),
            
            # Bloque 3 - Características más complejas
            layers.Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Bloque 4 - Reducción progresiva
            layers.Conv2D(128, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.005)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Capas densas con regularización extrema
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            # Capa de salida
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_feature_based_model(self) -> models.Model:
        """
        Construir modelo basado solo en características extraídas
        """
        model = models.Sequential([
            layers.Input(shape=(self.n_features,)),
            
            # Red profunda con regularización
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(16, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Salida
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_hybrid_cnn(self) -> models.Model:
        """
        Construir modelo híbrido que combina CNN y características extraídas
        """
        # Entrada de imagen
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        # Rama CNN
        x = layers.Conv2D(16, (3, 3), padding='same', 
                         kernel_regularizer=regularizers.l2(0.01))(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((4, 4))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(32, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((4, 4))(x)
        x = layers.Dropout(0.35)(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        cnn_features = layers.Dropout(0.5)(x)
        
        # Entrada de características
        feature_input = layers.Input(shape=(self.n_features,), name='feature_input')
        
        # Rama de características
        y = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(feature_input)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        feature_processed = layers.Dropout(0.4)(y)
        
        # Concatenar ambas ramas
        combined = layers.concatenate([cnn_features, feature_processed])
        
        # Capas finales
        z = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.Dropout(0.5)(z)
        
        z = layers.Dense(32, kernel_regularizer=regularizers.l2(0.01))(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.Dropout(0.4)(z)
        
        # Salida
        output = layers.Dense(1, activation='sigmoid', name='output')(z)
        
        # Crear modelo
        model = models.Model(inputs=[image_input, feature_input], outputs=output)
        
        return model
    
    def build_transfer_learning_model(self, base_model_name: str = 'efficientnet') -> models.Model:
        """
        Construir modelo con transfer learning
        
        Args:
            base_model_name: 'efficientnet', 'mobilenet' o 'resnet'
        """
        # Seleccionar modelo base
        if base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:  # resnet
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Congelar capas iniciales (fine-tuning parcial)
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Construir modelo
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def compile_model(self, learning_rate: float = 0.0001):
        """
        Compilar el modelo con optimizador y métricas
        """
        # Construir modelo según tipo
        if self.model_type == 'cnn':
            self.model = self.build_lightweight_cnn()
        elif self.model_type == 'feature_based':
            self.model = self.build_feature_based_model()
        elif self.model_type == 'hybrid_cnn':
            self.model = self.build_hybrid_cnn()
        elif self.model_type.startswith('transfer_'):
            base_name = self.model_type.split('_')[1]
            self.model = self.build_transfer_learning_model(base_name)
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {self.model_type}")
        
        # Optimizador
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )
        
        # Compilar
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print(f"Modelo {self.model_type} compilado exitosamente")
        print(f"Total de parámetros: {self.model.count_params():,}")
        
    def create_callbacks(self, model_name: str) -> List[callbacks.Callback]:
        """
        Crear callbacks para el entrenamiento
        """
        callback_list = [
            # Early stopping con paciencia alta
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=25,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            
            # Reducir learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Guardar mejor modelo
            callbacks.ModelCheckpoint(
                f'best_{model_name}.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Tensorboard
            callbacks.TensorBoard(
                log_dir=f'logs/{model_name}',
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def prepare_data(self, X_train, X_test, y_train, y_test, 
                    X_train_features=None, X_test_features=None):
        """
        Preparar datos según el tipo de modelo
        """
        if self.model_type == 'feature_based':
            # Solo características
            X_train_scaled = self.feature_scaler.fit_transform(X_train_features)
            X_test_scaled = self.feature_scaler.transform(X_test_features)
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        elif self.model_type == 'hybrid_cnn':
            # Imágenes y características
            X_train_features_scaled = self.feature_scaler.fit_transform(X_train_features)
            X_test_features_scaled = self.feature_scaler.transform(X_test_features)
            return ([X_train, X_train_features_scaled], 
                   [X_test, X_test_features_scaled], 
                   y_train, y_test)
        
        else:
            # Solo imágenes
            return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
             epochs: int = 100, batch_size: int = 16):
        """
        Entrenar el modelo
        """
        # Callbacks
        callback_list = self.create_callbacks(self.model_type)
        
        # Datos de validación
        if X_val is None:
            validation_split = 0.2
            validation_data = None
        else:
            validation_split = None
            validation_data = (X_val, y_val)
        
        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1,
            class_weight={0: 1.0, 1: 1.2}  # Peso ligeramente mayor para SCD
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test, save_results: bool = True):
        """
        Evaluar el modelo y generar reportes
        """
        # Predicciones
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Métricas
        test_loss, test_acc, test_auc, test_precision, test_recall = self.model.evaluate(X_test, y_test)
        
        # Reporte de clasificación
        report = classification_report(y_test, y_pred, 
                                     target_names=['Healthy', 'SCD'],
                                     output_dict=True)
        
        # AUC-ROC
        auc_score = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nResultados del modelo {self.model_type}:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"AUC: {test_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        
        if save_results:
            self._save_evaluation_plots(cm, fpr, tpr, auc_score)
            self._save_training_history()
            
            # Guardar métricas
            metrics = {
                'model_type': self.model_type,
                'test_accuracy': float(test_acc),
                'test_auc': float(test_auc),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
                'classification_report': report
            }
            
            with open(f'{self.model_type}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _save_evaluation_plots(self, cm, fpr, tpr, auc_score):
        """
        Guardar gráficos de evaluación
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Healthy', 'SCD'],
                   yticklabels=['Healthy', 'SCD'],
                   ax=axes[0])
        axes[0].set_title(f'Matriz de Confusión - {self.model_type}')
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
        axes[1].set_title(f'Curva ROC - {self.model_type}')
        axes[1].legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_training_history(self):
        """
        Guardar historial de entrenamiento
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Train')
        axes[1, 0].plot(self.history.history['val_auc'], label='Val')
        axes[1, 0].set_title('AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.suptitle(f'Historial de Entrenamiento - {self.model_type}')
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_history.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """
    Función principal para entrenar y evaluar modelos
    """
    print("ENTRENAMIENTO DE MODELOS PARA CLASIFICACIÓN DE HOLOGRAMAS")
    print("=" * 60)
    
    # Cargar datos
    print("\nCargando datasets...")
    
    # Cargar imágenes augmentadas
    img_data = np.load('augmented_dataset.npz')
    X_train_img = img_data['X_train']
    X_test_img = img_data['X_test']
    y_train = img_data['y_train']
    y_test = img_data['y_test']
    
    # Cargar características extraídas
    feat_data = np.load('extracted_features.npz')
    X_train_feat = feat_data['X_train_features']
    X_test_feat = feat_data['X_test_features']
    
    print(f"Imágenes de entrenamiento: {X_train_img.shape}")
    print(f"Características de entrenamiento: {X_train_feat.shape}")
    print(f"Imágenes de prueba: {X_test_img.shape}")
    print(f"Características de prueba: {X_test_feat.shape}")
    
    # Modelos a entrenar
    models_to_train = [
        'cnn',
        'feature_based',
        'hybrid_cnn',
        'transfer_efficientnet'
    ]
    
    results = {}
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Entrenando modelo: {model_type}")
        print(f"{'='*60}")
        
        # Crear clasificador
        classifier = HologramClassifier(
            input_shape=(512, 512, 3),
            n_features=X_train_feat.shape[1],
            model_type=model_type
        )
        
        # Compilar modelo
        classifier.compile_model(learning_rate=0.0001)
        
        # Preparar datos
        if model_type == 'feature_based':
            X_train, X_test, y_train_model, y_test_model = classifier.prepare_data(
                X_train_img, X_test_img, y_train, y_test,
                X_train_feat, X_test_feat
            )
        elif model_type == 'hybrid_cnn':
            X_train, X_test, y_train_model, y_test_model = classifier.prepare_data(
                X_train_img, X_test_img, y_train, y_test,
                X_train_feat, X_test_feat
            )
        else:
            X_train, X_test, y_train_model, y_test_model = X_train_img, X_test_img, y_train, y_test
        
        # Entrenar
        history = classifier.train(
            X_train, y_train_model,
            epochs=100,
            batch_size=16
        )
        
        # Evaluar
        metrics = classifier.evaluate(X_test, y_test_model)
        results[model_type] = metrics
        
        # Guardar modelo
        classifier.model.save(f'{model_type}_model.h5')
        print(f"Modelo guardado: {model_type}_model.h5")
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    for model_type, metrics in results.items():
        print(f"\n{model_type}:")
        print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  AUC: {metrics['test_auc']:.4f}")
        print(f"  Precision: {metrics['test_precision']:.4f}")
        print(f"  Recall: {metrics['test_recall']:.4f}")
    
    # Guardar resumen
    with open('models_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEntrenamiento completado!")
    print("Archivos generados:")
    print("  • [model_type]_model.h5 - Modelos entrenados")
    print("  • [model_type]_metrics.json - Métricas de cada modelo")
    print("  • [model_type]_evaluation.png - Gráficos de evaluación")
    print("  • [model_type]_history.png - Historial de entrenamiento")
    print("  • models_summary.json - Resumen comparativo")


if __name__ == "__main__":
    main() 