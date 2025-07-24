#!/usr/bin/env python3
"""
Pipeline de Preprocesamiento para Hologramas de Glóbulos Sanguíneos
Proyecto: Clasificación de células sanas vs SCD

Este módulo implementa técnicas especializadas para el preprocesamiento
de imágenes holográficas con el objetivo de mejorar la calidad y 
facilitar el entrenamiento del clasificador.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class HologramPreprocessor:
    """
    Clase para el preprocesamiento especializado de hologramas
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Inicializar el preprocesador
        
        Args:
            target_size: Tamaño objetivo para redimensionar las imágenes
        """
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
        self.preprocessing_stats = {}
        
    def load_and_preprocess_image(self, image_path: str, apply_hologram_enhancement: bool = True) -> np.ndarray:
        """
        Cargar y preprocesar una sola imagen
        
        Args:
            image_path: Ruta de la imagen
            apply_hologram_enhancement: Si aplicar mejoras específicas para hologramas
            
        Returns:
            Imagen preprocesada como array numpy
        """
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
        # Convertir a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aplicar pipeline de preprocesamiento
        processed_img = self._preprocess_pipeline(img_rgb, apply_hologram_enhancement)
        
        return processed_img
    
    def _preprocess_pipeline(self, img: np.ndarray, apply_hologram_enhancement: bool = True) -> np.ndarray:
        """
        Pipeline completo de preprocesamiento
        
        Args:
            img: Imagen original
            apply_hologram_enhancement: Si aplicar mejoras específicas para hologramas
            
        Returns:
            Imagen preprocesada
        """
        
        img_resized = self._smart_resize(img)
        
        if apply_hologram_enhancement:
            img_enhanced = self._hologram_enhancement(img_resized)
        else:
            img_enhanced = img_resized
            
        img_normalized = self._normalize_image(img_enhanced)
        
        img_contrast = self._adaptive_contrast_enhancement(img_normalized)
        
        return img_contrast
    
    def _smart_resize(self, img: np.ndarray) -> np.ndarray:
        """
        Redimensionado inteligente que preserva aspectos importantes
        
        Args:
            img: Imagen original
            
        Returns:
            Imagen redimensionada
        """
        h, w = img.shape[:2]
        target_h, target_w = self.target_size
        
        
        aspect_ratio = w / h
        target_aspect_ratio = target_w / target_h
        
        if aspect_ratio > target_aspect_ratio:
            # Imagen más ancha, ajustar por ancho
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # Imagen más alta, ajustar por alto
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        # Redimensionar manteniendo proporciones
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        #   (padding)
        if new_h < target_h or new_w < target_w:
            # Crear imagen con padding
            padded_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            
            # Calcular posición para centrar
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            # Colocar imagen redimensionada en el centro
            padded_img[start_y:start_y+new_h, start_x:start_x+new_w] = img_resized
            img_resized = padded_img
        
        return img_resized
    
    def _hologram_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Mejoras específicas para imágenes holográficas
        
        Args:
            img: Imagen de entrada
            
        Returns:
            Imagen con mejoras holográficas
        """
        # Convertir a escala de grises para análisis
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # 1. Filtro de realce de patrones de interferencia
        # Usar filtro gaussiano inverso para realzar detalles finos
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        enhanced = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # 2. Filtro de frecuencia para realzar patrones holográficos
        # Aplicar FFT para trabajar en dominio de frecuencia
        f_transform = np.fft.fft2(enhanced)
        f_shift = np.fft.fftshift(f_transform)
        
        # Crear máscara de paso alto para realzar detalles
        rows, cols = enhanced.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # Radio del filtro
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0
        
        # Aplicar máscara y transformada inversa
        f_shift_masked = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        # Normalizar resultado
        img_back = ((img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255).astype(np.uint8)
        
        # 3. Combinar imagen original con mejoras
        # Usar imagen mejorada como canal de intensidad
        if len(img.shape) == 3:
            # Mantener información de color pero mejorar detalles
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.addWeighted(hsv[:, :, 2], 0.7, img_back, 0.3, 0)
            enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return enhanced_color
        else:
            return img_back
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalización específica para hologramas
        
        Args:
            img: Imagen a normalizar
            
        Returns:
            Imagen normalizada
        """
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Normalización por canal
        if len(img.shape) == 3:
            # Para imágenes a color, normalizar cada canal independientemente
            normalized = np.zeros_like(img)
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                # Normalización robusta usando percentiles para evitar outliers
                p2, p98 = np.percentile(channel, (2, 98))
                normalized[:, :, i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
        else:
            # Para escala de grises
            p2, p98 = np.percentile(img, (2, 98))
            normalized = np.clip((img - p2) / (p98 - p2), 0, 1)
        
        return normalized
    
    def _adaptive_contrast_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Mejora de contraste adaptativo
        
        Args:
            img: Imagen normalizada
            
        Returns:
            Imagen con contraste mejorado
        """
        # Convertir a uint8 para CLAHE
        if img.dtype == np.float32:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        if len(img.shape) == 3:
            # Para imágenes a color, aplicar CLAHE en espacio LAB
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Para escala de grises
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
        
        # Convertir de vuelta a float32 normalizado
        return enhanced.astype(np.float32) / 255.0
    
    def load_dataset(self, dataset_path: str = "dataset", test_size: float = 0.2, 
                    random_state: int = 42, apply_hologram_enhancement: bool = True) -> Dict:
        """
        Cargar y preprocesar todo el dataset
        
        Args:
            dataset_path: Ruta del dataset
            test_size: Proporción del conjunto de prueba
            random_state: Semilla para reproducibilidad
            apply_hologram_enhancement: Si aplicar mejoras holográficas
            
        Returns:
            Diccionario con datos preprocesados
        """
        print("🔄 Cargando y preprocesando dataset...")
        
        # Cargar rutas de archivos y etiquetas
        image_paths = []
        labels = []
        
        dataset_path_obj = Path(dataset_path)
        classes = ["SCD", "Healthy"]
        
        for class_name in classes:
            class_path = dataset_path_obj / class_name
            if class_path.exists():
                image_files = list(class_path.glob("*.png"))
                print(f"   └─ {class_name}: {len(image_files)} imágenes")
                
                for img_path in image_files:
                    image_paths.append(str(img_path))
                    labels.append(class_name)
        
        # Codificar etiquetas
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # División estratificada
        X_train_paths, X_test_paths, y_train, y_test = train_test_split(
            image_paths, labels_encoded, test_size=test_size, 
            random_state=random_state, stratify=labels_encoded
        )
        
        print(f"\n📊 División del dataset:")
        print(f"   • Entrenamiento: {len(X_train_paths)} imágenes")
        print(f"   • Prueba: {len(X_test_paths)} imágenes")
        
        # Preprocesar imágenes
        print("\n🔄 Preprocesando imágenes...")
        
        X_train = []
        X_test = []
        
        # Procesar conjunto de entrenamiento
        print("   └─ Procesando conjunto de entrenamiento...")
        for i, path in enumerate(X_train_paths):
            if i % 20 == 0:
                print(f"      Progreso: {i+1}/{len(X_train_paths)}")
            processed_img = self.load_and_preprocess_image(path, apply_hologram_enhancement)
            X_train.append(processed_img)
        
        # Procesar conjunto de prueba
        print("   └─ Procesando conjunto de prueba...")
        for i, path in enumerate(X_test_paths):
            if i % 10 == 0:
                print(f"      Progreso: {i+1}/{len(X_test_paths)}")
            processed_img = self.load_and_preprocess_image(path, apply_hologram_enhancement)
            X_test.append(processed_img)
        
        # Convertir a arrays numpy
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Guardar estadísticas de preprocesamiento
        self.preprocessing_stats = {
            'target_size': self.target_size,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'train_distribution': {
                class_name: int(np.sum(y_train == i)) 
                for i, class_name in enumerate(self.label_encoder.classes_)
            },
            'test_distribution': {
                class_name: int(np.sum(y_test == i)) 
                for i, class_name in enumerate(self.label_encoder.classes_)
            },
            'hologram_enhancement': apply_hologram_enhancement
        }
        
        print(f"\n✅ Preprocesamiento completado!")
        print(f"   • Forma X_train: {X_train.shape}")
        print(f"   • Forma X_test: {X_test.shape}")
        print(f"   • Forma y_train: {y_train.shape}")
        print(f"   • Forma y_test: {y_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_paths': X_train_paths,
            'test_paths': X_test_paths,
            'label_encoder': self.label_encoder,
            'preprocessing_stats': self.preprocessing_stats
        }
    
    def visualize_preprocessing_effects(self, original_img_path: str, save_path: str = "preprocessing_comparison.png"):
        """
        Visualizar los efectos del preprocesamiento
        
        Args:
            original_img_path: Ruta de una imagen original
            save_path: Ruta donde guardar la comparación
        """
        print("🖼️ Generando visualización de efectos de preprocesamiento...")
        
        # Cargar imagen original
        original_img = cv2.imread(original_img_path)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Aplicar diferentes etapas del preprocesamiento
        step1_resized = self._smart_resize(original_rgb)
        step2_enhanced = self._hologram_enhancement(step1_resized)
        step3_normalized = self._normalize_image(step2_enhanced)
        step4_contrast = self._adaptive_contrast_enhancement(step3_normalized)
        
        # Crear visualización
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Efectos del Pipeline de Preprocesamiento', fontsize=16, fontweight='bold')
        
        # Imagen original (redimensionada para visualización)
        original_display = cv2.resize(original_rgb, (400, 300))
        axes[0, 0].imshow(original_display)
        axes[0, 0].set_title('Original\n(Redimensionada para display)')
        axes[0, 0].axis('off')
        
        # Redimensionado inteligente
        axes[0, 1].imshow(step1_resized)
        axes[0, 1].set_title(f'Redimensionado Inteligente\n{step1_resized.shape[:2]}')
        axes[0, 1].axis('off')
        
        # Mejoras holográficas
        axes[0, 2].imshow(step2_enhanced)
        axes[0, 2].set_title('Mejoras Holográficas\n(Realce de patrones)')
        axes[0, 2].axis('off')
        
        # Normalización
        axes[1, 0].imshow(step3_normalized)
        axes[1, 0].set_title('Normalización\n(Percentiles 2-98)')
        axes[1, 0].axis('off')
        
        # Mejora de contraste
        axes[1, 1].imshow(step4_contrast)
        axes[1, 1].set_title('Contraste Adaptativo\n(CLAHE)')
        axes[1, 1].axis('off')
        
        # Histograma de comparación
        axes[1, 2].hist(original_rgb.flatten(), bins=50, alpha=0.7, label='Original', color='red', density=True)
        axes[1, 2].hist((step4_contrast * 255).astype(np.uint8).flatten(), bins=50, alpha=0.7, label='Procesada', color='blue', density=True)
        axes[1, 2].set_title('Comparación de Histogramas')
        axes[1, 2].set_xlabel('Intensidad de Píxel')
        axes[1, 2].set_ylabel('Densidad')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualización guardada en: {save_path}")
    
    def save_preprocessing_config(self, filepath: str = "preprocessing_config.json"):
        """
        Guardar configuración de preprocesamiento
        
        Args:
            filepath: Ruta donde guardar la configuración
        """
        config = {
            'target_size': self.target_size,
            'preprocessing_stats': self.preprocessing_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuración guardada en: {filepath}")

def main():
    """
    Función principal para demostrar el pipeline de preprocesamiento
    """
    print("🔬 PIPELINE DE PREPROCESAMIENTO PARA HOLOGRAMAS")
    print("=" * 60)
    
    # Inicializar preprocesador
    preprocessor = HologramPreprocessor(target_size=(512, 512))
    
    # Cargar y preprocesar dataset completo
    dataset = preprocessor.load_dataset(
        dataset_path="dataset",
        test_size=0.2,
        random_state=42,
        apply_hologram_enhancement=True
    )
    
    # Describe
    print("\n ESTADÍSTICAS DEL DATASET PREPROCESADO:")
    print("=" * 50)
    stats = dataset['preprocessing_stats']
    print(f"Tamaño objetivo: {stats['target_size']}")
    print(f"Total de clases: {stats['n_classes']}")
    print(f"Clases: {', '.join(stats['classes'])}")
    print(f"\nDistribución de entrenamiento:")
    for class_name, count in stats['train_distribution'].items():
        print(f"   {class_name}: {count} imágenes")
    print(f"\nDistribución de prueba:")
    for class_name, count in stats['test_distribution'].items():
        print(f"   {class_name}: {count} imágenes")
    
    # Mostrar 2
    sample_paths = dataset['train_paths'][:2] 
    for i, path in enumerate(sample_paths):
        save_path = f"preprocessing_example_{i+1}.png"
        preprocessor.visualize_preprocessing_effects(path, save_path)
    
    # Guardar configuración
    preprocessor.save_preprocessing_config()
    
    # Guardar dataset preprocesado
    print("\n Guardando dataset preprocesado...")
    np.savez_compressed(
        'preprocessed_dataset.npz',
        X_train=dataset['X_train'],
        X_test=dataset['X_test'],
        y_train=dataset['y_train'],
        y_test=dataset['y_test']
    )
    
    print("✅ Pipeline de preprocesamiento completado!")
    print("📁 Archivos generados:")
    print("   • preprocessing_example_1.png")
    print("   • preprocessing_example_2.png") 
    print("   • preprocessing_config.json")
    print("   • preprocessed_dataset.npz")

if __name__ == "__main__":
    main() 