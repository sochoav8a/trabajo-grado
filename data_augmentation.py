#!/usr/bin/env python3
"""
Data Augmentation Especializado para Hologramas de Glóbulos Sanguíneos
Proyecto: Clasificación de células sanas vs SCD

Este módulo implementa técnicas avanzadas de data augmentation específicamente
diseñadas para hologramas, maximizando la variabilidad del dataset sin perder
características importantes para la clasificación.
"""

import numpy as np
import cv2
import random
from typing import Tuple, List, Dict, Callable, Union
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform, filters, exposure
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

class HologramAugmentator:
    """
    Clase para data augmentation especializado en hologramas
    """
    
    def __init__(self, 
                 augmentation_factor: int = 8,
                 preserve_hologram_features: bool = True,
                 apply_advanced_techniques: bool = True):
        """
        Inicializar el augmentador
        
        Args:
            augmentation_factor: Factor de multiplicación del dataset
            preserve_hologram_features: Si preservar características holográficas
            apply_advanced_techniques: Si aplicar técnicas avanzadas (Mixup, CutMix)
        """
        self.augmentation_factor = augmentation_factor
        self.preserve_hologram_features = preserve_hologram_features
        self.apply_advanced_techniques = apply_advanced_techniques
        self.augmentation_stats = {}
        
        # Configurar pipelines de augmentation
        self.setup_augmentation_pipelines()
    
    def setup_augmentation_pipelines(self):
        """Configurar diferentes pipelines de augmentation"""
        
        # Pipeline básico - transformaciones geométricas
        self.basic_augmentations = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=50, 
                sigma=5, 
                alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            )
        ])
        
        # Pipeline de mejoras visuales - preserva patrones holográficos
        self.visual_augmentations = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, 
                contrast_limit=0.15, 
                p=0.7
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
            A.Blur(blur_limit=3, p=0.2)
        ])
        
        # Pipeline avanzado - técnicas específicas para hologramas
        self.hologram_augmentations = A.Compose([
            A.OpticalDistortion(
                distort_limit=0.1, 
                shift_limit=0.05,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5, 
                distort_limit=0.1,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            ),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2)
        ])
        
        # Pipeline combinado
        self.combined_pipeline = A.Compose([
            A.OneOf([
                self.basic_augmentations,
                self.visual_augmentations,
                self.hologram_augmentations
            ], p=1.0),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.NoOp()
            ], p=0.5)
        ])
    
    def frequency_domain_augmentation(self, image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Augmentation en el dominio de frecuencia específico para hologramas
        
        Args:
            image: Imagen de entrada
            intensity: Intensidad de la transformación
            
        Returns:
            Imagen transformada
        """
        if len(image.shape) == 3:
            # Para imágenes a color, trabajar con canal de luminancia
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Aplicar FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Crear perturbación en el dominio de frecuencia
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Generar ruido aleatorio en frecuencias específicas
        noise_mask = np.random.normal(1.0, intensity, (rows, cols))
        
        # Aplicar máscara para preservar frecuencias centrales (características principales)
        preservation_radius = min(rows, cols) // 8
        y, x = np.ogrid[:rows, :cols]
        center_mask = (x - ccol) ** 2 + (y - crow) ** 2 <= preservation_radius ** 2
        noise_mask[center_mask] = 1.0
        
        # Aplicar perturbación
        f_shift_augmented = f_shift * noise_mask
        
        # Transformada inversa
        f_ishift = np.fft.ifftshift(f_shift_augmented)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        # Normalizar y convertir de vuelta
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            # Reconstituir imagen a color
            result = image.copy()
            # Aplicar transformación al canal de luminancia en espacio HSV
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = img_back
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            return result
        else:
            return img_back.astype(np.float32) / 255.0
    
    def interference_pattern_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Augmentation que simula variaciones en patrones de interferencia
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Imagen con patrones de interferencia modificados
        """
        h, w = image.shape[:2]
        
        # Generar patrón sinusoidal aleatorio
        frequency = random.uniform(0.5, 2.0)
        phase = random.uniform(0, 2 * np.pi)
        amplitude = random.uniform(0.02, 0.08)
        
        # Crear coordenadas
        x = np.linspace(0, 2 * np.pi * frequency, w)
        y = np.linspace(0, 2 * np.pi * frequency, h)
        X, Y = np.meshgrid(x, y)
        
        # Generar patrón de interferencia
        interference = amplitude * np.sin(X + phase) * np.sin(Y + phase)
        
        # Aplicar el patrón
        if len(image.shape) == 3:
            interference = np.expand_dims(interference, axis=2)
            interference = np.repeat(interference, 3, axis=2)
        
        augmented = image + interference
        return np.clip(augmented, 0, 1)
    
    def mixup_augmentation(self, image1: np.ndarray, image2: np.ndarray, 
                          label1: int, label2: int, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        Implementación de Mixup para hologramas
        
        Args:
            image1, image2: Imágenes a mezclar
            label1, label2: Etiquetas correspondientes
            alpha: Parámetro de la distribución Beta
            
        Returns:
            Imagen mezclada y etiqueta interpolada
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label
    
    def cutmix_augmentation(self, image1: np.ndarray, image2: np.ndarray,
                           label1: int, label2: int, alpha: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Implementación de CutMix para hologramas
        
        Args:
            image1, image2: Imágenes a mezclar
            label1, label2: Etiquetas correspondientes
            alpha: Parámetro para determinar el tamaño del corte
            
        Returns:
            Imagen con CutMix y etiqueta ajustada
        """
        h, w = image1.shape[:2]
        
        # Generar parámetros del corte
        lam = np.random.beta(alpha, alpha)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Posición aleatoria del corte
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Crear imagen mezclada
        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        # Ajustar etiqueta según el área mezclada
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label
    
    def augment_single_image(self, image: np.ndarray, label: int = 0) -> List[Tuple[np.ndarray, int]]:
        """
        Aplicar múltiples augmentations a una sola imagen
        
        Args:
            image: Imagen original
            label: Etiqueta de la imagen
            
        Returns:
            Lista de imágenes augmentadas con sus etiquetas
        """
        augmented_images = []
        
        # Imagen original
        augmented_images.append((image, label))
        
        # Aplicar augmentations básicas múltiples veces
        for i in range(self.augmentation_factor - 1):
            
            # Seleccionar tipo de augmentation
            aug_type = random.choice(['basic', 'visual', 'hologram', 'combined'])
            
            # Convertir a formato uint8 para albumentations
            img_uint8 = (image * 255).astype(np.uint8)
            
            if aug_type == 'basic':
                augmented = self.basic_augmentations(image=img_uint8)['image']
            elif aug_type == 'visual':
                augmented = self.visual_augmentations(image=img_uint8)['image']
            elif aug_type == 'hologram':
                augmented = self.hologram_augmentations(image=img_uint8)['image']
            else:
                augmented = self.combined_pipeline(image=img_uint8)['image']
            
            # Convertir de vuelta a float32
            augmented = augmented.astype(np.float32) / 255.0
            
            # Aplicar augmentations específicas para hologramas si está habilitado
            if self.preserve_hologram_features:
                if random.random() < 0.3:
                    augmented = self.frequency_domain_augmentation(augmented)
                if random.random() < 0.2:
                    augmented = self.interference_pattern_augmentation(augmented)
            
            augmented_images.append((augmented, label))
        
        return augmented_images
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar data augmentation a todo el dataset
        
        Args:
            X: Array de imágenes
            y: Array de etiquetas
            
        Returns:
            Dataset augmentado
        """
        print(f"🔄 Aplicando data augmentation (factor: {self.augmentation_factor})...")
        print(f"   Dataset original: {X.shape[0]} imágenes")
        
        augmented_X = []
        augmented_y = []
        
        # Procesar cada imagen
        for i, (image, label) in enumerate(zip(X, y)):
            if i % 20 == 0:
                print(f"   Progreso: {i+1}/{len(X)}")
            
            # Aplicar augmentations básicas
            aug_images = self.augment_single_image(image, label)
            
            for aug_img, aug_label in aug_images:
                augmented_X.append(aug_img)
                augmented_y.append(aug_label)
        
        # Aplicar técnicas avanzadas si está habilitado
        if self.apply_advanced_techniques:
            print("   └─ Aplicando técnicas avanzadas (Mixup, CutMix)...")
            advanced_X, advanced_y = self._apply_advanced_augmentations(augmented_X, augmented_y)
            augmented_X.extend(advanced_X)
            augmented_y.extend(advanced_y)
        
        # Convertir a arrays numpy
        X_augmented = np.array(augmented_X)
        y_augmented = np.array(augmented_y)
        
        # Mezclar el dataset
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        print(f"✅ Data augmentation completado!")
        print(f"   Dataset augmentado: {X_augmented.shape[0]} imágenes")
        print(f"   Factor de aumento: {X_augmented.shape[0] / X.shape[0]:.1f}x")
        
        # Guardar estadísticas
        self.augmentation_stats = {
            'original_size': int(X.shape[0]),
            'augmented_size': int(X_augmented.shape[0]),
            'augmentation_factor': self.augmentation_factor,
            'multiplication_achieved': float(X_augmented.shape[0] / X.shape[0]),
            'advanced_techniques_used': self.apply_advanced_techniques,
            'hologram_features_preserved': self.preserve_hologram_features
        }
        
        return X_augmented, y_augmented
    
    def _apply_advanced_augmentations(self, X_list: List[np.ndarray], 
                                    y_list: List[int], 
                                    n_samples: int = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Aplicar técnicas avanzadas como Mixup y CutMix
        
        Args:
            X_list: Lista de imágenes
            y_list: Lista de etiquetas
            n_samples: Número de muestras a generar
            
        Returns:
            Imágenes y etiquetas generadas con técnicas avanzadas
        """
        if n_samples is None:
            n_samples = len(X_list) // 4  # 25% del dataset con técnicas avanzadas
        
        advanced_X = []
        advanced_y = []
        
        for _ in range(n_samples):
            # Seleccionar dos imágenes aleatorias
            idx1, idx2 = random.sample(range(len(X_list)), 2)
            img1, label1 = X_list[idx1], y_list[idx1]
            img2, label2 = X_list[idx2], y_list[idx2]
            
            # Aplicar Mixup o CutMix aleatoriamente
            if random.random() < 0.5:
                # Mixup
                mixed_img, mixed_label = self.mixup_augmentation(img1, img2, label1, label2)
            else:
                # CutMix
                mixed_img, mixed_label = self.cutmix_augmentation(img1, img2, label1, label2)
            
            advanced_X.append(mixed_img)
            advanced_y.append(mixed_label)
        
        return advanced_X, advanced_y
    
    def visualize_augmentations(self, image: np.ndarray, label: int = 0, 
                               save_path: str = "augmentation_examples.png"):
        """
        Visualizar diferentes tipos de augmentation aplicados a una imagen
        
        Args:
            image: Imagen original
            label: Etiqueta de la imagen
            save_path: Ruta donde guardar la visualización
        """
        print("🖼️ Generando visualización de data augmentations...")
        
        # Generar ejemplos de diferentes augmentations
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Aplicar diferentes pipelines
        basic_aug = self.basic_augmentations(image=img_uint8)['image']
        visual_aug = self.visual_augmentations(image=img_uint8)['image']
        hologram_aug = self.hologram_augmentations(image=img_uint8)['image']
        combined_aug = self.combined_pipeline(image=img_uint8)['image']
        
        # Aplicar augmentations específicas
        freq_aug = (self.frequency_domain_augmentation(image) * 255).astype(np.uint8)
        interference_aug = (self.interference_pattern_augmentation(image) * 255).astype(np.uint8)
        
        # Crear visualización
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Ejemplos de Data Augmentation para Hologramas', fontsize=16, fontweight='bold')
        
        images = [
            (img_uint8, "Original"),
            (basic_aug, "Transformaciones\nGeométricas"),
            (visual_aug, "Mejoras\nVisuales"),
            (hologram_aug, "Augmentations\nHolográficas"),
            (combined_aug, "Pipeline\nCombinado"),
            (freq_aug, "Dominio de\nFrecuencia"),
            (interference_aug, "Patrones de\nInterferencia"),
            (img_uint8, "Reservado para\nTécnicas Avanzadas")
        ]
        
        for idx, (img, title) in enumerate(images):
            row = idx // 4
            col = idx % 4
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualización guardada en: {save_path}")
    
    def save_augmentation_config(self, filepath: str = "augmentation_config.json"):
        """
        Guardar configuración de augmentation
        
        Args:
            filepath: Ruta donde guardar la configuración
        """
        import json
        
        config = {
            'augmentation_factor': self.augmentation_factor,
            'preserve_hologram_features': self.preserve_hologram_features,
            'apply_advanced_techniques': self.apply_advanced_techniques,
            'augmentation_stats': self.augmentation_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuración de augmentation guardada en: {filepath}")

def main():
    """
    Función principal para demostrar el data augmentation
    """
    print("🔬 DATA AUGMENTATION PARA HOLOGRAMAS")
    print("=" * 50)
    
    # Cargar dataset preprocesado
    print("📁 Cargando dataset preprocesado...")
    data = np.load('preprocessed_dataset.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"   Dataset original: {X_train.shape[0]} imágenes de entrenamiento")
    
    # Inicializar augmentador
    augmentator = HologramAugmentator(
        augmentation_factor=8,
        preserve_hologram_features=True,
        apply_advanced_techniques=True
    )
    
    # Generar visualización de augmentations
    sample_image = X_train[0]
    sample_label = int(y_train[0])
    augmentator.visualize_augmentations(sample_image, sample_label)
    
    # Aplicar data augmentation al dataset de entrenamiento
    X_train_augmented, y_train_augmented = augmentator.augment_dataset(X_train, y_train)
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS DE DATA AUGMENTATION:")
    print("=" * 50)
    stats = augmentator.augmentation_stats
    print(f"Dataset original: {stats['original_size']} imágenes")
    print(f"Dataset augmentado: {stats['augmented_size']} imágenes")
    print(f"Factor configurado: {stats['augmentation_factor']}x")
    print(f"Factor real alcanzado: {stats['multiplication_achieved']:.1f}x")
    print(f"Técnicas avanzadas: {'✅' if stats['advanced_techniques_used'] else '❌'}")
    print(f"Preservación holográfica: {'✅' if stats['hologram_features_preserved'] else '❌'}")
    
    # Guardar dataset augmentado
    print(f"\n💾 Guardando dataset augmentado...")
    np.savez_compressed(
        'augmented_dataset.npz',
        X_train=X_train_augmented,
        y_train=y_train_augmented,
        X_test=X_test,
        y_test=y_test
    )
    
    # Guardar configuración
    augmentator.save_augmentation_config()
    
    print("✅ Data augmentation completado!")
    print("📁 Archivos generados:")
    print("   • augmentation_examples.png")
    print("   • augmented_dataset.npz")
    print("   • augmentation_config.json")

if __name__ == "__main__":
    main() 