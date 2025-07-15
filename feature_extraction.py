#!/usr/bin/env python3
"""
Extracción de Características Específicas para Hologramas de Glóbulos Sanguíneos
Proyecto: Clasificación de células sanas vs SCD
"""

import numpy as np
import cv2
from skimage import feature, filters, measure, morphology
from scipy import ndimage, stats, signal
from scipy.fftpack import fft2, fftshift
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HologramFeatureExtractor:
    """
    Clase para extraer características específicas de hologramas
    """
    
    def __init__(self, feature_config: Optional[Dict] = None):
        """
        Inicializar el extractor de características
        
        Args:
            feature_config: Configuración de características a extraer
        """
        self.feature_config = feature_config or self._default_feature_config()
        self.feature_names = []
        
    def _default_feature_config(self) -> Dict:
        """Configuración por defecto de características a extraer"""
        return {
            'texture_features': True,
            'morphological_features': True,
            'frequency_features': True,
            'hologram_specific_features': True,
            'statistical_features': True,
            'lbp_params': {'P': 8, 'R': 1, 'method': 'uniform'},
            'haralick_distances': [1, 2, 3],
            'haralick_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4]
        }
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extraer todas las características de una imagen
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Vector de características
        """
        features = []
        self.feature_names = []
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Extraer diferentes tipos de características
        if self.feature_config['texture_features']:
            texture_feats = self._extract_texture_features(gray)
            features.extend(texture_feats)
            
        if self.feature_config['morphological_features']:
            morph_feats = self._extract_morphological_features(gray)
            features.extend(morph_feats)
            
        if self.feature_config['frequency_features']:
            freq_feats = self._extract_frequency_features(gray)
            features.extend(freq_feats)
            
        if self.feature_config['hologram_specific_features']:
            holo_feats = self._extract_hologram_specific_features(gray)
            features.extend(holo_feats)
            
        if self.feature_config['statistical_features']:
            stat_feats = self._extract_statistical_features(gray)
            features.extend(stat_feats)
        
        return np.array(features)
    
    def _extract_texture_features(self, gray: np.ndarray) -> List[float]:
        """
        Extraer características de textura
        
        Args:
            gray: Imagen en escala de grises
            
        Returns:
            Lista de características de textura
        """
        features = []
        
        # 1. Local Binary Patterns
        lbp = feature.local_binary_pattern(
            gray, 
            self.feature_config['lbp_params']['P'], 
            self.feature_config['lbp_params']['R'], 
            method=self.feature_config['lbp_params']['method']
        )
        
        # Histograma LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        # Estadísticas del histograma LBP
        features.extend(hist[:10])  # Primeros 10 bins
        self.feature_names.extend([f'lbp_hist_bin_{i}' for i in range(10)])
        
        features.append(np.mean(hist))
        features.append(np.std(hist))
        features.append(stats.entropy(hist))
        self.feature_names.extend(['lbp_hist_mean', 'lbp_hist_std', 'lbp_hist_entropy'])
        
        # 2. Características de Haralick
        # Normalizar imagen para GLCM
        gray_norm = (gray / gray.max() * 255).astype(np.uint8)
        gray_quant = (gray_norm / 4).astype(np.uint8)  # Cuantizar a 64 niveles
        
        # Calcular GLCM
        glcm = feature.graycomatrix(
            gray_quant, 
            distances=self.feature_config['haralick_distances'],
            angles=self.feature_config['haralick_angles'],
            levels=64,
            symmetric=True,
            normed=True
        )
        
        # Propiedades de Haralick
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in properties:
            prop_vals = feature.graycoprops(glcm, prop).ravel()
            features.append(np.mean(prop_vals))
            features.append(np.std(prop_vals))
            self.feature_names.extend([f'haralick_{prop}_mean', f'haralick_{prop}_std'])
        
        # 3. Características de Gabor
        gabor_features = self._extract_gabor_features(gray)
        features.extend(gabor_features)
        
        return features
    
    def _extract_gabor_features(self, gray: np.ndarray) -> List[float]:
        """
        Extraer características usando filtros de Gabor
        
        Args:
            gray: Imagen en escala de grises
            
        Returns:
            Lista de características de Gabor
        """
        features = []
        
        # Parámetros de Gabor
        frequencies = [0.05, 0.1, 0.2]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for freq in frequencies:
            for theta in orientations:
                # Crear kernel de Gabor
                kernel = cv2.getGaborKernel(
                    (21, 21), 
                    sigma=5.0, 
                    theta=theta,
                    lambd=1/freq, 
                    gamma=0.5, 
                    psi=0
                )
                
                # Aplicar filtro
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                
                # Extraer estadísticas
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
                
                self.feature_names.extend([
                    f'gabor_freq{freq:.2f}_theta{theta:.2f}_mean',
                    f'gabor_freq{freq:.2f}_theta{theta:.2f}_std'
                ])
        
        return features
    
    def _extract_morphological_features(self, gray: np.ndarray) -> List[float]:
        """
        Extraer características morfológicas
        
        Args:
            gray: Imagen en escala de grises
            
        Returns:
            Lista de características morfológicas
        """
        features = []
        
        # Binarizar imagen usando Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Características del contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Área
            area = cv2.contourArea(largest_contour)
            features.append(area / (gray.shape[0] * gray.shape[1]))  # Área relativa
            self.feature_names.append('morph_relative_area')
            
            # Perímetro
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter)
            self.feature_names.append('morph_perimeter')
            
            # Circularidad
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                features.append(circularity)
            else:
                features.append(0)
            self.feature_names.append('morph_circularity')
            
            # Momentos de Hu
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(np.log(np.abs(hu_moments) + 1e-10)[:4])  # Primeros 4 momentos
            self.feature_names.extend([f'hu_moment_{i}' for i in range(4)])
            
            # Solidez
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                features.append(solidity)
            else:
                features.append(0)
            self.feature_names.append('morph_solidity')
            
        else:
            # Si no hay contornos, características por defecto
            features.extend([0] * 8)
            self.feature_names.extend([
                'morph_relative_area', 'morph_perimeter', 'morph_circularity',
                'hu_moment_0', 'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
                'morph_solidity'
            ])
        
        # Esqueleto morfológico
        skeleton = morphology.skeletonize(binary > 0)
        features.append(np.sum(skeleton) / binary.size)
        self.feature_names.append('morph_skeleton_ratio')
        
        return features
    
    def _extract_frequency_features(self, gray: np.ndarray) -> List[float]:
        """
        Extraer características del dominio de frecuencia
        
        Args:
            gray: Imagen en escala de grises
            
        Returns:
            Lista de características de frecuencia
        """
        features = []
        
        # FFT 2D
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Espectro de potencia
        power_spectrum = magnitude_spectrum ** 2
        
        # Dividir en anillos concéntricos
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        
        # Distancia desde el centro
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Características por anillo
        max_radius = min(center_x, center_y)
        n_rings = 5
        ring_width = max_radius / n_rings
        
        for i in range(n_rings):
            inner_radius = i * ring_width
            outer_radius = (i + 1) * ring_width
            
            # Máscara del anillo
            ring_mask = (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
            
            # Energía en el anillo
            ring_energy = np.sum(power_spectrum[ring_mask])
            total_energy = np.sum(power_spectrum)
            
            if total_energy > 0:
                features.append(ring_energy / total_energy)
            else:
                features.append(0)
            
            self.feature_names.append(f'freq_ring_{i}_energy_ratio')
        
        # Características adicionales del espectro
        # Energía en frecuencias bajas vs altas
        low_freq_mask = dist_from_center < (max_radius * 0.3)
        high_freq_mask = dist_from_center > (max_radius * 0.7)
        
        low_freq_energy = np.sum(power_spectrum[low_freq_mask])
        high_freq_energy = np.sum(power_spectrum[high_freq_mask])
        
        if low_freq_energy + high_freq_energy > 0:
            features.append(low_freq_energy / (low_freq_energy + high_freq_energy))
        else:
            features.append(0)
        self.feature_names.append('freq_low_high_ratio')
        
        # Entropía espectral
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
        features.append(spectral_entropy)
        self.feature_names.append('freq_spectral_entropy')
        
        return features
    
    def _extract_hologram_specific_features(self, gray: np.ndarray) -> List[float]:
        """
        Extraer características específicas de hologramas
        
        Args:
            gray: Imagen en escala de grises
            
        Returns:
            Lista de características holográficas
        """
        features = []
        
        # 1. Análisis de patrones de interferencia
        # Detectar franjas usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitud y dirección del gradiente
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Histograma de direcciones
        hist_dirs, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
        hist_dirs = hist_dirs / (np.sum(hist_dirs) + 1e-10)
        
        features.extend(hist_dirs[:4])  # Primeras 4 direcciones principales
        self.feature_names.extend([f'holo_gradient_dir_{i}' for i in range(4)])
        
        # Uniformidad direccional
        dir_entropy = -np.sum(hist_dirs * np.log(hist_dirs + 1e-10))
        features.append(dir_entropy)
        self.feature_names.append('holo_directional_entropy')
        
        # 2. Análisis de fase
        # Transformada de Hilbert para estimar fase
        hilbert_x = signal.hilbert(gray.astype(np.float64), axis=1)
        phase_x = np.angle(hilbert_x)
        
        # Estadísticas de fase
        features.append(np.mean(phase_x))
        features.append(np.std(phase_x))
        features.append(stats.skew(phase_x.ravel()))
        self.feature_names.extend(['holo_phase_mean', 'holo_phase_std', 'holo_phase_skew'])
        
        # 3. Contraste de franjas
        # Calcular contraste local
        kernel_size = 5
        gray_float = gray.astype(np.float64)
        local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        local_std = np.sqrt(cv2.blur(gray_float**2, (kernel_size, kernel_size)) - local_mean**2)
        
        contrast = local_std / (local_mean + 1e-10)
        features.append(np.mean(contrast))
        features.append(np.std(contrast))
        self.feature_names.extend(['holo_local_contrast_mean', 'holo_local_contrast_std'])
        
        # 4. Análisis de modulación
        # Perfil de intensidad en dirección horizontal y vertical
        h_profile = np.mean(gray, axis=0)
        v_profile = np.mean(gray, axis=1)
        
        # FFT de los perfiles para detectar frecuencia de modulación
        h_fft = np.abs(np.fft.fft(h_profile))[:len(h_profile)//2]
        v_fft = np.abs(np.fft.fft(v_profile))[:len(v_profile)//2]
        
        # Frecuencia dominante
        h_dominant_freq = np.argmax(h_fft[1:]) + 1  # Ignorar DC
        v_dominant_freq = np.argmax(v_fft[1:]) + 1
        
        features.append(h_dominant_freq / len(h_profile))
        features.append(v_dominant_freq / len(v_profile))
        self.feature_names.extend(['holo_h_dominant_freq', 'holo_v_dominant_freq'])
        
        return features
    
    def _extract_statistical_features(self, gray: np.ndarray) -> List[float]:
        """
        Extraer características estadísticas básicas
        
        Args:
            gray: Imagen en escala de grises
            
        Returns:
            Lista de características estadísticas
        """
        features = []
        
        # Estadísticas de primer orden
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(stats.skew(gray.ravel()))
        features.append(stats.kurtosis(gray.ravel()))
        
        self.feature_names.extend(['stat_mean', 'stat_std', 'stat_skew', 'stat_kurtosis'])
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            features.append(np.percentile(gray, p))
            self.feature_names.append(f'stat_percentile_{p}')
        
        # Entropía
        hist, _ = np.histogram(gray, bins=32, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        features.append(entropy)
        self.feature_names.append('stat_entropy')
        
        # Rango dinámico
        features.append(np.ptp(gray))  # peak-to-peak
        self.feature_names.append('stat_range')
        
        return features
    
    def extract_features_from_dataset(self, X: np.ndarray, save_features: bool = True) -> np.ndarray:
        """
        Extraer características de todo un dataset
        
        Args:
            X: Array de imágenes
            save_features: Si guardar las características extraídas
            
        Returns:
            Matriz de características
        """
        print(f"Extrayendo características de {X.shape[0]} imágenes...")
        
        features_list = []
        
        for i, image in enumerate(X):
            if i % 50 == 0:
                print(f"   Progreso: {i+1}/{X.shape[0]}")
            
            features = self.extract_features(image)
            features_list.append(features)
        
        features_matrix = np.array(features_list)
        
        print(f"Características extraídas: {features_matrix.shape}")
        print(f"Total de características por imagen: {features_matrix.shape[1]}")
        
        if save_features:
            # Guardar nombres de características
            with open('feature_names.txt', 'w') as f:
                for name in self.feature_names:
                    f.write(f"{name}\n")
            
            print("Nombres de características guardados en: feature_names.txt")
        
        return features_matrix
    
    def create_feature_dataframe(self, features_matrix: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """
        Crear DataFrame con características y etiquetas
        
        Args:
            features_matrix: Matriz de características
            labels: Etiquetas correspondientes
            
        Returns:
            DataFrame con características
        """
        df = pd.DataFrame(features_matrix, columns=self.feature_names)
        df['label'] = labels
        
        return df
    
    def analyze_feature_importance(self, features_df: pd.DataFrame, save_plot: bool = True):
        """
        Analizar importancia de características usando correlación con la etiqueta
        
        Args:
            features_df: DataFrame con características y etiquetas
            save_plot: Si guardar el gráfico
        """
        import matplotlib.pyplot as plt
        
        # Calcular correlación con la etiqueta
        correlations = features_df.corr()['label'].drop('label').abs().sort_values(ascending=False)
        
        # Top 20 características más importantes
        top_features = correlations.head(20)
        
        if save_plot:
            plt.figure(figsize=(10, 8))
            top_features.plot(kind='barh')
            plt.xlabel('Correlación Absoluta con Etiqueta')
            plt.title('Top 20 Características Más Importantes')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Análisis de importancia guardado en: feature_importance.png")
        
        return correlations


def main():
    """
    Función principal para demostrar la extracción de características
    """
    print("EXTRACCIÓN DE CARACTERÍSTICAS PARA HOLOGRAMAS")
    print("=" * 50)
    
    # Cargar dataset augmentado
    print("Cargando dataset augmentado...")
    data = np.load('augmented_dataset.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"   Dataset de entrenamiento: {X_train.shape[0]} imágenes")
    print(f"   Dataset de prueba: {X_test.shape[0]} imágenes")
    
    # Inicializar extractor
    extractor = HologramFeatureExtractor()
    
    # Extraer características del conjunto de entrenamiento
    print("\nExtrayendo características del conjunto de entrenamiento...")
    X_train_features = extractor.extract_features_from_dataset(X_train)
    
    # Extraer características del conjunto de prueba
    print("\nExtrayendo características del conjunto de prueba...")
    X_test_features = extractor.extract_features_from_dataset(X_test, save_features=False)
    
    # Crear DataFrames
    train_df = extractor.create_feature_dataframe(X_train_features, y_train)
    test_df = extractor.create_feature_dataframe(X_test_features, y_test)
    
    # Analizar importancia de características
    print("\nAnalizando importancia de características...")
    correlations = extractor.analyze_feature_importance(train_df)
    
    # Mostrar características más importantes
    print("\nTop 10 características más correlacionadas con la etiqueta:")
    print(correlations.head(10))
    
    # Guardar características extraídas
    print("\nGuardando características extraídas...")
    np.savez_compressed(
        'extracted_features.npz',
        X_train_features=X_train_features,
        X_test_features=X_test_features,
        y_train=y_train,
        y_test=y_test,
        feature_names=extractor.feature_names
    )
    
    # Guardar DataFrames
    train_df.to_csv('train_features.csv', index=False)
    test_df.to_csv('test_features.csv', index=False)
    
    print("\nExtracción de características completada!")
    print("Archivos generados:")
    print("   • extracted_features.npz")
    print("   • feature_names.txt")
    print("   • train_features.csv")
    print("   • test_features.csv")
    print("   • feature_importance.png")
    
    # Resumen de características
    print(f"\nResumen:")
    print(f"   Total de características extraídas: {len(extractor.feature_names)}")
    print(f"   Dimensiones finales entrenamiento: {X_train_features.shape}")
    print(f"   Dimensiones finales prueba: {X_test_features.shape}")


if __name__ == "__main__":
    main() 