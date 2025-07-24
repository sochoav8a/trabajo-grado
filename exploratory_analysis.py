#!/usr/bin/env python3
"""
Análisis Exploratorio del Dataset de Hologramas de Glóbulos Sanguíneos
Proyecto: Clasificación de células sanas vs SCD
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skimage import feature, filters, measure
import warnings
warnings.filterwarnings('ignore')

# Configuración para gráficos en español
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (15, 10)

class HologramDatasetAnalyzer:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = Path(dataset_path)
        self.classes = ["SCD", "Healthy"]
        self.class_paths = {
            "SCD": self.dataset_path / "SCD",
            "Healthy": self.dataset_path / "Healthy"
        }
        self.results = {}
        
    def load_image_info(self):
        """Cargar información básica de todas las imágenes"""
        print("Cargando información del dataset...")
        
        image_info = []
        for class_name in self.classes:
            class_path = self.class_paths[class_name]
            image_files = list(class_path.glob("*.png"))
            
            print(f"   -> {class_name}: {len(image_files)} imágenes")
            
            for img_path in image_files:
                # Información básica del archivo
                file_size = img_path.stat().st_size / (1024*1024)  # MB
                
                # Cargar imagen para obtener dimensiones
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width, channels = img.shape
                    
                    image_info.append({
                        'clase': class_name,
                        'archivo': img_path.name,
                        'ruta': str(img_path),
                        'alto': height,
                        'ancho': width,
                        'canales': channels,
                        'tamaño_mb': file_size
                    })
        
        self.image_df = pd.DataFrame(image_info)
        return self.image_df
    
    def basic_statistics(self):
        """Estadísticas básicas del dataset"""
        print("\nESTADÍSTICAS BÁSICAS DEL DATASET")
        print("=" * 50)
        
        # Resumen por clase
        summary = self.image_df.groupby('clase').agg({
            'archivo': 'count',
            'tamaño_mb': ['mean', 'std', 'min', 'max'],
            'alto': ['mean', 'std'],
            'ancho': ['mean', 'std']
        }).round(2)
        
        print("Distribución por clase:")
        for clase in self.classes:
            count = len(self.image_df[self.image_df['clase'] == clase])
            percentage = (count / len(self.image_df)) * 100
            print(f"   {clase}: {count} imágenes ({percentage:.1f}%)")
        
        print(f"\nTotal de imágenes: {len(self.image_df)}")
        print(f"Resolución promedio: {self.image_df['alto'].mean():.0f} x {self.image_df['ancho'].mean():.0f}")
        print(f"Tamaño promedio por imagen: {self.image_df['tamaño_mb'].mean():.1f} MB")
        print(f"Espacio total del dataset: {self.image_df['tamaño_mb'].sum():.1f} MB")
        
        return summary
    
    def visualize_samples(self, samples_per_class=4):
        """Visualizar muestras de cada clase"""
        print(f"\nVisualizando {samples_per_class} muestras por clase...")
        
        fig, axes = plt.subplots(2, samples_per_class, figsize=(20, 10))
        fig.suptitle('Muestras del Dataset de Hologramas', fontsize=16, fontweight='bold')
        
        for class_idx, class_name in enumerate(self.classes):
            class_images = self.image_df[self.image_df['clase'] == class_name]
            sample_images = class_images.sample(min(samples_per_class, len(class_images)))
            
            for img_idx, (_, img_info) in enumerate(sample_images.iterrows()):
                if img_idx >= samples_per_class:
                    break
                    
                # Cargar y redimensionar imagen para visualización
                img = cv2.imread(img_info['ruta'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Redimensionar para visualización
                img_resized = cv2.resize(img_rgb, (400, 300))
                
                ax = axes[class_idx, img_idx]
                ax.imshow(img_resized)
                ax.set_title(f'{class_name}\n{img_info["archivo"][:15]}...', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_pixel_distributions(self, sample_size=5):
        """Analizar distribuciones de píxeles"""
        print(f"\nAnalizando distribuciones de píxeles (muestra de {sample_size} por clase)...")
        
        pixel_data = {class_name: [] for class_name in self.classes}
        
        for class_name in self.classes:
            class_images = self.image_df[self.image_df['clase'] == class_name]
            sample_images = class_images.sample(min(sample_size, len(class_images)))
            
            for _, img_info in sample_images.iterrows():
                img = cv2.imread(img_info['ruta'])
                # Convertir a escala de grises para análisis más simple
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Redimensionar para acelerar análisis
                gray_resized = cv2.resize(gray, (500, 375))
                pixel_data[class_name].extend(gray_resized.flatten())
        
        # Crear histogramas
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis de Distribución de Píxeles', fontsize=16, fontweight='bold')
        
        # Histogramas por clase
        for idx, class_name in enumerate(self.classes):
            ax = axes[0, idx]
            ax.hist(pixel_data[class_name], bins=50, alpha=0.7, 
                   color='red' if class_name == 'SCD' else 'blue', density=True)
            ax.set_title(f'Histograma - {class_name}')
            ax.set_xlabel('Intensidad de Píxel')
            ax.set_ylabel('Densidad')
            ax.grid(True, alpha=0.3)
        
        # Comparación directa
        ax = axes[1, 0]
        for class_name in self.classes:
            color = 'red' if class_name == 'SCD' else 'blue'
            ax.hist(pixel_data[class_name], bins=50, alpha=0.6, 
                   label=class_name, color=color, density=True)
        ax.set_title('Comparación de Distribuciones')
        ax.set_xlabel('Intensidad de Píxel')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Estadísticas
        ax = axes[1, 1]
        stats_data = []
        for class_name in self.classes:
            data = np.array(pixel_data[class_name])
            stats_data.append([
                class_name,
                np.mean(data),
                np.std(data),
                np.median(data),
                np.percentile(data, 25),
                np.percentile(data, 75)
            ])
        
        stats_df = pd.DataFrame(stats_data, columns=['Clase', 'Media', 'Std', 'Mediana', 'Q25', 'Q75'])
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=stats_df.round(2).values,
                        colLabels=stats_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Estadísticas por Clase')
        
        plt.tight_layout()
        plt.savefig('pixel_distribution_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return pixel_data, stats_df
    
    def analyze_textures(self, sample_size=3):
        """Análisis de texturas usando Local Binary Patterns"""
        print(f"\nAnalizando texturas (muestra de {sample_size} por clase)...")
        
        texture_features = {class_name: [] for class_name in self.classes}
        
        fig, axes = plt.subplots(2, sample_size, figsize=(15, 8))
        fig.suptitle('Análisis de Texturas - Local Binary Patterns', fontsize=16, fontweight='bold')
        
        for class_idx, class_name in enumerate(self.classes):
            class_images = self.image_df[self.image_df['clase'] == class_name]
            sample_images = class_images.sample(min(sample_size, len(class_images)))
            
            for img_idx, (_, img_info) in enumerate(sample_images.iterrows()):
                if img_idx >= sample_size:
                    break
                
                # Cargar imagen
                img = cv2.imread(img_info['ruta'])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Redimensionar para análisis
                gray_resized = cv2.resize(gray, (500, 375))
                
                # Calcular Local Binary Pattern
                lbp = feature.local_binary_pattern(gray_resized, P=8, R=1, method='uniform')
                
                # Visualizar
                ax = axes[class_idx, img_idx]
                ax.imshow(lbp, cmap='gray')
                ax.set_title(f'{class_name} - LBP\n{img_info["archivo"][:10]}...')
                ax.axis('off')
                
                # Guardar características de textura
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
                texture_features[class_name].append(lbp_hist)
        
        plt.tight_layout()
        plt.savefig('texture_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return texture_features
    
    def hologram_specific_analysis(self, sample_size=2):
        """Análisis específico para hologramas: patrones de interferencia"""
        print(f"\nAnálisis específico de hologramas (muestra de {sample_size} por clase)...")
        
        fig, axes = plt.subplots(2, sample_size * 3, figsize=(18, 8))
        fig.suptitle('Análisis Específico de Hologramas', fontsize=16, fontweight='bold')
        
        for class_idx, class_name in enumerate(self.classes):
            class_images = self.image_df[self.image_df['clase'] == class_name]
            sample_images = class_images.sample(min(sample_size, len(class_images)))
            
            for img_idx, (_, img_info) in enumerate(sample_images.iterrows()):
                if img_idx >= sample_size:
                    break
                
                # Cargar imagen
                img = cv2.imread(img_info['ruta'])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, (400, 300))
                
                base_col = img_idx * 3
                
                # Imagen original
                axes[class_idx, base_col].imshow(gray_resized, cmap='gray')
                axes[class_idx, base_col].set_title(f'{class_name} - Original')
                axes[class_idx, base_col].axis('off')
                
                # Filtro de frecuencias (FFT)
                fft = np.fft.fft2(gray_resized)
                fft_shift = np.fft.fftshift(fft)
                magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
                
                axes[class_idx, base_col + 1].imshow(magnitude_spectrum, cmap='hot')
                axes[class_idx, base_col + 1].set_title(f'{class_name} - FFT')
                axes[class_idx, base_col + 1].axis('off')
                
                # Detección de bordes (Canny)
                edges = cv2.Canny(gray_resized, 50, 150)
                
                axes[class_idx, base_col + 2].imshow(edges, cmap='gray')
                axes[class_idx, base_col + 2].set_title(f'{class_name} - Bordes')
                axes[class_idx, base_col + 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('hologram_specific_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generar reporte completo del análisis"""
        print("\nREPORTE COMPLETO DEL ANÁLISIS")
        print("=" * 60)
        
        total_images = len(self.image_df)
        scd_count = len(self.image_df[self.image_df['clase'] == 'SCD'])
        healthy_count = len(self.image_df[self.image_df['clase'] == 'Healthy'])
        
        report = f"""
RESUMEN EJECUTIVO DEL DATASET
============================================================

COMPOSICIÓN DEL DATASET:
   - Total de imágenes: {total_images}
   - Células SCD (enfermas): {scd_count} ({scd_count/total_images*100:.1f}%)
   - Células Healthy (sanas): {healthy_count} ({healthy_count/total_images*100:.1f}%)
   - Balance del dataset: {'Balanceado' if abs(scd_count - healthy_count) < 20 else 'Desbalanceado'}

CARACTERÍSTICAS TÉCNICAS:
   - Resolución: {self.image_df['alto'].iloc[0]} x {self.image_df['ancho'].iloc[0]} píxeles
   - Formato: PNG a color (3 canales)
   - Tamaño promedio: {self.image_df['tamaño_mb'].mean():.1f} MB por imagen
   - Espacio total: {self.image_df['tamaño_mb'].sum():.1f} MB

DESAFÍOS IDENTIFICADOS:
   - Dataset pequeño ({total_images} imágenes) -> Alto riesgo de overfitting
   - Alta resolución -> Necesidad de optimización computacional
   - Imágenes holográficas -> Requiere preprocesamiento especializado

RECOMENDACIONES PARA EL CLASIFICADOR:
   1. Data Augmentation agresiva (rotaciones, flips, variaciones de contraste)
   2. Transfer Learning con modelos pre-entrenados
   3. Validación cruzada estratificada
   4. Regularización fuerte (dropout, batch normalization)
   5. Early stopping para prevenir overfitting
   6. Ensemble de modelos para mayor robustez

PRÓXIMOS PASOS:
   1. Implementar pipeline de preprocesamiento
   2. Desarrollar estrategias de data augmentation
   3. Extraer características específicas de hologramas
   4. Diseñar arquitectura CNN optimizada
        """
        
        print(report)
        
        # Guardar reporte
        with open('dataset_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

def main():
    """Función principal para ejecutar todo el análisis"""
    print("ANÁLISIS EXPLORATORIO DEL DATASET DE HOLOGRAMAS")
    print("=" * 60)
    
    # Inicializar analizador
    analyzer = HologramDatasetAnalyzer()
    
    # Ejecutar análisis completo
    analyzer.load_image_info()
    analyzer.basic_statistics()
    analyzer.visualize_samples()
    analyzer.analyze_pixel_distributions()
    analyzer.analyze_textures()
    analyzer.hologram_specific_analysis()
    analyzer.generate_report()
    
    print("\nAnálisis completo terminado!")
    print("Archivos generados:")
    print("   - dataset_samples.png")
    print("   - pixel_distribution_analysis.png")
    print("   - texture_analysis.png")
    print("   - hologram_specific_analysis.png")
    print("   - dataset_analysis_report.txt")

if __name__ == "__main__":
    main() 