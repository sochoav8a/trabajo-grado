# Clasificación de hologramas de glóbulos sanguíneos
_Exported on 7/15/2025 at 00:06:36 GMT-5 from Cursor (1.2.4)_

---

**User**

Quiero que analices la carpeta dataset, ahi hay hologramas de globulos sanguineos, mi objetivo es realizar un clasificador para identificar sanos y enfermos, como podrás ver una de las limitaciones es que tenemos que entrenar con apenas 200 imagenes y obtener buenos resultados sin overfitting, analiza el dataset y hagamos un plan de trabajo

---

**Cursor**

¡Perfecto! Voy a analizar el dataset de hologramas de glóbulos sanguíneos para ayudarte a crear un clasificador efectivo con pocas imágenes. Empezaré explorando la estructura del dataset.

Excelente, veo que tienes un dataset organizado en dos clases: SCD (enfermedad de células falciformes) y Healthy (células sanas). Vamos a explorar cada carpeta para entender mejor la distribución de datos.

Ahora veo la estructura del dataset:

**Distribución de datos:**
- **SCD (células enfermas):** ~80 imágenes
- **Healthy (células sanas):** ~96 imágenes
- **Total:** ~176 imágenes

**Observaciones iniciales:**
- Imágenes de alta resolución (5-7MB cada una)
- Dataset relativamente balanceado
- Nombres con timestamps sugieren capturas secuenciales

Ahora voy a examinar algunas imágenes para entender mejor las características visuales:

Ahora tengo una comprensión clara del dataset. Vamos a crear un plan de trabajo para desarrollar un clasificador efectivo:

## 📊 Análisis del Dataset de Hologramas

### **Características del Dataset:**
- **Total de imágenes:** ~176 (80 SCD + 96 Healthy)
- **Resolución:** 3000×4000 píxeles (muy alta resolución)
- **Formato:** PNG a color (3 canales)
- **Tamaño por imagen:** 5-7MB
- **Balance:** Relativamente balanceado (52% healthy, 48% SCD)

### **Desafíos Identificados:**
1. **Dataset pequeño** → Alto riesgo de overfitting
2. **Alta resolución** → Necesidad de optimización computacional
3. **Hologramas** → Requiere preprocesamiento especializado
4. **Variabilidad limitada** → Necesidad de augmentation agresiva

## 🚀 Plan de Trabajo Estratégico

He creado un plan de 10 pasos para desarrollar un clasificador robusto:

### **Fase 1: Análisis y Preprocesamiento**
- Análisis exploratorio detallado
- Pipeline de preprocesamiento optimizado para hologramas
- Estrategias de data augmentation específicas

### **Fase 2: Desarrollo del Modelo**
- Extracción de características específicas de hologramas
- Arquitectura CNN ligera con regularización fuerte
- Transfer learning con modelos pre-entrenados

### **Fase 3: Validación y Optimización**
- Validación cruzada estratificada
- Técnicas anti-overfitting avanzadas
- Ensemble de modelos para mayor robustez

¿Te gustaría que comience implementando el análisis exploratorio detallado? Esto nos dará insights importantes sobre las diferencias visuales entre células sanas y enfermas, y nos ayudará a definir las mejores estrategias de preprocesamiento y augmentation.

---

**User**

me parece

---

**Cursor**

¡Perfecto! Vamos a comenzar con el análisis exploratorio detallado. Marcaré la primera tarea como en progreso y empezaré la implementación.

```python
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
        print("📊 Cargando información del dataset...")
        
        image_info = []
        for class_name in self.classes:
            class_path = self.class_paths[class_name]
            image_files = list(class_path.glob("*.png"))
            
            print(f"   └─ {class_name}: {len(image_files)} imágenes")
            
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
        print("\n📈 ESTADÍSTICAS BÁSICAS DEL DATASET")
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
        print(f"\n🖼️  Visualizando {samples_per_class} muestras por clase...")
        
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
        print(f"\n📊 Analizando distribuciones de píxeles (muestra de {sample_size} por clase)...")
        
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
        print(f"\n🔍 Analizando texturas (muestra de {sample_size} por clase)...")
        
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
        print(f"\n🌊 Análisis específico de hologramas (muestra de {sample_size} por clase)...")
        
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
        print("\n📋 REPORTE COMPLETO DEL ANÁLISIS")
        print("=" * 60)
        
        total_images = len(self.image_df)
        scd_count = len(self.image_df[self.image_df['clase'] == 'SCD'])
        healthy_count = len(self.image_df[self.image_df['clase'] == 'Healthy'])
        
        report = f"""
RESUMEN EJECUTIVO DEL DATASET
════════════════════════════════════════════════════════════

📊 COMPOSICIÓN DEL DATASET:
   • Total de imágenes: {total_images}
   • Células SCD (enfermas): {scd_count} ({scd_count/total_images*100:.1f}%)
   • Células Healthy (sanas): {healthy_count} ({healthy_count/total_images*100:.1f}%)
   • Balance del dataset: {'✅ Balanceado' if abs(scd_count - healthy_count) < 20 else '⚠️ Desbalanceado'}

🖼️ CARACTERÍSTICAS TÉCNICAS:
   • Resolución: {self.image_df['alto'].iloc[0]} x {self.image_df['ancho'].iloc[0]} píxeles
   • Formato: PNG a color (3 canales)
   • Tamaño promedio: {self.image_df['tamaño_mb'].mean():.1f} MB por imagen
   • Espacio total: {self.image_df['tamaño_mb'].sum():.1f} MB

🔍 DESAFÍOS IDENTIFICADOS:
   • Dataset pequeño ({total_images} imágenes) → Alto riesgo de overfitting
   • Alta resolución → Necesidad de optimización computacional
   • Imágenes holográficas → Requiere preprocesamiento especializado

💡 RECOMENDACIONES PARA EL CLASIFICADOR:
   1. Data Augmentation agresiva (rotaciones, flips, variaciones de contraste)
   2. Transfer Learning con modelos pre-entrenados
   3. Validación cruzada estratificada
   4. Regularización fuerte (dropout, batch normalization)
   5. Early stopping para prevenir overfitting
   6. Ensemble de modelos para mayor robustez

🚀 PRÓXIMOS PASOS:
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
    print("🔬 ANÁLISIS EXPLORATORIO DEL DATASET DE HOLOGRAMAS")
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
    
    print("\n✅ Análisis completo terminado!")
    print("📁 Archivos generados:")
    print("   • dataset_samples.png")
    print("   • pixel_distribution_analysis.png")
    print("   • texture_analysis.png")
    print("   • hologram_specific_analysis.png")
    print("   • dataset_analysis_report.txt")

if __name__ == "__main__":
    main()
```

Veo que necesitamos instalar las dependencias. Voy a crear un archivo `requirements.txt` y luego instalar las librerías necesarias:

```pip-requirements
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
opencv-python>=4.5.0
Pillow>=8.3.0
tensorflow>=2.8.0
torch>=1.11.0
torchvision>=0.12.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

¡Excelente! Las dependencias se instalaron correctamente. Ahora ejecutemos el análisis exploratorio:

¡Excelente! El análisis exploratorio se completó exitosamente. Veamos los resultados detallados:

## 🎯 **ANÁLISIS EXPLORATORIO COMPLETADO**

¡Excelente! He terminado el análisis exploratorio detallado del dataset. Aquí están los **hallazgos clave**:

### 📊 **Resultados del Análisis:**

**Composición del Dataset:**
- **Total:** 196 imágenes (más de las estimadas inicialmente)
- **SCD (enfermas):** 83 imágenes (42.3%)
- **Healthy (sanas):** 113 imágenes (57.7%)
- **Balance:** Ligeramente desbalanceado pero manejable

**Características Técnicas:**
- **Resolución:** 3000×4000 píxeles (¡12 megapíxeles!)
- **Tamaño promedio:** 6.4 MB por imagen
- **Espacio total:** 1.25 GB
- **Calidad:** Imágenes holográficas de muy alta resolución

### 🔍 **Archivos Generados:**
✅ `dataset_samples.png` - Muestras visuales de ambas clases
✅ `pixel_distribution_analysis.png` - Análisis estadístico de píxeles
✅ `texture_analysis.png` - Patrones de textura (LBP)
✅ `hologram_specific_analysis.png` - Análisis FFT y detección de bordes
✅ `dataset_analysis_report.txt` - Reporte ejecutivo completo

### 🚨 **Desafíos Confirmados:**
1. **Dataset pequeño** (196 imágenes) → Riesgo alto de overfitting
2. **Alta resolución** → Necesidad de optimización computacional
3. **Ligero desbalance** → Requerirá estratificación cuidadosa
4. **Naturaleza holográfica** → Preprocesamiento especializado

### 🚀 **Próximo Paso Recomendado:**
Basándome en estos resultados, sugiero proceder con el **pipeline de preprocesamiento** que será crucial para manejar la alta resolución y optimizar las imágenes holográficas.

¿Te gustaría que implemente el pipeline de preprocesamiento ahora? Esto incluirá:
- Redimensionado inteligente
- Normalización específica para hologramas
- Filtros de realce de contraste
- Preparación para data augmentation

---

**User**

puedes continuar

---

**Cursor**

¡Perfecto! Continuemos con el pipeline de preprocesamiento. Marcaré la siguiente tarea como en progreso y la implementaré.

```python
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
        # 1. Redimensionado inteligente
        img_resized = self._smart_resize(img)
        
        # 2. Mejoras específicas para hologramas
        if apply_hologram_enhancement:
            img_enhanced = self._hologram_enhancement(img_resized)
        else:
            img_enhanced = img_resized
            
        # 3. Normalización
        img_normalized = self._normalize_image(img_enhanced)
        
        # 4. Mejora de contraste adaptativo
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
        
        # Calcular ratio de aspecto
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
        
        # Rellenar para llegar al tamaño objetivo (padding)
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
        
        dataset_path = Path(dataset_path)
        classes = ["SCD", "Healthy"]
        
        for class_name in classes:
            class_path = dataset_path / class_name
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
    
    # Mostrar estadísticas
    print("\n📊 ESTADÍSTICAS DEL DATASET PREPROCESADO:")
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
    
    # Generar visualización de una imagen de ejemplo
    sample_paths = dataset['train_paths'][:2]  # Tomar dos ejemplos
    for i, path in enumerate(sample_paths):
        save_path = f"preprocessing_example_{i+1}.png"
        preprocessor.visualize_preprocessing_effects(path, save_path)
    
    # Guardar configuración
    preprocessor.save_preprocessing_config()
    
    # Guardar dataset preprocesado
    print("\n💾 Guardando dataset preprocesado...")
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
```

Voy a corregir los errores de tipos en el código:

```python
        # Cargar rutas de archivos y etiquetas
        image_paths = []
        labels = []
        
        dataset_path = Path(dataset_path)
        classes = ["SCD", "Healthy"]
        
        for class_name in classes:
            class_path = dataset_path / class_name

->

        # Cargar rutas de archivos y etiquetas
        image_paths = []
        labels = []
        
        dataset_path_obj = Path(dataset_path)
        classes = ["SCD", "Healthy"]
        
        for class_name in classes:
            class_path = dataset_path_obj / class_name
```

Perfecto, los errores de tipos específicos fueron corregidos. Los errores de importación restantes no son problemáticos ya que las librerías están instaladas. Ahora ejecutemos el pipeline de preprocesamiento:

¡Excelente! El pipeline de preprocesamiento se completó exitosamente. Vamos a marcar la tarea como completada y verificar los resultados:

## 🎯 **PIPELINE DE PREPROCESAMIENTO COMPLETADO**

¡Excelente trabajo! El pipeline de preprocesamiento se ejecutó perfectamente. Aquí están los **resultados clave**:

### 📊 **Transformación del Dataset:**

**Antes del Preprocesamiento:**
- 1.25 GB de espacio
- 196 imágenes de 3000×4000 píxeles
- Sin optimizaciones

**Después del Preprocesamiento:**
- 69 MB de espacio (**95% de reducción!**)
- 196 imágenes de 512×512 píxeles
- Optimizadas para hologramas

### 🔧 **Técnicas Aplicadas:**

1. **Redimensionado Inteligente:** Preserva proporciones con padding
2. **Mejoras Holográficas:** Filtros FFT + realce de patrones de interferencia
3. **Normalización Robusta:** Usando percentiles 2-98 para evitar outliers
4. **Contraste Adaptativo:** CLAHE en espacio LAB para mejor distribución

### 📈 **División Estratificada:**
- **Entrenamiento:** 156 imágenes (Healthy: 90, SCD: 66)
- **Prueba:** 40 imágenes (Healthy: 23, SCD: 17)
- **Balance mantenido:** ~58% Healthy, ~42% SCD

### 📁 **Archivos Generados:**
✅ `preprocessed_dataset.npz` - Dataset optimizado (69MB)
✅ `preprocessing_example_1.png` - Visualización de efectos
✅ `preprocessing_example_2.png` - Comparación paso a paso
✅ `preprocessing_config.json` - Configuración y estadísticas

### 🚀 **Próximo Paso Crítico:**
Con solo **156 imágenes de entrenamiento**, necesitamos implementar **Data Augmentation agresiva** para generar variaciones sintéticas y prevenir overfitting.

¿Quieres que continúe implementando las estrategias de Data Augmentation? Esto incluirá:
- Rotaciones y flips específicos para hologramas
- Variaciones de contraste y brillo
- Transformaciones elásticas
- Filtros de ruido y blur
- Técnicas avanzadas como Mixup y CutMix

---

**User**

continua

---

**Cursor**

¡Perfecto! Continuemos con las estrategias de Data Augmentation. Marcaré la tarea correspondiente como en progreso e implementaré técnicas especializadas para hologramas.

```python
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
from albumentations.pytorch import ToTensorV2
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
            A.Flip(p=0.5),
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
    
    def augment_single_image(self, image: np.ndarray, label: int = None) -> List[Tuple[np.ndarray, int]]:
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
    
    def visualize_augmentations(self, image: np.ndarray, label: int = None, 
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
    sample_label = y_train[0]
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
```

Primero voy a instalar las librerías faltantes y corregir los errores de tipo:

Ahora voy a corregir los errores de tipo donde se pasa None en lugar de int:

```python
    def augment_single_image(self, image: np.ndarray, label: int = None) -> List[Tuple[np.ndarray, int]]:
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

->

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
```

```python
    def visualize_augmentations(self, image: np.ndarray, label: int = None, 
                               save_path: str = "augmentation_examples.png"):

->

    def visualize_augmentations(self, image: np.ndarray, label: int = 0, 
                               save_path: str = "augmentation_examples.png"):
```

```python
augmentator.visualize_augmentations(sample_image, sample_label)
```

```python
     # Generar visualización de augmentations
     sample_image = X_train[0]
     sample_label = y_train[0]
     augmentator.visualize_augmentations(sample_image, sample_label)

->

     # Generar visualización de augmentations
     sample_image = X_train[0]
     sample_label = int(y_train[0])
     augmentator.visualize_augmentations(sample_image, sample_label)
```

Los errores de importación no son problemáticos ya que las librerías están instaladas. Voy a ejecutar el script de data augmentation:

Voy a corregir el error de indentación:

```python
     # Generar visualización de augmentations
     sample_image = X_train[0]
      sample_label = int(y_train[0])
     augmentator.visualize_augmentations(sample_image, sample_label)

->

     # Generar visualización de augmentations
     sample_image = X_train[0]
     sample_label = int(y_train[0])
     augmentator.visualize_augmentations(sample_image, sample_label)
```

```python
    )
    
    # Generar visualización de augmentations
    sample_image = X_train[0]
    sample_label = int(y_train[0])
    augmentator.visualize_augmentations(sample_image, sample_label)
    
    # Aplicar data augmentation al dataset de entrenamiento
```

El problema es que estamos importando PyTorch que no necesitamos. Voy a quitar esa importación:

```python
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
```

Perfecto, ahora intentemos ejecutar el script:

Hay un problema con la API de Albumentations. Voy a corregir los nombres de las transformaciones:

```python
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
```

Ahora intentemos ejecutar el script nuevamente:

