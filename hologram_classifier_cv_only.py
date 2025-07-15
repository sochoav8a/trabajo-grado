#!/usr/bin/env python3
"""
Clasificador de hologramas - Solo validación cruzada
Usa todos los datos disponibles sin separar test set
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MinimalFeatureExtractor:
    """Extractor minimalista para máxima generalización"""
    
    def extract_features(self, image):
        """Extrae solo características muy básicas"""
        features = []
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
            
        # 1. Solo 3 estadísticas básicas
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray)
        ])
        
        # 2. Histograma muy simplificado (4 bins)
        hist, _ = np.histogram(gray, bins=4, range=(0, 256), density=True)
        features.extend(hist)
        
        # 3. Un solo valor de textura
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        features.append(np.std(lbp))
        
        return np.array(features)

def load_all_data():
    """Carga TODOS los datos disponibles sin separar test"""
    print("\n📂 Cargando todos los datos disponibles...")
    
    # Cargar datos preprocesados
    data = np.load('preprocessed_dataset.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Combinar TODO
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    
    print(f"   Total de imágenes: {len(y_all)}")
    print(f"   Distribución: {np.sum(y_all==0)} Healthy, {np.sum(y_all==1)} SCD")
    print(f"   Proporción SCD: {np.mean(y_all):.2%}")
    
    return X_all, y_all

def evaluate_models_cv_only(X_features, y, n_splits=10, n_repeats=5):
    """Evaluación exhaustiva solo con validación cruzada"""
    
    print(f"\n🔍 Evaluación con {n_repeats}x{n_splits}-fold CV ({n_repeats*n_splits} evaluaciones)...")
    
    # Modelos ultra-simples con máxima regularización
    models = {
        'decision_tree_depth2': DecisionTreeClassifier(
            max_depth=2,  # Árbol muy superficial
            min_samples_leaf=10,  # Muchas muestras por hoja
            random_state=42
        ),
        'logistic_l2_high': LogisticRegression(
            C=0.01,  # Regularización muy alta
            penalty='l2',
            solver='liblinear',
            random_state=42
        ),
        'rf_tiny': RandomForestClassifier(
            n_estimators=10,  # Muy pocos árboles
            max_depth=2,
            min_samples_leaf=15,  # Muchas muestras por hoja
            max_features=2,  # Pocas características por split
            random_state=42
        ),
        'svm_linear_c001': SVC(
            kernel='linear',
            C=0.001,  # Regularización extrema
            probability=True,
            random_state=42
        )
    }
    
    # Crear pipeline con selección de características
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 Evaluando {name}...")
        
        # Pipeline con selección mínima de características
        pipeline = Pipeline([
            ('selector', SelectKBest(f_classif, k=5)),  # Solo 5 características
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Validación cruzada repetida estratificada
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        
        # Métricas múltiples
        scoring = {
            'roc_auc': 'roc_auc',
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'accuracy': 'accuracy'
        }
        
        # Evaluar
        cv_results = cross_validate(
            pipeline, X_features, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calcular estadísticas
        results[name] = {
            'test_auc_mean': np.mean(cv_results['test_roc_auc']),
            'test_auc_std': np.std(cv_results['test_roc_auc']),
            'test_auc_all': cv_results['test_roc_auc'],
            'train_auc_mean': np.mean(cv_results['train_roc_auc']),
            'train_auc_std': np.std(cv_results['train_roc_auc']),
            'test_ba_mean': np.mean(cv_results['test_balanced_accuracy']),
            'test_ba_std': np.std(cv_results['test_balanced_accuracy']),
            'overfit_score': np.mean(cv_results['train_roc_auc']) - np.mean(cv_results['test_roc_auc'])
        }
        
        # Mostrar resultados
        print(f"   Test AUC: {results[name]['test_auc_mean']:.3f} ± {results[name]['test_auc_std']:.3f}")
        print(f"   Train AUC: {results[name]['train_auc_mean']:.3f} ± {results[name]['train_auc_std']:.3f}")
        print(f"   Overfitting: {results[name]['overfit_score']:.3f}")
        print(f"   Test Balanced Acc: {results[name]['test_ba_mean']:.3f} ± {results[name]['test_ba_std']:.3f}")
        
        # Mostrar distribución de AUC
        auc_scores = results[name]['test_auc_all']
        print(f"   Distribución AUC: min={np.min(auc_scores):.3f}, "
              f"Q1={np.percentile(auc_scores, 25):.3f}, "
              f"median={np.median(auc_scores):.3f}, "
              f"Q3={np.percentile(auc_scores, 75):.3f}, "
              f"max={np.max(auc_scores):.3f}")
    
    return results

def analyze_feature_importance(X_features, y):
    """Analiza qué características son más informativas"""
    print("\n🔬 Análisis de características...")
    
    extractor = MinimalFeatureExtractor()
    feature_names = ['mean', 'std', 'median', 'hist_0', 'hist_1', 'hist_2', 'hist_3', 'lbp_std']
    
    # ANOVA F-score
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X_features, y)
    
    # Mostrar scores
    print("\n   F-scores de características:")
    for i, (name, score) in enumerate(zip(feature_names, selector.scores_)):
        print(f"   {i+1}. {name}: {score:.2f}")

def main():
    print("🔬 EVALUACIÓN HONESTA CON VALIDACIÓN CRUZADA")
    print("=" * 60)
    print("Usando TODOS los datos disponibles (sin test set separado)")
    
    # 1. Cargar todos los datos
    X_all, y_all = load_all_data()
    
    # 2. Extraer características mínimas
    print("\n📐 Extrayendo características mínimas...")
    extractor = MinimalFeatureExtractor()
    
    X_features = []
    for i, img in enumerate(X_all):
        if i % 50 == 0:
            print(f"   Progreso: {i}/{len(X_all)}")
        features = extractor.extract_features(img)
        X_features.append(features)
    
    X_features = np.array(X_features)
    print(f"   Características extraídas: {X_features.shape[1]}")
    
    # 3. Analizar características
    analyze_feature_importance(X_features, y_all)
    
    # 4. Evaluación exhaustiva con CV
    results = evaluate_models_cv_only(X_features, y_all, n_splits=10, n_repeats=5)
    
    # 5. Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    # Ordenar por AUC de test
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_auc_mean'], reverse=True)
    
    print("\nRanking por AUC (validación cruzada):")
    for i, (name, res) in enumerate(sorted_models):
        print(f"{i+1}. {name}:")
        print(f"   - AUC: {res['test_auc_mean']:.3f} ± {res['test_auc_std']:.3f}")
        print(f"   - Balanced Acc: {res['test_ba_mean']:.3f} ± {res['test_ba_std']:.3f}")
        print(f"   - Overfitting: {res['overfit_score']:.3f}")
    
    # Mejor modelo
    best_model = sorted_models[0]
    print(f"\n✅ Mejor modelo: {best_model[0]}")
    print(f"   AUC promedio: {best_model[1]['test_auc_mean']:.3f}")
    
    # Análisis de variabilidad
    print("\n📈 Análisis de estabilidad:")
    for name, res in sorted_models[:2]:  # Top 2 modelos
        auc_scores = res['test_auc_all']
        print(f"\n{name}:")
        print(f"   - Coeficiente de variación: {np.std(auc_scores)/np.mean(auc_scores):.3f}")
        print(f"   - Rango intercuartil: {np.percentile(auc_scores, 75) - np.percentile(auc_scores, 25):.3f}")
        print(f"   - % folds con AUC > 0.7: {np.mean(auc_scores > 0.7)*100:.1f}%")
        print(f"   - % folds con AUC > 0.8: {np.mean(auc_scores > 0.8)*100:.1f}%")
        print(f"   - % folds con AUC > 0.9: {np.mean(auc_scores > 0.9)*100:.1f}%")
    
    # Guardar resultados
    np.savez(
        'cv_only_results.npz',
        results=results,
        X_features=X_features,
        y_all=y_all
    )
    
    print("\n💾 Resultados guardados en 'cv_only_results.npz'") 

if __name__ == "__main__":
    main() 