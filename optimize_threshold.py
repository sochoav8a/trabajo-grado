#!/usr/bin/env python3
"""
Optimización de Threshold y Análisis de Desbalance
Proyecto: Clasificación de células sanas vs SCD
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import json

def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """
    Encontrar el threshold óptimo para maximizar una métrica
    
    Args:
        y_true: Etiquetas verdaderas
        y_scores: Probabilidades predichas
        metric: 'f1', 'balanced_accuracy', o 'youden'
    
    Returns:
        threshold óptimo, valor de la métrica
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if metric == 'youden':
        # Índice de Youden: maximizar TPR - FPR
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
    elif metric == 'f1':
        # Maximizar F1-score
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'balanced_accuracy':
        # Maximizar accuracy balanceado
        balanced_accs = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
            balanced_accs.append(balanced_acc)
        optimal_idx = np.argmax(balanced_accs)
    
    optimal_threshold = thresholds[optimal_idx]
    
    # Calcular métricas con threshold óptimo
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
    
    return optimal_threshold, y_pred_optimal

def analyze_class_distribution(y_train, y_test):
    """Analizar distribución de clases"""
    train_counts = np.bincount(y_train.astype(int))
    test_counts = np.bincount(y_test.astype(int))
    
    print("\nDistribución de clases:")
    print(f"  Train - Healthy: {train_counts[0]} ({train_counts[0]/len(y_train)*100:.1f}%)")
    print(f"  Train - SCD: {train_counts[1]} ({train_counts[1]/len(y_train)*100:.1f}%)")
    print(f"  Test - Healthy: {test_counts[0]} ({test_counts[0]/len(y_test)*100:.1f}%)")
    print(f"  Test - SCD: {test_counts[1]} ({test_counts[1]/len(y_test)*100:.1f}%)")
    
    return train_counts, test_counts

def plot_threshold_analysis(y_true, y_scores, model_name):
    """Visualizar análisis de threshold"""
    thresholds = np.linspace(0, 1, 100)
    metrics = {'precision': [], 'recall': [], 'f1': [], 'balanced_acc': []}
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        # Calcular métricas
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['balanced_acc'].append(balanced_acc)
    
    # Encontrar thresholds óptimos
    optimal_f1_idx = np.argmax(metrics['f1'])
    optimal_ba_idx = np.argmax(metrics['balanced_acc'])
    
    # Graficar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Métricas vs Threshold
    ax1.plot(thresholds, metrics['precision'], label='Precision', color='blue')
    ax1.plot(thresholds, metrics['recall'], label='Recall', color='red')
    ax1.plot(thresholds, metrics['f1'], label='F1-Score', color='green')
    ax1.plot(thresholds, metrics['balanced_acc'], label='Balanced Accuracy', color='purple')
    
    ax1.axvline(x=thresholds[optimal_f1_idx], color='green', linestyle='--', alpha=0.7)
    ax1.axvline(x=thresholds[optimal_ba_idx], color='purple', linestyle='--', alpha=0.7)
    ax1.axvline(x=0.5, color='black', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{model_name} - Métricas vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Trade-off Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ax2.plot(recall, precision, color='blue', lw=2)
    ax2.fill_between(recall, precision, alpha=0.2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{model_name} - Curva Precision-Recall')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nThresholds óptimos para {model_name}:")
    print(f"  F1-Score máximo: {metrics['f1'][optimal_f1_idx]:.3f} en threshold {thresholds[optimal_f1_idx]:.3f}")
    print(f"  Balanced Accuracy máximo: {metrics['balanced_acc'][optimal_ba_idx]:.3f} en threshold {thresholds[optimal_ba_idx]:.3f}")
    
    return thresholds[optimal_f1_idx], thresholds[optimal_ba_idx]

def evaluate_with_optimal_threshold(y_true, y_scores, threshold, model_name):
    """Evaluar con threshold óptimo"""
    y_pred = (y_scores >= threshold).astype(int)
    
    print(f"\nResultados con threshold óptimo ({threshold:.3f}) para {model_name}:")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'SCD']))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualizar matriz de confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Etiquetas
    classes = ['Healthy', 'SCD']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'{model_name} - Matriz de Confusión (Threshold={threshold:.3f})',
           ylabel='Verdadero',
           xlabel='Predicción')
    
    # Rotar etiquetas
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Añadir valores
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix_optimal.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Función principal"""
    print("ANÁLISIS DE THRESHOLD Y OPTIMIZACIÓN")
    print("=" * 60)
    
    # Cargar datos
    data = np.load('preprocessed_dataset.npz')
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Analizar distribución
    analyze_class_distribution(y_train, y_test)
    
    # Cargar resultados de modelos guardados (simulación)
    # En un caso real, cargarías las predicciones guardadas
    # Aquí simularemos con datos de ejemplo
    
    # Para CNN (alta precisión, bajo recall)
    np.random.seed(42)
    n_test = len(y_test)
    
    # Simular predicciones de CNN (conservadora)
    cnn_scores = np.random.beta(2, 5, n_test)  # Sesgo hacia valores bajos
    cnn_scores[y_test == 1] += np.random.normal(0.3, 0.1, np.sum(y_test == 1))
    cnn_scores = np.clip(cnn_scores, 0, 1)
    
    # Análisis de threshold para CNN
    print("\n" + "="*60)
    print("Análisis CNN Ultra Ligera")
    optimal_f1, optimal_ba = plot_threshold_analysis(y_test, cnn_scores, 'CNN_UltraLight')
    
    # Evaluar con diferentes thresholds
    print("\nComparación de thresholds:")
    print("\n1. Threshold estándar (0.5):")
    evaluate_with_optimal_threshold(y_test, cnn_scores, 0.5, 'CNN_Standard')
    
    print("\n2. Threshold óptimo F1:")
    evaluate_with_optimal_threshold(y_test, cnn_scores, optimal_f1, 'CNN_OptimalF1')
    
    print("\n3. Threshold óptimo Balanced Accuracy:")
    evaluate_with_optimal_threshold(y_test, cnn_scores, optimal_ba, 'CNN_OptimalBA')
    
    # Recomendaciones
    print("\n" + "="*60)
    print("RECOMENDACIONES:")
    print("1. Usar threshold óptimo en lugar de 0.5 fijo")
    print("2. Considerar pesos de clase durante el entrenamiento")
    print("3. Usar métricas balanceadas (F1, Balanced Accuracy) para optimización")
    print("4. Implementar SMOTE o técnicas de balanceo para el dataset pequeño")
    print("5. Considerar costo-beneficio médico al elegir threshold final")

if __name__ == "__main__":
    main() 