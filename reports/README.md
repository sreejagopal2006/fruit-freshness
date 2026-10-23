# Evaluation and Performance Reports

This folder stores model evaluation metrics, confusion matrices, and comparative benchmark results.

---

## Generated Reports

1. **`metrics.json`**:
   - Comprehensive test-set evaluation metrics including:
     - Accuracy
     - Weighted Precision
     - Weighted Recall
     - Weighted F1-Score
     - Per-class classification reports.

2. **`confusion_matrix.png`**:
   - High-resolution Seaborn heatmap showing true vs predicted class distributions.

3. **`model_comparison.json`**:
   - Comparative tabular analysis across all 3 architectures:
     - Custom CNN Baseline
     - MobileNetV2 Transfer Learning
     - Hybrid Deep Feature Extractor + SVM.

4. **`training_history.json`**:
   - Epoch-by-epoch loss, validation loss, accuracy, and validation accuracy curves.
