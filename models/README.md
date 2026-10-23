# Model Artifacts Directory

This folder stores serialized weights, transfer learning models, and hybrid machine learning classifiers.

---

## Artifact Descriptions

1. **`cnn_model.keras`**:
   - Custom Convolutional Neural Network baseline.
   - Architecture: Conv2D -> BatchNorm -> ReLU -> MaxPooling2D -> Dropout -> Dense.

2. **`mobilenet_model.keras`**:
   - Transfer learning architecture based on pretrained MobileNetV2 (ImageNet weights).
   - Fine-tuned top classification head with GlobalAveragePooling2D and Dropout.

3. **`mobilenet_extractor.keras`**:
   - Truncated MobileNetV2 feature extractor outputting 1280-dimensional feature embedding vectors.

4. **`hybrid_svm_model.joblib`**:
   - Scikit-learn Support Vector Machine (RBF kernel) trained on extracted MobileNetV2 deep features.
   - Fast inference with calibrated probability estimates.

5. **`hybrid_scaler.joblib`**:
   - StandardScaler fitted on deep feature vectors for numerical stability and optimal SVM margin maximization.

6. **`label_encoder.joblib`**:
   - Label encoder preserving class string mappings (e.g. `fresh_apple` -> 0, `rotten_apple` -> 1).
