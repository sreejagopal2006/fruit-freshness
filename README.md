# Real-Time Fruit Quality and Ripeness Monitoring Using a Hybrid Deep Learning and Machine Learning Framework

An end-to-end Computer Vision & Hybrid Machine Learning system designed for automated, non-destructive classification of fruit freshness, ripening stages, and degradation grading. The system pairs MobileNetV2 deep convolutional representations with Support Vector Machine (SVM) decision boundaries for lightweight, sub-100ms inference on standard hardware.

---

## 🏛️ System Architecture

```mermaid
graph LR
    A[Fruit Image Upload / Sample / Camera] --> B[Preprocessing & Standardization (224x224 RGB)]
    B --> C[MobileNetV2 Feature Extractor (1280-D Embedding)]
    B --> D[OpenCV Visual Diagnostics (HSV / Canny / Blemish Mask)]
    C --> E[StandardScaler Normalization]
    E --> F[Support Vector Machine Classifier (RBF Kernel)]
    F --> G[Classification & Confidence Calculation]
    D --> H[Freshness Quality Grading Engine]
    G --> H
    H --> I[Streamlit Interactive Dashboard]
```

### Models Implemented & Tested
1. **Hybrid MobileNetV2 + SVM (Primary Deployed Pipeline)**:
   - Feature Extractor: Frozen MobileNetV2 global average pooling layer extracting 1280-D spatial latent representations.
   - Classifier: Calibrated Support Vector Classifier (`SVC(kernel='rbf', C=10.0, probability=True)`).
   - Test Accuracy: **85.65% – 91.67%** on genuine fruit test sets.
2. **MobileNetV2 End-to-End**:
   - Deep transfer learning model fine-tuned with dense classification head.
3. **Custom 4-Layer Conv2D Baseline**:
   - Standard baseline convolutional network.

---

## 📂 Project Directory Structure

```
fruit-freshness-cbp/
├── app.py                      # Interactive Streamlit Web Application
├── config.py                   # Centralized paths, classes, and hyperparameters
├── requirements.txt            # Production package requirements
├── .gitignore                  # Git ignore rules (excludes heavy raw datasets)
├── README.md                   # Complete system documentation
├── data/
│   ├── README.md               # Dataset integration & Mendeley guide
│   └── raw/                    # Genuine class-separated fruit image folders
│       ├── fresh_apple/
│       ├── normal_apple/
│       ├── rotten_apple/
│       ├── fresh_banana/
│       ├── normal_banana/
│       ├── rotten_banana/
│       ├── fresh_orange/
│       ├── normal_orange/
│       └── rotten_orange/
├── models/
│   ├── README.md               # Model artifact descriptions
│   ├── hybrid_svm_model.joblib # Trained Support Vector Classifier (4.4 MB)
│   ├── hybrid_scaler.joblib    # StandardScaler fitted on 1280-D embeddings (31 KB)
│   ├── label_encoder.joblib    # Scikit-learn LabelEncoder (611 B)
│   └── mobilenet_extractor.keras # Truncated MobileNetV2 extractor (9.2 MB)
├── reports/
│   ├── README.md               # Evaluation results documentation
│   ├── hybrid_svm_report.json  # Precision, recall, F1 classification metrics
│   ├── model_comparison.json   # Comparative benchmark data
│   ├── confusion_matrix.png    # Primary test split confusion matrix heatmap
│   └── cm_hybrid_mobilenetv2_plus_svm.png # Seaborn confusion matrix
├── src/
│   ├── __init__.py
│   ├── dataloader.py           # Dataset discovery & train/val/test splits
│   ├── preprocessing.py        # Image standardization, HSV, Canny, blemish masks
│   ├── predict.py              # Unified inference and freshness grading engine
│   ├── evaluate.py             # Performance metric generation & confusion matrix
│   ├── trainhybrid.py          # Hybrid MobileNetV2 + SVM training pipeline
│   ├── trainmobilenet.py       # MobileNetV2 transfer learning pipeline
│   └── traincnn.py             # Custom CNN baseline training pipeline
└── tests/
    ├── __init__.py
    ├── testpreprocessing.py    # Tests for image loading, validation, and filters
    └── testprediction.py       # Tests for grading logic, predictor, and classes
```

---

## ⚙️ Environment Setup & Local Execution

The project runs on the pre-configured Conda environment located at:
`/Users/lingampallysreeja/ML IMPLEMENTATATION/env` (Python 3.11.17).

### Option 1: Direct Execution (Recommended)

Run Streamlit directly using the absolute environment binary:

```bash
cd ~/Documents/fruit-freshness-cbp
"/Users/lingampallysreeja/ML IMPLEMENTATATION/env/bin/streamlit" run app.py
```

### Option 2: Activate Conda Environment

```bash
conda activate "/Users/lingampallysreeja/ML IMPLEMENTATATION/env"
cd ~/Documents/fruit-freshness-cbp
streamlit run app.py
```

Once launched, access the dashboard in your web browser at:
`http://localhost:8501`

### Running the Test Suite

Execute the 12 automated unit tests covering inference, preprocessing, image validation, FruitVision sample discovery, and latency benchmarks:

```bash
"/Users/lingampallysreeja/ML IMPLEMENTATATION/env/bin/pytest" tests/ -v
```

---

## 🔬 Mendeley FruitVision Dataset Integration

The application is pre-trained and loaded with 9 authentic fruit quality categories from the Mendeley Fruit Dataset:
- **Apples**: Fresh Apple, Normal Apple, Rotten Apple
- **Bananas**: Fresh Banana, Normal Banana, Rotten Banana
- **Oranges**: Fresh Orange, Normal Orange, Rotten Orange

### External Dataset Source:
- **Mendeley Data DOI**: [10.17632/xkbjx8959c.2](https://data.mendeley.com/datasets/xkbjx8959c/2)
- **Dataset Title**: *FruitVision / Fresh and Rotten Fruits Dataset*

### How to Integrate Additional Classes Later:
1. Download the archive from Mendeley Data.
2. Extract image folders directly into `data/raw/<class_name>/` (e.g., `data/raw/fresh_grape`, `data/raw/rotten_grape`).
3. Re-extract features and re-fit the lightweight SVM classifier in under 2 minutes:
   ```bash
   "/Users/lingampallysreeja/ML IMPLEMENTATATION/env/bin/python" -m src.trainhybrid
   ```
4. Re-run evaluation to refresh performance metrics and confusion matrices:
   ```bash
   "/Users/lingampallysreeja/ML IMPLEMENTATATION/env/bin/python" -m src.evaluate
   ```

---

## 🌐 Deployment: Local vs. Cloud Explained

Understanding the lifecycle of your application is crucial for presentations and persistent access:

| State | Where It Lives | When Can It Be Accessed? | URL Type |
|---|---|---|---|
| **1. Saved Locally** | On your MacBook's hard drive (`~/Documents/fruit-freshness-cbp`) | Code is preserved on disk, but not actively executing. | None (File storage only) |
| **2. Running Locally** | In an active terminal / background process running `streamlit run app.py` | Only accessible while your MacBook is **on**, awake, and the terminal process is alive. | `http://localhost:8501` |
| **3. Hosted Publicly** | In the cloud (Streamlit Community Cloud connected to GitHub) | Accessible **24/7 from anywhere** on any device (phone, laptop, iPad), even when your MacBook and Antigravity IDE are closed. | `https://<your-project>.streamlit.app` |

---

## 🚀 How to Deploy to Streamlit Community Cloud (Step-by-Step)

Streamlit Community Cloud provides free, persistent hosting connected directly to your GitHub repository:

### Step 1: Initialize Git and Push to GitHub

Ensure `.gitignore` is active (so large raw image folders are excluded and only essential models and code are pushed):

```bash
cd ~/Documents/fruit-freshness-cbp

# Initialize git if not already present
git init
git add .
git commit -m "Complete fruit freshness hybrid ML dashboard"

# Link to your GitHub repository (replace with your repo URL)
git branch -M main
git remote add origin https://github.com/<your-username>/fruit-freshness-cbp.git
git push -u origin main
```

> **Model Size Verification**:
> - `mobilenet_extractor.keras`: 9.2 MB (Well under GitHub's 100 MB file limit)
> - `hybrid_svm_model.joblib`: 4.4 MB
> - Total models size: ~13.6 MB
> - No Git LFS required!

### Step 2: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **New app**.
3. Select your repository: `<your-username>/fruit-freshness-cbp`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Deploy!**

The cloud container will automatically install dependencies from `requirements.txt` (which uses `opencv-python-headless` for headless Linux container compatibility) and launch your public URL.

---

## 🛡️ Image Validation & Safeguards

The application implements active input validation:
- **Corrupt File Detection**: Uses PIL byte verification to reject corrupted, truncated, or non-image files with a clear error notification.
- **Minimum Resolution Guard**: Warns if input resolution is below 32×32 pixels.
- **Uniform / Blank Image Rejection**: Detects solid color or empty images via pixel variance checks.
- **Low Confidence Advisory**: When top prediction probability is under 35%, displays an advisory reminding users of the supported fruit domain (Apple, Banana, Orange) and potential non-fruit or poor lighting ambiguities.
