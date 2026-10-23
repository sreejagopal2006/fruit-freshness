# Dataset Documentation: Fresh and Rotten Fruits

This project utilizes image data based on the **Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality** (Mendeley Data DOI: [10.17632/bdd69gyhv8.1](https://data.mendeley.com/datasets/bdd69gyhv8/1)).

---

## 1. Directory Structure

Place raw fruit image folders under `data/raw/` (or configure a custom directory in `config.py` / via the Streamlit interface).

The dataset directory must follow the standard class-folder hierarchy:

```
data/
├── raw/
│   ├── fresh_apple/
│   │   ├── fresh_apple_001.jpg
│   │   ├── fresh_apple_002.png
│   │   └── ...
│   ├── rotten_apple/
│   │   ├── rotten_apple_001.jpg
│   │   └── ...
│   ├── fresh_banana/
│   ├── rotten_banana/
│   ├── fresh_orange/
│   ├── rotten_orange/
│   ├── fresh_grape/
│   ├── rotten_grape/
│   ├── fresh_guava/
│   ├── rotten_guava/
│   ├── fresh_jujube/
│   ├── rotten_jujube/
│   ├── fresh_pomegranate/
│   ├── rotten_pomegranate/
│   ├── fresh_strawberry/
│   └── rotten_strawberry/
└── README.md
```

### Alternative Common Folder Layouts Supported
The automatic dataset loader in `src/dataloader.py` seamlessly supports:
1. **Single Root with Class Folders**: `data/raw/<class_name>/*.jpg`
2. **Pre-split Kaggle/Mendeley Folders**:
   ```
   data/raw/
   ├── train/
   │   ├── freshapples/
   │   └── rottenapples/
   └── test/
       ├── freshapples/
       └── rottenapples/
   ```
   If `train` and `test` directories are detected, the dataloader preserves the splits or aggregates them cleanly without duplicate leakage.

---

## 2. Supported Image Formats
- JPEG (`.jpg`, `.jpeg`, `.JPG`, `.JPEG`)
- PNG (`.png`, `.PNG`)

Corrupted, zero-byte, or unreadable image files are automatically detected, logged, and filtered out during validation.

---

## 3. Dataset Classes (16 Classes)
1. `fresh_apple` / `freshapples`
2. `rotten_apple` / `rottenapples`
3. `fresh_banana` / `freshbanana`
4. `rotten_banana` / `rottenbanana`
5. `fresh_orange` / `freshoranges`
6. `rotten_orange` / `rottenoranges`
7. `fresh_grape`
8. `rotten_grape`
9. `fresh_guava`
10. `rotten_guava`
11. `fresh_jujube`
12. `rotten_jujube`
13. `fresh_pomegranate`
14. `rotten_pomegranate`
15. `fresh_strawberry`
16. `rotten_strawberry`

---

## 4. Manual Download Instructions
1. Visit the Mendeley Data repository: [https://data.mendeley.com/datasets/bdd69gyhv8/1](https://data.mendeley.com/datasets/bdd69gyhv8/1).
2. Download the dataset archive.
3. Unpack into `data/raw/`.
4. Run dataset inspection via CLI:
   ```bash
   python -m src.dataloader --data-dir data/raw
   ```
