# BRACS Histopathology Classification

A research project for breast histopathology image classification on the BRACS dataset. The project uses frozen UNI features with a TransMIL + DTFD-MIL multiple-instance learning classifier, followed by a small calibration step for confidence and review flagging.


## Task

The submitted model works on the three-class BRACS grouping used in our experiments:

| Class | BRACS categories |
|---|---|
| Benign | Normal, PB, UDH |
| Atypia | FEA, ADH |
| Malignant | DCIS, IC |

This grouping was used because it is more stable for limited-data validation than a direct seven-class model, while still separating the main clinical decision regions. Seven-class and WSI-level extensions are planned as future work.

## Method

```text
ROI image
-> patch preprocessing
-> UNI ViT-L feature extraction
-> TransMIL + DTFD-MIL classifier
-> ensemble prediction
-> calibrated decision / review flag
```

The validation protocol uses patient-disjoint splits to reduce leakage between training and evaluation cases.

## Files

| File | Description |
|---|---|
| `streamlit_app.py` | Local dashboard for demo and result review |
| `live_roi_inference.py` | Live ROI inference helper |
| `run_validation_3x5.py` | Main 3-seed x 5-fold validation runner |
| `run_validation_1x5.py` | Smaller validation/debug runner |
| `rebuild_feature_cache_fixed.py` | Feature-cache builder |
| `posthoc_calibrate_predictions.py` | Post-hoc decision calibration |
| `decision_calibration.py` | Calibration utilities |
| `BRACS_TransMIL_DTFD_LOCAL.ipynb` | Notebook version of the local workflow |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Artifacts Expected

The repository is code-only. For training or live inference, place the local artifacts in these paths:

```text
data/bracs_roi/                       # BRACS ROI PNG images
weights/pytorch_model.bin             # UNI weights
work/bracs_features.h5                # generated feature cache
work/transmil_dtfd_output_*/          # trained checkpoints and prediction CSVs
```

The Streamlit dashboard can open without these files, but live prediction and result pages need the weights, feature cache, and checkpoint folder. Download BRACS data and UNI weights from their official sources. These paths are ignored by Git.

## Training

Typical local flow:

```bash
python rebuild_feature_cache_fixed.py
python run_validation_3x5.py
python posthoc_calibrate_predictions.py --help
```

The main run used:

```text
3 seeds x 5 folds
15 checkpoints
patient-disjoint validation
frozen UNI features
TransMIL + DTFD-MIL classifier
```

## Results

Research project run:

| Metric | Value |
|---|---:|
| Raw out-of-fold accuracy | 81.29% |
| Raw macro-F1 | 0.7885 |
| Frozen-threshold accuracy | 80.56% |
| Metric-optimized accuracy | 82.02% |
| Metric-optimized macro-F1 | 0.8013 |

These results are from the local project run and should be treated as research results, not clinical performance claims.

## Future Work

- WSI tiling using BRACS annotations
- four-class and seven-class classification experiments
- external validation with pathology-lab cases
- FastAPI inference service for deployment-style testing
- calibration monitoring and case-review audit logs

## License

Original project code is provided under `LICENSE-MIT.txt`. Third-party models, datasets, and research code keep their own licenses. See `THIRD_PARTY_NOTICES.md` before redistributing weights, checkpoints, or derived code.

## Disclaimer

This project is for research and education. It is not a diagnostic medical device.
