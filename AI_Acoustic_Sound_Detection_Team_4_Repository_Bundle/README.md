# AI-Based Emergency Acoustic Event Detection and Alert System

**Course:** ANALOG ELECTRONIC CIRCUIT DESIGN — 25EC 1302E  
**Team:** 4  
**Team members:** Y. SWAJITH (2510040011), C. VANSHIKA (2510040017), Y. SOMANATH (2510040018), SAI KRISHNA (2510040089)  
**Guide/CC-CI:** Dr A. Bhuvaneshwari  
**Provided GitHub URL:** https://github.com/swajith-26/AI-Acoustic-Sound-Detection.git

## Project summary
An undergraduate ECE project proposing acoustic event detection through analog audio acquisition, digital signal processing and supervised classification.

## Status
This bundle includes starter feature extraction/model-training code and the report/presentation. It does not claim hardware integration or measured accuracy results are complete. Confirm the final dataset labels before training.

## Structure
- `src/features.py` — WAV loading and compact feature extraction
- `src/train_model.py` — train/evaluate Decision Tree and Random Forest from a labelled manifest
- `src/realtime_demo.py` — placeholder for future live capture integration
- `requirements.txt` — Python dependencies
- `data/README.md` — dataset setup and split guidance
- `docs/implementation-status.md` — completion checklist

## Dataset setup
Obtain datasets through official project sources and follow their terms. Create `data/manifest.csv` with columns `filepath,label,source_id`. Source ID identifies the original recording. Keep all segments from the same source in the same data split.

The proposed labels may not all exist in one dataset. Inspect metadata and document any substitutes or supplemental recordings.

## Run
```bash
pip install -r requirements.txt
python src/train_model.py --manifest data/manifest.csv --output models
```

This is an educational prototype, not a certified emergency alert system.
