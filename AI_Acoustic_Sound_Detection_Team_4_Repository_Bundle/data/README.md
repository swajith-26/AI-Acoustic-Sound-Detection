# Dataset setup
1. Obtain ESC-50 and/or UrbanSound8K from official sources and review their terms.
2. Inspect metadata and confirm labels before class mapping.
3. Create `manifest.csv` with `filepath,label,source_id`.
4. Do not claim alarm/normal labels exist unless confirmed; use a documented substitute or supplemental recordings if needed.
5. Keep all parts of one original recording in the same data split.
6. Do not commit restricted audio files.
