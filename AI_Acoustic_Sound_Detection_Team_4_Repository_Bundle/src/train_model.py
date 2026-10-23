import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import load_wav, extract_features

FEATURES = ["rms", "zcr", "dominant_frequency", "spectral_energy", "spectral_centroid"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with filepath,label,source_id")
    ap.add_argument("--output", default="models")
    ap.add_argument("--sr", type=int, default=16000)
    args=ap.parse_args()
    manifest=pd.read_csv(args.manifest)
    required={"filepath","label","source_id"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    rows=[]
    for row in manifest.itertuples(index=False):
        sr,audio=load_wav(row.filepath,args.sr)
        feat=extract_features(audio,sr)
        feat.update(label=str(row.label),source_id=str(row.source_id),filepath=str(row.filepath))
        rows.append(feat)
    df=pd.DataFrame(rows).dropna(subset=FEATURES+["label","source_id"])
    X,y,groups=df[FEATURES],df["label"],df["source_id"]
    if y.nunique()<2:
        raise ValueError("At least two distinct classes are required.")
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    train_idx,test_idx=next(split.split(X,y,groups))
    Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx]; ytr,yte=y.iloc[train_idx],y.iloc[test_idx]
    models={"decision_tree":DecisionTreeClassifier(random_state=42,class_weight="balanced"),
            "random_forest":RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")}
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name,model in models.items():
        pipe=make_pipeline(SimpleImputer(strategy="median"),model)
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        print(f"\n=== {name} ===")
        print("Accuracy:",round(accuracy_score(yte,pred),4))
        print(classification_report(yte,pred,zero_division=0))
        labels=sorted(y.unique())
        print("Labels:",labels); print("Confusion matrix:")
        print(confusion_matrix(yte,pred,labels=labels))
        joblib.dump({"pipeline":pipe,"features":FEATURES,"labels":labels,"sample_rate":args.sr},out/f"{name}.joblib")
    print("\nInspect class counts/splits and confirm independent source recordings before interpreting metrics.")

if __name__=="__main__":
    main()
