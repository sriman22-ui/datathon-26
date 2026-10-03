import json
feat=open('ml_pipeline/features.py').read()
cells=[]
md=lambda s: cells.append({"cell_type":"markdown","metadata":{},"source":s})
code=lambda s: cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})
md("""# Smart Campus energy — prediction notebook (Track 1)

Loads the trained artifact `model.pkl` and predicts `energy_usage` for every row of the test file. **No training happens here.**

The model (trained offline, see the training notebook):
* **Average of two structural ridge regressions** (v6 design, and the v8 design with group-tuned regularisation + previous-usage slope by type x 4-hour block): building, hour, weekday and month effects; building-type x hour daily profiles; building-type x weekend; per-building slopes for temperature, occupancy and previous-hour usage; humidity x type; a cooling hinge `max(T-29, 0)`. Linear structure keeps predictions sensible outside the training range.
* **Missing inputs**: each missing feature is imputed by an average of LightGBM and XGBoost models trained for that exact availability pattern (fit on train rows plus unlabeled test features, test rows weighted x10, to follow test's shifted feature distribution), then the full ridge is applied; small learned offsets per missing feature.
* Rows missing **both** occupancy and previous_usage: average of the ridge prediction and a NaN-native LightGBM.

Works on shuffled rows with or without an `id` column; output order = input order.""")
code("""import os
import numpy as np
import pandas as pd
import joblib

TEST_PATH = os.environ.get("DATATHON_INPUT_PATH", "test.csv")
OUTPUT_PATH = os.environ.get("DATATHON_OUTPUT_PATH", "predictions.csv")
MODEL_PATH = "model.pkl\"""".replace('\\"','"'))
code("# ---- feature engineering: identical to training (ml_pipeline/features.py) ----\n"+feat.split('"""',2)[2].strip())
code('''def xgb_frame(df, av):
    X = gbm_frame(df, av).copy(); X["b"] = X.b.cat.codes
    return X

def impute(pair, df, av):
    """Average of the LightGBM and XGBoost imputers for this availability pattern."""
    return 0.5 * (pair[0].predict(gbm_frame(df, av)) + pair[1].predict(xgb_frame(df, av)))

def predict(art, df):
    """Pattern-aware prediction (identical to ml_pipeline/artifact.py::predict)."""
    df = clean(df).reset_index(drop=True)
    out = np.zeros(len(df))
    miss = df[list(NUMS.values())].isna().values
    keys = [tuple(r) for r in miss]
    for pat in set(keys):
        rows = np.array([i for i, k in enumerate(keys) if k == pat])
        sub = df.iloc[rows].copy()
        ms = tuple(a for a, m in zip(FULL, pat) if m)
        av = tuple(a for a in FULL if a not in ms)
        for mv in ms:  # impute each missing input from the available ones
            sub[NUMS[mv]] = impute(art["imputers"][(mv, av)], df.iloc[rows], av)
        p = 0.5 * (art["main"].predict(design(sub, FULL, art["hinge"])) + art["main8"].predict(design_v8(sub, art["hinge"])))
        p = p + sum(art["offsets"][m] for m in ms)
        if "o" in ms and "p" in ms and art["c_mode"] != "ridge":
            t = art["tree"].predict(gbm_frame(df.iloc[rows], FULL))
            p = t if art["c_mode"] == "tree" else 0.5 * (p + t)
        out[rows] = p
    return out''')
code('''art = joblib.load(MODEL_PATH)
test = pd.read_csv(TEST_PATH)
preds = predict(art, test)
assert len(preds) == len(test) and np.isfinite(preds).all()
print(art["version"], "| rows:", len(test), "| mean prediction: %.2f" % preds.mean())''')
code('''pd.DataFrame({"prediction": preds}).to_csv(OUTPUT_PATH, index=False)
print(f"wrote {len(preds)} predictions to {OUTPUT_PATH}")''')
nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python"}},"cells":cells}
json.dump(nb,open('deliver/prediction_notebook.ipynb','w'),indent=1)
