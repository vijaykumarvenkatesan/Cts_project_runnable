import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import re
from rapidfuzz import process

st.set_page_config(page_title="Medical Fraud Detection", layout="wide")
st.title("🩺 Medical Fraud Detection — Inference App")

# --------------------------
# 🎨 Custom Styling (Deep Blue + Purple Theme)
# --------------------------
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1aff;  /* Deep Blue */
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;  /* White text */
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #8000ff;  /* Purple */
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #4b0082;  /* Darker purple */
        color: white;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #1a1aff !important;  /* Deep Blue numbers */
    }

    /* Tables */
    .stDataFrame thead tr th {
        background-color: #8000ff !important; /* Purple header */
        color: white !important;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f3e6ff !important;  /* Very light purple rows */
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #ffffff !important;  /* White rows */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 (The rest of the logic remains the same as the last version I gave you)
# Keep all the feature engineering, aggregation, EDA, and model prediction logic unchanged.
# Just copy-paste the full previous working code, and replace ONLY the <style> block with this new one.

# --------------------------
# Sidebar uploads
# --------------------------
st.sidebar.header("Upload dataset(s) and model")

merged_file = st.sidebar.file_uploader("Merged dataset (optional)", type=["csv"], key="merged")
benef_file  = st.sidebar.file_uploader("Beneficiary CSV", type=["csv"], key="benef")
inp_file    = st.sidebar.file_uploader("Inpatient CSV", type=["csv"], key="inp")
out_file    = st.sidebar.file_uploader("Outpatient CSV", type=["csv"], key="out")
train_file  = st.sidebar.file_uploader("Train/Providers CSV (optional)", type=["csv"], key="train")

model_file = st.sidebar.file_uploader("Pretrained model (.pkl or joblib)", type=["pkl","joblib"], key="model")
threshold  = st.sidebar.slider("Decision threshold", min_value=0.05, max_value=0.95,
                               value=0.5, step=0.01, key="threshold")
run = st.sidebar.button("Run Inference", key="run")

# --------------------------
# Utility: fuzzy column resolver
# --------------------------
def resolve_column_name(df: pd.DataFrame, expected: str, score_cutoff=80):
    if df is None or df.empty:
        return None
    match = process.extractOne(expected, df.columns, score_cutoff=score_cutoff)
    return match[0] if match else None

def get_col(df: pd.DataFrame, expected: str):
    col = resolve_column_name(df, expected)
    return col if col else None

# --------------------------
# Feature engineering functions
# --------------------------
def to_dt(x): return pd.to_datetime(x, errors="coerce")

BOOL_MAP = {"yes":1,"true":1,"1":1,"y":1,"no":0,"false":0,"0":0,"n":0}

def normalize_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower().map(BOOL_MAP)
    if s.isna().any():
        num = pd.to_numeric(series, errors='coerce')
        if num.notna().any():
            s = (num > 0).astype(int)
    return s.fillna(0).astype(int)

def parse_claim_dates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    start_col, end_col = get_col(d,"ClaimStartDt"), get_col(d,"ClaimEndDt")
    adm_col, dis_col   = get_col(d,"AdmissionDt"), get_col(d,"DischargeDt")
    if start_col: d[start_col] = to_dt(d[start_col])
    if end_col:   d[end_col]   = to_dt(d[end_col])
    if adm_col:   d[adm_col]   = to_dt(d[adm_col])
    if dis_col:   d[dis_col]   = to_dt(d[dis_col])
    if start_col and end_col:
        d["claim_days"] = (d[end_col]-d[start_col]).dt.days.clip(lower=0)
    if adm_col and dis_col:
        d["los_days"] = (d[dis_col]-d[adm_col]).dt.days.clip(lower=0)
    return d

def claim_level_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    money_cols = [c for c in d.columns if re.search(r"reimbursed|payment|paid", c, re.I)]
    d["money_sum"] = d[money_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1) if money_cols else 0.0
    diag_cols = [c for c in d.columns if re.search(r"diag", c, re.I)]
    proc_cols = [c for c in d.columns if re.search(r"proc", c, re.I)]
    d["n_diag_codes"] = d[diag_cols].astype(str).replace({"nan":np.nan}).notna().sum(axis=1) if diag_cols else 0
    d["n_proc_codes"] = d[proc_cols].astype(str).replace({"nan":np.nan}).notna().sum(axis=1) if proc_cols else 0
    phys_cols = [c for c in d.columns if "physician" in c.lower()]
    d["n_physicians_on_claim"] = d[phys_cols].astype(str).replace({"nan":np.nan}).nunique(axis=1) if phys_cols else 0
    ded_cols = [c for c in d.columns if "deductible" in c.lower()]
    d["deductible_paid_sum"] = d[ded_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1) if ded_cols else 0.0
    return d

def provider_agg(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    prov_col = get_col(df,"Provider")
    if not prov_col: return pd.DataFrame(columns=["Provider"])
    grp = df.groupby(prov_col)
    agg = grp.agg(
        **{
            f"{tag}_n_claims": (get_col(df,"ClaimID"),"count") if get_col(df,"ClaimID") else (prov_col,"size"),
            f"{tag}_total_money":("money_sum","sum") if "money_sum" in df.columns else (prov_col,"size"),
            f"{tag}_mean_money":("money_sum","mean") if "money_sum" in df.columns else (prov_col,"size"),
        }
    )
    if "claim_days" in df.columns:
        agg[f"{tag}_mean_claim_days"] = grp["claim_days"].mean()
    if "los_days" in df.columns and tag=="inp":
        agg[f"{tag}_mean_los"] = grp["los_days"].mean()
    return agg.reset_index().rename(columns={prov_col:"Provider"})

def build_beneficiary_rollup(benef: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    bene_col, prov_col = get_col(benef,"BeneID"), get_col(links,"Provider")
    if not bene_col or not prov_col:
        return pd.DataFrame(columns=["Provider"])
    b = benef.copy()
    bool_cols = [c for c in b.columns if c.lower().startswith("chroniccond") or c.lower().startswith("renal") or "isdeceased" in c.lower()]
    for c in bool_cols: b[c] = normalize_bool(b[c])
    agg = {c:"mean" for c in bool_cols}
    if not agg:
        return pd.DataFrame(columns=["Provider"])
    res = links.merge(b, left_on=bene_col, right_on=bene_col, how="left")
    if prov_col not in res.columns:
        return pd.DataFrame(columns=["Provider"])
    res = res.groupby(prov_col).agg(agg)
    res.columns = ["bene_"+c for c in res.columns]
    return res.reset_index().rename(columns={prov_col:"Provider"})

def aggregate_provider_features(train, benef, inp, outp, target_name: str = None):
    prov_col_train = get_col(train,"Provider")
    inp, outp = parse_claim_dates(inp), parse_claim_dates(outp)
    inp, outp = claim_level_features(inp), claim_level_features(outp)
    prov_inp, prov_outp = provider_agg(inp,"inp"), provider_agg(outp,"outp")
    bene_col_inp, prov_col_inp = get_col(inp,"BeneID"), get_col(inp,"Provider")
    bene_col_outp, prov_col_outp = get_col(outp,"BeneID"), get_col(outp,"Provider")
    links_parts = []
    if bene_col_inp and prov_col_inp: links_parts.append(inp[[bene_col_inp,prov_col_inp]].drop_duplicates())
    if bene_col_outp and prov_col_outp: links_parts.append(outp[[bene_col_outp,prov_col_outp]].drop_duplicates())
    links = pd.concat(links_parts, ignore_index=True).drop_duplicates() if links_parts else pd.DataFrame(columns=["BeneID","Provider"])
    prov_bene = build_beneficiary_rollup(benef, links)
    features = train[[prov_col_train]].copy() if prov_col_train else pd.DataFrame()
    features = features.rename(columns={prov_col_train:"Provider"})
    for df in [prov_inp, prov_outp, prov_bene]:
        if not df.empty: features = features.merge(df,on="Provider",how="left")
    for c in features.select_dtypes(include=[np.number]).columns:
        features[c] = features[c].fillna(0)
    return features, features.drop(columns=["Provider"], errors="ignore"), None

# --------------------------
# Model / EDA
# --------------------------
def load_model_with_schema(path_or_buffer):
    obj = joblib.load(path_or_buffer)
    if isinstance(obj,dict) and "model" in obj: return obj["model"], obj.get("feature_names")
    model=obj; feat=getattr(model,'feature_names_in_',None)
    if feat is not None: feat=list(feat)
    try: feat=model.get_booster().feature_names
    except Exception: pass
    return model, feat

def predict_and_output(model, train_feats, df, threshold=0.5):
    X = df.drop(columns=["Provider"], errors="ignore") if "Provider" in df.columns else df.copy()
    provider = df["Provider"] if "Provider" in df.columns else pd.Series(np.arange(len(df)),name="Provider")
    if train_feats:
        for m in [c for c in train_feats if c not in X.columns]: X[m]=0.0
        X=X.reindex(columns=train_feats)
    proba=model.predict_proba(X)[:,1]
    pred=(proba>=threshold).astype(int)
    return pd.DataFrame({"Provider":provider,"fraud_score":proba,"fraud_pred":pred})

# --------------------------
# EDA with grid layout
# --------------------------
def run_eda(agg_df: pd.DataFrame):
    st.subheader("EDA: Aggregated Provider-level Dataset")
    st.write("Shape:", agg_df.shape)

    st.markdown("**Numeric summary**")
    st.dataframe(agg_df.describe().T)

    num_cols = agg_df.select_dtypes(include=[np.number]).columns
    top_feats = [c for c in num_cols][:12]

    if top_feats:
        st.markdown("**Feature distributions (first 12 numeric features)**")
        for i in range(0, len(top_feats), 4):
            cols = st.columns(4)
            for j, c in enumerate(top_feats[i:i+4]):
                with cols[j]:
                    fig, ax = plt.subplots()
                    ax.hist(agg_df[c].dropna(), bins=30, color="#007bff", edgecolor="black")
                    ax.set_title(c)
                    st.pyplot(fig)

# --------------------------
# Main Run
# --------------------------
if run:
    if not model_file: st.error("Please upload a pretrained model."); st.stop()

    if merged_file is not None:
        merged=pd.read_csv(merged_file,low_memory=False)
        st.success("Merged dataset uploaded successfully!"); st.dataframe(merged.head(25))
        prov_col=get_col(merged,"Provider")
        train=pd.DataFrame({"Provider":merged[prov_col].unique()}) if prov_col else pd.DataFrame()
        agg_df,X,y_series=aggregate_provider_features(train,merged,merged,merged)

    elif all([benef_file,inp_file,out_file]):
        benef=pd.read_csv(benef_file,low_memory=False)
        inp=pd.read_csv(inp_file,low_memory=False)
        outp=pd.read_csv(out_file,low_memory=False)
        prov_col=get_col(inp,"Provider") or get_col(outp,"Provider")
        train=pd.DataFrame({"Provider":inp[prov_col].unique()}) if prov_col else pd.DataFrame()
        agg_df,X,y_series=aggregate_provider_features(train,benef,inp,outp)

    else:
        st.error("Please upload either a merged dataset OR all four datasets."); st.stop()

    run_eda(agg_df)

    model,train_feats=load_model_with_schema(model_file)
    st.info(f"Model loaded. Expects {len(train_feats) if train_feats else 'unknown'} features.")

    out=predict_and_output(model,train_feats,agg_df,threshold=threshold)
    st.subheader("Top 50 providers by fraud score")
    st.dataframe(out.sort_values("fraud_score",ascending=False).head(50))

    csv_agg=agg_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download aggregated features",csv_agg,"provider_features.csv","text/csv",key="download_features")
    csv_out=out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions",csv_out,"fraud_predictions.csv","text/csv",key="download_predictions")

else:
    st.info("Upload either a merged dataset or the 4 datasets, plus a pretrained model, then click 'Run Inference'.")
