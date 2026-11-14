import io
import re
import sys
import zipfile
import subprocess
import urllib.request
import numpy as np
import pandas as pd

try:
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openpyxl"])


def is_yes(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip().str.lower().isin(["1", "1.0", "y", "yes", "true"])


# ==============================================================================
# PART 1: UCI Z-ALIZADEH SANI (303 patients, 127 women)
# ==============================================================================
zip_url = "https://archive.ics.uci.edu/static/public/412/z+alizadeh+sani.zip"
req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
try:
    zip_bytes = urllib.request.urlopen(req, timeout=30).read()
except Exception:
    page_html = urllib.request.urlopen(
        urllib.request.Request(
            "https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani",
            headers={"User-Agent": "Mozilla/5.0"},
        ),
        timeout=30,
    ).read().decode("utf-8", errors="ignore")
    match = re.search(r'https://archive\.ics\.uci\.edu/static/public/412/[^"\']+\.zip', page_html)
    zip_bytes = urllib.request.urlopen(
        urllib.request.Request(match.group(0), headers={"User-Agent": "Mozilla/5.0"}),
        timeout=30,
    ).read()

with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    data_files = [
        f for f in zf.namelist()
        if f.lower().endswith((".xlsx", ".xls", ".csv")) and not f.startswith("__MACOSX")
    ]
    with zf.open(data_files[0]) as f:
        raw_sani = pd.read_csv(f) if data_files[0].lower().endswith(".csv") else pd.read_excel(f)

raw_sani.columns = [str(c).strip().lower().replace(" ", "") for c in raw_sani.columns]

df_sani = pd.DataFrame({
    "age": pd.to_numeric(raw_sani["age"], errors="coerce"),
    "sex": raw_sani["sex"].astype(str).str.strip().str.lower().map({"male": 1, "fmale": 0, "female": 0}),
    "cp": np.select(
        [
            is_yes(raw_sani["typicalchestpain"]),
            is_yes(raw_sani["atypical"]),
            is_yes(raw_sani["nonanginal"]),
        ],
        [1, 2, 3],
        default=4,
    ),
    "trestbps": pd.to_numeric(raw_sani["bp"], errors="coerce"),
    "chol": np.nan,  # Left blank/None to test native missing-value handling
    "fbs": (pd.to_numeric(raw_sani["fbs"], errors="coerce") > 120).astype(int),
    "restecg": np.select(
        [
            is_yes(raw_sani["lvh"]),
            is_yes(raw_sani["stdepression"]) | is_yes(raw_sani["tinversion"]),
        ],
        [2, 1],
        default=0,
    ),
    "target": (raw_sani["cath"].astype(str).str.strip().str.lower() == "cad").astype(int),
})


# ==============================================================================
# PART 2: FRAMINGHAM HEART STUDY (4,240 patients, ~2,420 women)
# ==============================================================================
framingham_urls = [
    "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
    "https://raw.githubusercontent.com/TarekDib03/Analytics/master/Week3%20-%20Logistic%20Regression/Data/framingham.csv",
    "https://raw.githubusercontent.com/sharmaroshan/Heart-Disease-Prediction/master/framingham.csv",
    "https://raw.githubusercontent.com/kelvin-kinyua/Framingham-Heart-Disease-Prediction/main/framingham.csv",
]

raw_fram = None
for url in framingham_urls:
    try:
        csv_text = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20
        ).read().decode("utf-8")
        candidate = pd.read_csv(io.StringIO(csv_text))
        if {"male", "age", "sysBP", "totChol", "TenYearCHD"}.issubset(candidate.columns):
            raw_fram = candidate
            break
    except Exception:
        continue

if raw_fram is None:
    raise RuntimeError("Could not download Framingham dataset from public mirrors.")

gluc = pd.to_numeric(raw_fram["glucose"], errors="coerce")
df_fram = pd.DataFrame({
    "age": pd.to_numeric(raw_fram["age"], errors="coerce"),
    "sex": pd.to_numeric(raw_fram["male"], errors="coerce"),
    "cp": np.nan,  # Not recorded in Framingham
    "trestbps": pd.to_numeric(raw_fram["sysBP"], errors="coerce"),
    "chol": pd.to_numeric(raw_fram["totChol"], errors="coerce"),
    "fbs": np.where(gluc.isna(), np.nan, (gluc > 120).astype(float)),
    "restecg": np.nan,  # Not recorded in Framingham
    "target": pd.to_numeric(raw_fram["TenYearCHD"], errors="coerce"),
}).dropna(subset=["target"]).reset_index(drop=True)
df_fram["target"] = df_fram["target"].astype(int)


# ==============================================================================
# PART 3: SAVE & DOWNLOAD BOTH CLEANED CSVs
# ==============================================================================
df_sani.to_csv("z_alizadeh_sani_clean.csv", index=False)
df_fram.to_csv("framingham_clean.csv", index=False)

for name, df in [("z_alizadeh_sani_clean.csv", df_sani), ("framingham_clean.csv", df_fram)]:
    print(f"SUCCESS -> {name}: {len(df)} rows | Women={(df['sex'] == 0).sum()} | Men={(df['sex'] == 1).sum()}")

try:
    from google.colab import files
    files.download("z_alizadeh_sani_clean.csv")
    files.download("framingham_clean.csv")
except ImportError:
    pass
