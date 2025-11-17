import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("Folders created for plots and models.\n")

print("Loading data...")
df = pd.read_table("NSDUH_2023_Tab.txt", delimiter="\t", low_memory=False)
print(f"Data loaded. Total rows: {len(df)}\n")

print("Processing target variable...")
df["IRPYUD5ALC"] = df["IRPYUD5ALC"].replace({94: np.nan, 97: np.nan, 98: np.nan, 99: np.nan})
df = df.dropna(subset=["IRPYUD5ALC"])
df["IRPYUD5ALC"] = df["IRPYUD5ALC"].astype(int)
y = df["IRPYUD5ALC"]
X = df.drop(columns=["IRPYUD5ALC"])
print(f"Target processed. Remaining rows: {len(df)}\n")

print("Dropping flagged/unwanted features...")
leakage_keywords = [
    "ALC", "ALCUSE", "ALCUS", "ALDAY", "ALDAYS", "ALCPROD", "ALCDRINK", "ALCEVER", "ALCUS30D",
    "ALCFREQ", "ALCBEER", "ALCWINE", "ALCSPIR", "ALCFLG", "ALCOHOL", "ALCDEP", "ALCSEV", "ALCABUSE",
    "ALCABDAY", "ALCANY", "ALCAGE", "ALCINT", "ALCBNG", "BNGALC", "BINGE", "HVYDRK", "UD5ILALANY",
    "UD5ALNILANY", "UD5ILNALANY", "UD5ILAALANY", "UD5ILLANY",
    "UDALTIMEGET", "UDALLESSEFF", "UDALNOTSTOP", "UDALBLCKCTD", "UDALFMLYCTD", "UDALMNTLCTD",
    "DRUG", "DRUGUSE", "DRG", "COC", "COCAINE", "HER", "HEROIN", "MJ", "MARIJ", "CANN", "HASH",
    "OPI", "OPIOID", "PAINREL", "STIM", "STIMULANT", "TRQ", "TRANQ", "SED", "SEDATIVE",
    "INHAL", "LSD", "HALLUC", "PSYCH", "METH", "CRK", "CRACK", "KRAT", "KRATOM", "CBD", "VAP",
    "CIG", "TOB", "NIC", "SMOKE", "CHEW", "SNUFF", "PIPE", "CGR", "PNR", "NMU", "NMANY",
    "ANYREC", "ANYYR", "ANYUSE", "RECFLAG", "RECENT", "USE30D", "USEYR", "USEMON", "USEDAY",
    "SUT", "TX", "TREAT", "SERV", "MHTRT", "MHTR", "RECOV", "HELP", "SUPPORT", "COUNS",
    "REHAB", "ADDICT", "SUD", "AUD", "DRINKTX", "DETOX", "ABSTAIN", "QUIT",
    "IRPYUD", "IRANY", "IRMJ", "IRALC", "IRDRUG", "IRREC", "IRDEP", "IRSU", "IRPNR", "IRCOC",
    "YMDESUD", "YMDEAUD", "YMDEIMAD", "YMDEIMAD5YR", "YMIMR", "YMDE", "YMIM", "YMANY",
    "CDUFLAG", "HRFQFLG", "HPALC", "CABUY", "FUCD", "STMN", "COLD", "UAD", "CDNOCGMO", "CDCGMO",
    "SUB", "DETox", "AL", "MRJ", "BNG", "BING", "CAB", "STM", "CAD", "SVY", "CAFRESP2",
    "PIPFLAG", "TQSDANYFLG", "CASURCVR", "CASUPROB2",
    "SUNTPTHNK", "SUNTCOST", "SOLVENEVER", "IRCOSUITRYYR", "SUNTNOFND", "RXTRAMMIS",
    "KIDNYDSAG", "SUNTCONSQ", "IMFNEDFLAG", "PRPROUD2", "OTCFLAG", "IRMHTERED", "CACERVIX",
    "LANGVER", "INHDYPMO", "SVBZONMYR", "SUNTINSCV", "GHBMON", "CIRROSAGE", "SUNTTIME",
    "RXBARBMIS", "YCOUNMDE", "RXLORAMIS", "RXHYDMMIS", "COPDAGE", "MILSTAT", "DRVINMARJ", "BLNTREC", "MRBSTWAY", "MJ", "BLNTAGE", "MKMFREEYR",
    "IRIMPREMEM", "IRIMPSOC", "IRIMPRESP", "IIIMPCONCN", "IIIMPREMEM", "IIIMPSOC",
    "IIIMPGOUT", "IIIMPGOUTM", "IIIMPSOCM", "IIIMPWORK", "IIIMPHHLDM", "IIIMPHHLD",
    "IIIMPRESPM", "IIIMPPEOP", "IIIMPPEOPM", "KSSLR6MAX", "ILLFLAG", "ILIMFOTHFG",
    "DMTAMTFXY", "PCP", "KETMINESK", "PEYOTE", "PSILCY", "MESC", "MKMGREWYR",
    "MKMTRADEYR", "MKMBGHTYR", "MKMOTHWAYYR", "DRVINMARJ", "MKMAMTPDCOM2"
]

substance_keywords = [
    "dab", "drp", "jnt", "pill", "edble", "lt", "oth", "disp", "rec", "smk", "mkmbght"
]

all_keywords = leakage_keywords + substance_keywords

cols_to_remove = [
    c for c in X.columns
    if any(k.lower() in c.lower() for k in all_keywords)
]

X = X.drop(columns=cols_to_remove)
print(f"Dropped {len(cols_to_remove)} features. Remaining features: {X.shape[1]}\n")

print("Encoding categorical variables and filling missing values...")
cat_cols = X.select_dtypes(include="object").columns
X[cat_cols] = X[cat_cols].fillna('Unknown')
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

num_cols = X.select_dtypes(include=np.number).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
print(f"Encoding complete. Total features: {X.shape[1]}\n")

print("Adding limited polynomial interaction terms...")
top_num_cols = X[num_cols].var().sort_values(ascending=False).head(10).index
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X[top_num_cols]), columns=poly.get_feature_names_out(top_num_cols))
X = pd.concat([X, X_poly], axis=1)
print(f"Polynomial interactions added. Total features now: {X.shape[1]}\n")

print("Adding K-Means cluster labels...")
scaler_k = StandardScaler()
X_scaled_for_kmeans = scaler_k.fit_transform(X[top_num_cols])
kmeans = KMeans(n_clusters=5, random_state=42, verbose=0)
X['cluster_label'] = kmeans.fit_predict(X_scaled_for_kmeans)
print("K-Means cluster labels added.\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}\n")

print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE, training set size: {len(y_train_res)}\n")

print("Scaling features...")
scaler_final = StandardScaler()
X_train_res_scaled = scaler_final.fit_transform(X_train_res)
X_test_scaled = scaler_final.transform(X_test)
print("Feature scaling complete.\n")

print("Computing class weights...")
classes = np.unique(y_train_res)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights computed.\n")

print("Selecting top 3 features using univariate importance...")
coefs_placeholder = np.abs(np.corrcoef(X_train_res_scaled, y_train_res, rowvar=False)[-1, :-1])
top3_indices = np.argsort(coefs_placeholder)[-60:]
X_train_final = X_train_res_scaled[:, top3_indices]
X_test_final = X_test_scaled[:, top3_indices]
top3_features = [X.columns[i] for i in top3_indices]
print(f"Top 3 features selected: {top3_features}\n")

print("Training ElasticNet Logistic Regression on top 3 features...")
model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    max_iter=5000,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_final, y_train_res)
print("Training complete.\n")

print("Evaluating model...")
y_pred = model.predict(X_test_final)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nFeatures used in the model and their importance (absolute coefficient values):")
coefs = model.coef_[0]
feature_importance = sorted(zip(top3_features, coefs), key=lambda x: abs(x[1]), reverse=True)

for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")

print("Saving model and scaler...")
joblib.dump(model, "models/AUD_ElasticNet_top3_model.pkl")
joblib.dump(scaler_final, "models/AUD_ElasticNet_top3_scaler.pkl")
print(f"Model and scaler saved. Training complete on top 3 features.")
