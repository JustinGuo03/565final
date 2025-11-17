import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
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
    "SUNTPTHNK", "SUNTCOST", "SOLVENEVER", "IRCOSUITRYYR", "SUNTNOFND", "RXTRAMMIS"
]

substance_keywords = [
    "alc", "drink", "beer", "wine", "liquor", "depnd", "depend", "use", "abuse",
    "drug", "sub", "sud", "addict", "rehab", "tx", "treat",
    "nic", "smok", "ftnd", "ndss", "tob", "cig", "pipe", "chew",
    "mj", "marj", "mr", "blnt", "blunt", "cann", "hash",
    "air", "duster", "inhal", "solv", "halluc", "psyched", "lsd", "ket", "pcp", "pey", "mesc"
]

all_keywords = leakage_keywords + substance_keywords
cols_to_remove = [c for c in X.columns if any(k.lower() in c.lower() for k in all_keywords)]
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
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42, verbose=0)
X['cluster_label'] = kmeans.fit_predict(X_scaled_for_kmeans)
print("K-Means cluster labels added.\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}\n")

print("=" * 80)
print("INTERPRETABLE DECISION TREE FEATURE SELECTION")
print("=" * 80)

# Stage 1: Mutual Information Filter
print("\n[STAGE 1] Computing Mutual Information scores...")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_threshold = np.percentile(mi_scores, 60)
mi_mask = mi_scores > mi_threshold
X_train_mi = X_train.loc[:, mi_mask]
X_test_mi = X_test.loc[:, mi_mask]
print(f"Mutual Information filter: {X_train_mi.shape[1]} features retained (from {X_train.shape[1]})")

# Stage 2: RFE with Decision Tree
print("\n[STAGE 2] Applying Recursive Feature Elimination (RFE) with Decision Tree...")
rfe_estimator = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
n_features_to_select = min(30, X_train_mi.shape[1])
rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=5)
rfe.fit(X_train_mi, y_train)

X_train_final = X_train_mi.loc[:, rfe.support_]
X_test_final = X_test_mi.loc[:, rfe.support_]
final_features = X_train_final.columns.tolist()
print(f"RFE selection: {len(final_features)} features retained")
print(f"\nFinal Feature Set:\n{final_features}")

print("\n" + "=" * 80)
print("TRAINING INTERPRETABLE DECISION TREE")
print("=" * 80)

final_tree = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42
)
final_tree.fit(X_train_final, y_train)
print(f"Tree depth: {final_tree.get_depth()}")
print(f"Number of leaves: {final_tree.get_n_leaves()}")

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

y_pred = final_tree.predict(X_test_final)
y_pred_proba = final_tree.predict_proba(X_test_final)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

try:
    if len(np.unique(y_test)) > 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"ROC AUC Score: {roc_auc:.4f}")
except:
    print("ROC AUC Score: Not computed")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nConfusion matrix saved to plots/confusion_matrix.png")

plt.figure(figsize=(25, 15))
plot_tree(
    final_tree,
    feature_names=final_features,
    class_names=[str(c) for c in final_tree.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree for AUD Prediction', fontsize=16)
plt.tight_layout()
plt.savefig('plots/decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("Decision tree visualization saved to plots/decision_tree.png")

tree_rules = export_text(final_tree, feature_names=final_features)
print("\nDecision Tree Rules:")
print(tree_rules)

with open('models/decision_tree_rules.txt', 'w') as f:
    f.write(tree_rules)
print("\nTree rules saved to models/decision_tree_rules.txt")

joblib.dump(final_tree, "models/AUD_DecisionTree_interpretable.pkl")
joblib.dump(final_features, "models/final_features.pkl")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nFinal model uses {len(final_features)} features")
print(f"Tree has {final_tree.get_n_leaves()} leaf nodes and depth {final_tree.get_depth()}")
print("\nSaved artifacts:")
print(" - models/AUD_DecisionTree_interpretable.pkl")
print(" - models/final_features.pkl")
print(" - models/decision_tree_rules.txt")
print(" - plots/confusion_matrix.png")
print(" - plots/decision_tree.png")
