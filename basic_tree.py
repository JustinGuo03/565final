import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
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
    "PIPFLAG", "TQSDANYFLG", "CASURCVR", "CASUPROB2",
    "SUNTPTHNK", "SUNTCOST", "SOLVENEVER", "IRCOSUITRYYR", "SUNTNOFND", "RXTRAMMIS",
    "KIDNYDSAG", "SUNTCONSQ", "IMFNEDFLAG", "PRPROUD2", "OTCFLAG", "IRMHTERED", "CACERVIX",
    "LANGVER", "INHDYPMO", "SVBZONMYR", "SUNTINSCV", "GHBMON", "CIRROSAGE", "SUNTTIME",
    "RXBARBMIS", "YCOUNMDE", "RXLORAMIS", "RXHYDMMIS", "COPDAGE", "MILSTAT", "DRVINMARJ",
    "BLNTREC", "MRBSTWAY", "MJ", "BLNTAGE", "MKMFREEYR",
    "IRIMPREMEM", "IRIMPSOC", "IRIMPRESP", "IIIMPCONCN", "IIIMPREMEM", "IIIMPSOC",
    "IIIMPGOUT", "IIIMPGOUTM", "IIIMPSOCM", "IIIMPWORK", "IIIMPHHLDM", "IIIMPHHLD",
    "IIIMPRESPM", "IIIMPPEOP", "IIIMPPEOPM", "KSSLR6MAX", "ILLFLAG", "ILIMFOTHFG",
    "DMTAMTFXY", "PCP", "KETMINESK", "PEYOTE", "PSILCY", "MESC", "MKMGREWYR",
    "MKMTRADEYR", "MKMBGHTYR", "MKMOTHWAYYR", "DRVINMARJ", "MKMAMTPDCOM2",
    "MKMLOO", # marijuana / drug consumption
    "PYUD5INH", # inhalants
    "SMKY", # smoke / tobacco
    "INHLYR", # inhalant / smoke-less tobacco
    "UDCC", # substance consequences / problems
    "UDHASTR", # substance urges / cravings
    "UDPRWD", # substance withdrawal / runny
    "IICDU", # first drug use age
    "IICD2YFU", # last year drug follow-up
    "LGAS" , "IRDSTNGD30", "NOBOOKY2", "IRDSTRST12", "IRDSTNGD12", "DSTNGD30", "DSTCHR12", "IRDSTCHR30", "IRDSTHOP30", "DSTNRV12", "IRDSTNRV12", "IRDSTHOP12", "DSTEFF12", "DSTCHR30", "IIDSTCHR12", "DSTHOP12", "NOMARR2", "SPPAINT", "CG30EST", "DSTNGD12", "AKSSLR6WRST", "IIDSTRST12", "IIDSTNGD12", "DSTRST12",
    "CLEFLU", "PREGAGE2", "IRIMPHHLD", "IRIMPWORK", "IRDSTWORST", "IRDSTRST30",
    "CI30EST", "SMKAGLAST", "SMIPPPY", "SMKMOLAST", "GAS", "GLUE", "FELTMARKR", "AMYLNIT", "NITOXID", "SMMIPY", "RXHYDCANY", "INHLAGLST", "AMIPY", "RXBZOTANY", "INHLMOLST", "LMMIPY", "MICATPY", "DSTHOP30", "ADDPR2WK", "KSSLR6MON", "BOOKED", "IRDSTEFF30", "ADSUITPAYR"
    # possibly substance-related
]

substance_keywords = [
    "alc", "drink", "beer", "wine", "liquor", "depnd", "depend", "use", "abuse",
    "drug", "sub", "sud", "addict", "rehab", "tx", "treat",
    "nic", "smok", "ftnd", "ndss", "tob", "cig", "pipe", "chew",
    "mj", "marj", "mr", "blnt", "blunt", "cann", "hash",
    "air", "duster", "inhal", "solv", "halluc", "psyched", "lsd", "ket", "pcp", "pey", "mesc",
    "cagv", "cafre", "cagvwho", "alcuse", "alcflg",
    "casu", "casuprob", "ircd", "iralc", "irrec", "irpnr", "irany", "irdrug", "mkmbght", # catches MKMBGHT30N2, MKMBGHTDISP, MKMBGHTDRPS
    "mkmloogm", # catches MKMLOOGMCOM2
    "oth", # catches OTHAEROS
    "smklss"
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

print("=" * 80)
print("MULTI-STAGE FEATURE SELECTION PIPELINE")
print("=" * 80)

# Stage 1: Mutual Information Filter
print("\n[STAGE 1] Computing Mutual Information scores...")
mi_scores = mutual_info_classif(X_train_res, y_train_res, random_state=42)
mi_threshold = np.percentile(mi_scores, 60) # Keep top 40%
mi_mask = mi_scores > mi_threshold
X_train_mi = X_train_res.loc[:, mi_mask]
X_test_mi = X_test.loc[:, mi_mask]
print(f"Mutual Information filter: {X_train_mi.shape[1]} features retained (from {X_train_res.shape[1]})")

# Stage 2: Random Forest Feature Importance
print("\n[STAGE 2] Training Random Forest for feature importance...")
rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_selector.fit(X_train_mi, y_train_res)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X_train_mi.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20).to_string(index=False))

# Select features using importance threshold
selector = SelectFromModel(rf_selector, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train_mi)
X_test_selected = selector.transform(X_test_mi)
selected_features = X_train_mi.columns[selector.get_support()].tolist()
print(f"\nRandom Forest selection: {len(selected_features)} features retained")

# Stage 3: Recursive Feature Elimination with Cross-Validation
print("\n[STAGE 3] Applying Recursive Feature Elimination (RFE)...")
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

n_features_to_select = min(30, len(selected_features))
rfe_estimator = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=5)
rfe.fit(X_train_selected_df, y_train_res)

X_train_final = X_train_selected_df.loc[:, rfe.support_]
X_test_final = X_test_selected_df.loc[:, rfe.support_]
final_features = X_train_final.columns.tolist()
print(f"RFE selection: {len(final_features)} features retained")
print(f"\nFinal Feature Set:\n{final_features}")

# Save feature importance plot
plt.figure(figsize=(12, 8))
top_features = feature_importances.head(30)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 30 Features by Random Forest Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFeature importance plot saved to plots/feature_importance.png")

print("\n" + "=" * 80)
print("TRAINING INTERPRETABLE DECISION TREE")
print("=" * 80)

# Train final interpretable decision tree with cost-complexity pruning
print("\nFinding optimal pruning parameter (ccp_alpha)...")
base_tree = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42
)
base_tree.fit(X_train_final, y_train_res)

# Get pruning path
path = base_tree.cost_complexity_pruning_path(X_train_final, y_train_res)
ccp_alphas = path.ccp_alphas[:-1] # Exclude the last alpha (fully pruned)
impurities = path.impurities[:-1]

# Test different alpha values with cross-validation
print("Testing pruning levels with cross-validation...")
cv_scores = []
for ccp_alpha in ccp_alphas[::5]: # Test every 5th alpha for speed
    tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',
        ccp_alpha=ccp_alpha,
        random_state=42
    )
    scores = cross_val_score(tree, X_train_final, y_train_res, cv=5, scoring='accuracy', n_jobs=-1)
    cv_scores.append(scores.mean())

# Find best alpha
best_idx = np.argmax(cv_scores)
best_alpha = ccp_alphas[::5][best_idx]
print(f"Best ccp_alpha: {best_alpha:.6f} (CV accuracy: {cv_scores[best_idx]:.4f})")

# Train final pruned tree
print("\nTraining final pruned decision tree...")
final_tree = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    ccp_alpha=best_alpha,
    random_state=42
)
final_tree.fit(X_train_final, y_train_res)
print(f"Tree depth: {final_tree.get_depth()}")
print(f"Number of leaves: {final_tree.get_n_leaves()}")

# Evaluate on test set
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

y_pred = final_tree.predict(X_test_final)
y_pred_proba = final_tree.predict_proba(X_test_final)

print("\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Multi-class ROC AUC (if more than 2 classes)
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

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Pruned Decision Tree')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nConfusion matrix saved to plots/confusion_matrix.png")

# Visualize tree
print("\nSaving decision tree visualization...")
plt.figure(figsize=(25, 15))
plot_tree(
    final_tree,
    feature_names=final_features,
    class_names=[str(c) for c in final_tree.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Pruned Decision Tree for AUD Prediction', fontsize=16)
plt.tight_layout()
plt.savefig('plots/decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("Decision tree visualization saved to plots/decision_tree.png")

# Export tree rules as text
print("\nDecision Tree Rules:")
tree_rules = export_text(final_tree, feature_names=final_features)
print(tree_rules)

# Save tree rules to file
with open('models/decision_tree_rules.txt', 'w') as f:
    f.write(tree_rules)
print("\nTree rules saved to models/decision_tree_rules.txt")

# Save models
print("\nSaving models and artifacts...")
joblib.dump(final_tree, "models/AUD_DecisionTree_pruned.pkl")
joblib.dump(rf_selector, "models/AUD_RandomForest_selector.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")
joblib.dump(final_features, "models/final_features.pkl")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nFinal model uses {len(final_features)} features")
print(f"Tree has {final_tree.get_n_leaves()} leaf nodes and depth {final_tree.get_depth()}")
print("\nSaved artifacts:")
print(" - models/AUD_DecisionTree_pruned.pkl")
print(" - models/AUD_RandomForest_selector.pkl")
print(" - models/selected_features.pkl")
print(" - models/final_features.pkl")
print(" - models/decision_tree_rules.txt")
print(" - plots/feature_importance.png")
print(" - plots/confusion_matrix.png")
print(" - plots/decision_tree.png")
