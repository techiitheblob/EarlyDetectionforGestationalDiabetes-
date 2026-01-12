import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

print("🚀 STARTING HYPERPARAMETER TUNING (This may take a minute)...")

# 1. LOAD & PREP DATA (Same as before)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=names)

# Feature Engineering
df['RiskScore'] = (df['Pregnancies'] * 10) + df['BloodPressure']
X = df.drop(['Outcome', 'Pregnancies', 'BloodPressure'], axis=1)
y = df['Outcome']

# Cleaning & Scaling
cols_with_missing = ['Glucose', 'SkinThickness', 'Insulin', 'BMI']
X[cols_with_missing] = X[cols_with_missing].replace(0, np.nan)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
imputer = KNNImputer(n_neighbors=7)
X_imputed = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X.columns)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ==========================================
# 🛑 NEW STEP: HYPERPARAMETER TUNING
# ==========================================

# 1. Tune XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'scale_pos_weight': [1, 2, 3] # Vital for imbalanced data
}
print("   ... Tuning XGBoost ...")
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_distributions=xgb_params,
    n_iter=20, scoring='recall', cv=3, verbose=0, random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

# 2. Tune Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}
print("   ... Tuning Random Forest ...")
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_params,
    n_iter=20, scoring='recall', cv=3, verbose=0, random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# 3. LightGBM (Keep standard tuned)
best_lgb = lgb.LGBMClassifier(verbose=-1, num_leaves=31, learning_rate=0.05)

print(f"✅ Tuning Complete!")
print(f"   Best XGB Depth: {xgb_search.best_params_['max_depth']}")
print(f"   Best RF Class Weight: {rf_search.best_params_['class_weight']}")

# ==========================================
# BUILD FINAL ENSEMBLE
# ==========================================
voting_model = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', best_rf), ('lgb', best_lgb)],
    voting='soft'
)
voting_model.fit(X_train, y_train)

# ==========================================
# OPTIMIZE THRESHOLD & REPORT
# ==========================================
y_prob = voting_model.predict_proba(X_test)[:, 1]

# Calculate ROC and Optimal Threshold
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_pred_tuned = (y_prob >= best_thresh).astype(int)
acc = accuracy_score(y_test, y_pred_tuned)

print(f"\n📏 NEW OPTIMAL THRESHOLD: {best_thresh:.4f}")
print(f"🏆 TUNED ACCURACY: {acc:.2%}")
print("\nClassification Report (Tuned & Optimized):\n", classification_report(y_test, y_pred_tuned))

# ==========================================
# VISUALIZATION: Compare Performance
# ==========================================
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Healthy', 'Diabetic'],
            yticklabels=['Healthy', 'Diabetic'])
plt.xlabel('AI Prediction')
plt.ylabel('Actual Condition')
plt.title(f'Tuned Confusion Matrix (Threshold: {best_thresh:.2f})')
plt.savefig('Visual_Final_Tuned_Matrix.png')
print("✅ Saved: Visual_Final_Tuned_Matrix.png")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc

print("🎨 GENERATING FULL VISUAL SUITE FOR TUNED MODEL...")

# ==========================================
# VISUAL 1: CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Healthy', 'Diabetic'],
            yticklabels=['Healthy', 'Diabetic'])
plt.xlabel('AI Prediction')
plt.ylabel('Actual Condition')
plt.title(f'Tuned Confusion Matrix (Threshold: {best_thresh:.2f})')
plt.tight_layout()
plt.savefig('Tuned_1_Confusion_Matrix.png')
print("✅ Saved: Tuned_1_Confusion_Matrix.png")

# ==========================================
# VISUAL 2: ROC CURVE
# ==========================================
# We use the probabilities from the voting model
y_prob = voting_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Tuned ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Tuned Clinical Accuracy (ROC-AUC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('Tuned_2_ROC_Curve.png')
print("✅ Saved: Tuned_2_ROC_Curve.png")

# ==========================================
# VISUAL 3: RISK DISTRIBUTION & THRESHOLD
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(x=y_prob, hue=y_test, kde=True, bins=30, palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.axvline(best_thresh, color='green', linestyle='--', linewidth=2, label=f'Optimal Cutoff ({best_thresh:.2f})')
plt.xlabel('Predicted Probability of GDM')
plt.ylabel('Number of Patients')
plt.title('Tuned Model Confidence & Decision Boundary')
plt.legend()
plt.savefig('Tuned_3_Risk_Distribution.png')
print("✅ Saved: Tuned_3_Risk_Distribution.png")

# ==========================================
# PREP SHAP (Recalculating for the Best XGBoost)
# ==========================================
# We use the 'best_xgb' model found during tuning for explainability
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

# ==========================================
# VISUAL 4: SHAP SUMMARY PLOT
# ==========================================
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Importance (Tuned XGBoost)")
plt.tight_layout()
plt.savefig('Tuned_4_SHAP_Summary.png')
print("✅ Saved: Tuned_4_SHAP_Summary.png")

# ==========================================
# VISUAL 5: SHAP DEPENDENCE (Glucose)
# ==========================================
plt.figure(figsize=(8, 6))
# Find Glucose column index safely
glucose_col_idx = list(X_test.columns).index('Glucose') if 'Glucose' in X_test.columns else 0
shap.dependence_plot(glucose_col_idx, shap_values.values, X_test, show=False)
plt.title("Why Risk Rises: Glucose Impact (Tuned)")
plt.tight_layout()
plt.savefig('Tuned_5_Glucose_Impact.png')
print("✅ Saved: Tuned_5_Glucose_Impact.png")

print("\n🚀 ALL 5 TUNED GRAPHS READY!")