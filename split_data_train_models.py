import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# 1. LOAD GERMAN CREDIT DATASET
df = pd.read_csv(r'C:\Users\irene\OneDrive\Υπολογιστής\TuE\BEP\Code\Bachelor-End-Project-Code---XAI\data\german.csv')

target_col = "Creditability"

X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()





# 2. SPLIT INTO 60 / 20 / 20
# First hold out the final 20% test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split remaining 80% into:
# 60% initial train and 20% additional set
# 20 is 25% of 80, so test_size=0.25 here
X_train_initial, X_additional, y_train_initial, y_additional = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
print(f"Initial train set: {X_train_initial.shape[0]} rows")
print(f"Additional set:    {X_additional.shape[0]} rows")
print(f"Test set:          {X_test.shape[0]} rows")


# ============================================================
# 3. PREPROCESSING
# ============================================================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# 4. XGBOOST MODEL FUNCTION

def build_xgb_pipeline():
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb)
    ])

    return model


# 5. EVALUATION FUNCTION

def evaluate_model(model, X_eval, y_eval, model_name="Model"):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_eval, y_pred),
        "f1": f1_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred),
        "recall": recall_score(y_eval, y_pred),
        "roc_auc": roc_auc_score(y_eval, y_prob)
    }

    print(f"\n================ {model_name} ================")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"F1-score : {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_eval, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred, digits=4))

    return results, y_pred, y_prob



# 6. MODEL 1: TRAIN ON INITIAL 60%

model_1 = build_xgb_pipeline()
model_1.fit(X_train_initial, y_train_initial)

results_1, y_pred_1, y_prob_1 = evaluate_model(
    model_1,
    X_test,
    y_test,
    model_name="Model 1 (trained on 60%)"
)


# 7. MODEL 2: RETRAIN ON 60% + 20% = 80%
X_train_retrained = pd.concat([X_train_initial, X_additional], axis=0)
y_train_retrained = pd.concat([y_train_initial, y_additional], axis=0)

model_2 = build_xgb_pipeline()
model_2.fit(X_train_retrained, y_train_retrained)

results_2, y_pred_2, y_prob_2 = evaluate_model(
    model_2,
    X_test,
    y_test,
    model_name="Model 2 (retrained on 80%)"
)


# 8. COMPARE THE TWO MODELS

comparison_df = pd.DataFrame({
    "true_label": y_test.values,
    "pred_model_1": y_pred_1,
    "pred_model_2": y_pred_2,
    "prob_model_1": y_prob_1,
    "prob_model_2": y_prob_2
}, index=X_test.index)

comparison_df["prediction_changed"] = (
    comparison_df["pred_model_1"] != comparison_df["pred_model_2"]
)

comparison_df["abs_probability_shift"] = (
    comparison_df["prob_model_2"] - comparison_df["prob_model_1"]
).abs()

print('model comparison')
print(f"Prediction change rate: {comparison_df['prediction_changed'].mean():.4f}")
print(f"Average probability shift: {comparison_df['abs_probability_shift'].mean():.4f}")

print("\nExamples where prediction changed:")
print(comparison_df[comparison_df["prediction_changed"]].head())


# 9. SUMMARY TABLE
summary_table = pd.DataFrame([results_1, results_2])
print("summary table")
print(summary_table)

#
# # 10. OPTIONAL: SAVE OUTPUTS
# summary_table.to_csv("model_performance_summary.csv", index=False)
# comparison_df.to_csv("model_prediction_comparison.csv", index=True)
#
# print("\nSaved:")
# print("- model_performance_summary.csv")
# print("- model_prediction_comparison.csv")