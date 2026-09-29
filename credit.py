"""
Credit Card Fraud Detection
Simplified version - works on Python 3.14
Uses Logistic Regression + XGBoost only (no Random Forest)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_data(path='creditcard.csv'):
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['Class'].value_counts()}")
    print(f"\nFraud percentage: {df['Class'].mean() * 100:.4f}%")
    return df


def perform_eda(df):
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['Class'].value_counts().plot(kind='bar', ax=axes[0],
                                     color=['steelblue', 'red'])
    axes[0].set_title('Class Distribution (0=Normal, 1=Fraud)')
    axes[0].set_xticklabels(['Normal', 'Fraud'], rotation=0)

    axes[1].hist(df['Amount'], bins=50, color='green', alpha=0.7)
    axes[1].set_title('Transaction Amount Distribution')
    axes[1].set_xlabel('Amount')
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=100)
    plt.close()

    corr = df.corr()['Class'].sort_values(ascending=False)
    print(f"\nTop correlations with fraud:\n{corr.head(10)}")


def preprocess(df):
    print("\n" + "=" * 60)
    print("STEP 3: Preprocessing")
    print("=" * 60)

    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    df = df[['scaled_time', 'scaled_amount'] +
            [c for c in df.columns if c not in ['scaled_time', 'scaled_amount', 'Class']] +
            ['Class']]

    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"Features shape: {X.shape}")
    return X, y


def split_and_balance(X, y):
    print("\n" + "=" * 60)
    print("STEP 4: Train-Test Split + SMOTE")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Frauds before SMOTE: {y_train.sum()}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Frauds after SMOTE: {y_train_res.sum()}")
    print(f"Balanced shape: {X_train_res.shape}")
    return X_train_res, X_test, y_train_res, y_test


def train_models(X_train, y_train):
    print("\n" + "=" * 60)
    print("STEP 5: Training Models")
    print("=" * 60)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=1,
            tree_method='hist'
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[OK] {name} trained")
    return trained


def evaluate_models(models, X_test, y_test):
    print("\n" + "=" * 60)
    print("STEP 6: Model Evaluation")
    print("=" * 60)

    results = {}
    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'ROC-AUC': roc
        }

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Normal', 'Fraud']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Fraud Detection')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('roc_curves.png', dpi=100)
    plt.close()

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    results_df.to_csv('model_comparison.csv')

    best = results_df['F1'].idxmax()
    print(f"\nBest Model (by F1): {best}")
    return results_df


def feature_importance(model, feature_names, top_n=15):
    print("\n" + "=" * 60)
    print("STEP 7: Feature Importance")
    print("=" * 60)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices], color='teal')
    plt.xticks(range(top_n),
               [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    plt.close()

    print(f"\nTop {top_n} features:")
    for i in indices:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")


def main():
    df = load_data('creditcard.csv')
    perform_eda(df)

    X, y = preprocess(df.copy())
    X_train, X_test, y_train, y_test = split_and_balance(X, y)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

    feature_importance(models['XGBoost'], X.columns.tolist())

    import joblib
    joblib.dump(models['XGBoost'], 'fraud_model.pkl')
    print("\n[OK] Model saved as fraud_model.pkl")


if __name__ == '__main__':
    main()