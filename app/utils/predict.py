import numpy as np
import pandas as pd
import joblib
import os
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)


class CandidateModelPipeline:
    def __init__(self):
        self.models = [
            self._wrap_model(RandomForestClassifier(n_estimators=100, class_weight='balanced'), "RandomForest"),
            self._wrap_model(xgb.XGBClassifier(objective='binary:logistic', n_estimators=200, learning_rate=0.05, eval_metric='logloss'), "XGBoost"),
            self._wrap_model(LogisticRegression(max_iter=1000, class_weight='balanced'), "LogisticRegression"),
        ]

    def _wrap_model(self, model, name):
        return {"model": model, "name": name}

    def _prepare_ml_data(self, df, target_column, sequence_length):
        data = df.drop(columns=[target_column]).values
        target = df[target_column].values
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(target[i + sequence_length])
        X = np.array(X).reshape(len(X), -1)
        y = np.array(y)
        return X, y

    def _remove_high_correlation(self, df, threshold=0.9):
        corr = df.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return df.drop(columns=to_drop)

    def _plot_roc_curve(self, y_true, y_scores, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {float(roc_auc):.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def _plot_precision_at_k(self, ranked_candidates, max_k=1000):
        precisions = []
        ks = range(1, max_k + 1)
        aprovados = ranked_candidates['approved'].values
        for k in ks:
            precisions.append(np.sum(aprovados[:k]) / k)
        plt.figure(figsize=(10, 5))
        plt.plot(ks, precisions, label='Precision@K', color='blue')
        plt.axhline(y=ranked_candidates['approved'].mean(), color='gray', linestyle='--', label='Base approval rate')
        plt.xlabel('Top K ranked candidates')
        plt.ylabel('Precision@K')
        plt.title('Ranking Effectiveness')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self, df, target_column, models_dir="models", sequence_length=10, plot_metrics=False):
        logging.info("Iniciando pipeline...")

        df = self._remove_high_correlation(df)
        X, y = self._prepare_ml_data(df, target_column, sequence_length)

        df = df.iloc[-X.shape[0]:].copy()
        X = pd.DataFrame(X, index=df.index)
        y = pd.Series(y, index=df.index)

        X['original_index'] = df.index
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        X_train_index = X_train['original_index']
        X_test_index = X_test['original_index']
        X_train = X_train.drop(columns=['original_index'])
        X_test = X_test.drop(columns=['original_index'])

        best_model = None
        best_score = 0
        best_ranked = None

        for entry in self.models:
            model = entry["model"]
            name = entry["name"]

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            score = roc_auc_score(y_test, y_pred_proba)

            logging.info(f"{name} ROC-AUC: {score:.4f}")
            if plot_metrics:
                self._plot_roc_curve(y_test, y_pred_proba, name)

            ranked_candidates = pd.DataFrame({
                'candidate_id': X_test_index,
                'approval_probability': y_pred_proba,
                'approved': y_test.values
            }).sort_values(by='approval_probability', ascending=False)

            if plot_metrics:
                self._plot_precision_at_k(ranked_candidates)

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
                best_ranked = ranked_candidates

        if best_model:
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(best_model, os.path.join(models_dir, f"{best_model_name}.joblib"))
            logging.info(f"Modelo salvo como {best_model_name}.joblib")

        return best_model, best_ranked
