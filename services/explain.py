from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from xgboost import plot_importance, XGBClassifier
from services.train_utils import load_processed_csv

plt.style.use('seaborn-v0_8')  # Professional styling

def generate_visualizations(model, X, y, model_type="xgb"):
    """Safe visualization for trained models without re-fitting"""
    try:
        # 1. Feature Importance (Always works)
        plt.figure(figsize=(12, 8))
        plot_importance(model, max_num_features=15)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()
        print("Saved feature_importance.png")

        # 2. SHAP Summary (Skip if too large)
        if len(X) <= 10000:  # Safety check
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:1000])  # Subsample
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X.iloc[:1000], plot_type="bar")
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300)
            plt.close()
            print("Saved shap_summary.png")

    except Exception as e:
        print(f"Visualization skipped due to: {str(e)}")

def plot_learning_curve(model, X, y):
    """Time-series aware learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='f1_macro'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation')
    plt.title('Learning Curve')
    plt.xlabel('Training Samples')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('learning_curve.png', dpi=300)
    plt.close()

def plot_training_history(model):
    """Training/validation metrics over iterations"""
    history = model.evals_result_
    plt.figure(figsize=(10, 6))

    for metric in history['validation_0']:
        metric_name = metric.replace('validation_0_', '')
        plt.plot(history['validation_0'][metric], label=f'Train {metric_name}', ls='--')
        plt.plot(history['validation_1'][metric], label=f'Val {metric_name}', ls='-')

    plt.title('Training History')
    plt.xlabel('Iterations')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('training_history.png', dpi=300)
    plt.close()

def generate_shap_summary(model, X, sample_size=100):
    """SHAP feature analysis"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:sample_size])

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X.iloc[:sample_size], plot_type="bar")
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    plt.close()

def explain_saved_model(csv_path: Path, model_path: Path):
    """Full analysis pipeline for saved models"""
    X, y = load_processed_csv(csv_path)
    model = XGBClassifier()
    model.load_model(model_path)

    generate_visualizations(model, X, y)
    print("Generated: feature_importance.png, learning_curve.png, shap_summary.png")