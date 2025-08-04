import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def train_and_evaluate_model(df):
    print("\n--- Starting Model Training and Evaluation ---")

    features = [
        'views', 'likes', 'comments_count', 'duration_seconds', 
        'days_since_publish', 'views_per_day', 'title_len', 'desc_len',
        'title_sentiment', 'desc_sentiment', 'mean_toxicity', 'pct_toxic_comments'
    ]
    
    X = df[features]
    y = df['viral_label']

    # Stratify only if each class has at least 2 samples
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify)

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print(" Model training complete.")

    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Viral', 'Viral'], yticklabels=['Not Viral', 'Viral'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Feature importances plot
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title('Feature Importances')
    plt.show()

    # SHAP explainability
    print("\n--- Generating SHAP Explainability Plots ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification shap output shape
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_summary = shap_values[1]
    else:
        shap_summary = shap_values

    # Summary plot
    shap.summary_plot(shap_summary, X_test, plot_type="bar", show=True)

    # Force plot for one viral example if exists
    viral_indices = y_test[y_test == 1].index
    if len(viral_indices) > 0:
        first_viral_idx = viral_indices[0]
        loc = X_test.index.get_loc(first_viral_idx)
        shap.force_plot(
            explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
            shap_summary[loc],
            X_test.iloc[loc],
            matplotlib=True,
            show=True
        )
    else:
        print("No viral examples in test set for force plot.")

    # Save model
    model_save_path = "viral_predictor_model.joblib"
    joblib.dump(model, model_save_path)
    print(f" Model saved to {model_save_path}")

if __name__ == '__main__':
    file_path = "2_preprocessing/clean_data.csv"  # Adjust your path here
    df_cleaned = load_data(file_path)
    if df_cleaned is not None:
        train_and_evaluate_model(df_cleaned)
