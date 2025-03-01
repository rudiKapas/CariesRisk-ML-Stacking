import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from joblib import Parallel, delayed
from sklearn.preprocessing import label_binarize

# File paths
train_file_path = r'C:\Users\user\Documents\MachineLearning\Pandas\data\vectorizeData.xlsx'

# Load training data
print("Loading training data...")
train_data = pd.read_excel(train_file_path)
print("Data loaded successfully.")

# Separate dependent and independent variables for training data
y_train = train_data.iloc[:, 0] - 1  # Original labels (1=Low, 2=Moderate, 3=High)
X_train = train_data.iloc[:, 1:]  # Predictive features

# Binarize labels for multi-class ROC calculation
y_train_bin = label_binarize(y_train, classes=[1, 2, 3])

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Select k-fold values from 2 to 21
k_values = list(range(2, 22))

def evaluate_model(k_folds):
    print(f"Starting training with {k_folds}-fold cross-validation...")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    start_time = time.time()

    # Ensure there are at least two classes in y_train
    if len(np.unique(y_train)) < 2:
        print(f"Skipping {k_folds}-fold training due to insufficient class labels.")
        return []

    # Run GridSearchCV
    model = XGBClassifier()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    # Store results for all models trained in GridSearchCV
    results = []
    total_models = len(grid_search.cv_results_['params'])
    for i in range(total_models):
        progress = ((i + 1) / total_models) * 100
        print(f"{i + 1}/{total_models} models trained; {progress:.2f}% completed")

        results.append({
            'k-fold': k_folds,
            'Params': grid_search.cv_results_['params'][i],
            'Accuracy': grid_search.cv_results_['mean_test_score'][i],
            'F1-score': grid_search.cv_results_['mean_train_score'][i],
            'ROC-AUC': grid_search.cv_results_['mean_test_score'][i],
            'Sensitivity': np.nan,  # Placeholder (calculated separately)
            'Specificity': np.nan,  # Placeholder (calculated separately)
            'Training Time': time.time() - start_time
        })
    return results

# Run evaluations in parallel for different k-folds
print("Starting parallel model evaluations...")
all_results = Parallel(n_jobs=-1)(delayed(evaluate_model)(k) for k in k_values)
print("All evaluations completed.")

# Flatten results (since we now return multiple models per k-fold)
flattened_results = [model for kfold_results in all_results for model in kfold_results]

# Convert results to DataFrame and save
results_df = pd.DataFrame(flattened_results)
results_df.to_csv('xgb_results.csv', index=False)
print("Training completed. Results saved to xgb_results.csv")

# Plot and save ROC curve for the highest accuracy model
best_model = max(flattened_results, key=lambda x: x['Accuracy'])
plt.figure(figsize=(8, 6))
plt.plot(best_model['ROC Curve'][0], best_model['ROC Curve'][1], label=f'Macro AUC = {best_model["ROC-AUC"]:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for Best XGB Model')
plt.legend()
plt.grid(True)
plt.savefig('roc_auc_curve_xgb.png')
plt.show()
print("ROC-AUC Curve saved as roc_auc_curve_xgb.png")
