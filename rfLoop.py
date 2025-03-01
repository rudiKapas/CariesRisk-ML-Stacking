import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from joblib import Parallel, delayed
from sklearn.preprocessing import label_binarize

# File paths
train_file_path = r'C:\Users\user\Documents\MachineLearning\Pandas\data\vectorizeData.xlsx'

# Load training data
print("Loading training data...")
train_data = pd.read_excel(train_file_path)
print("Data loaded successfully.")

# Separate dependent and independent variables
y_train = train_data.iloc[:, 0]  # Original labels (1=Low, 2=Moderate, 3=High)
X_train = train_data.iloc[:, 1:]  # Predictive features

# Binarize labels for multi-class ROC calculation
classes = [1, 2, 3]
y_train_bin = label_binarize(y_train, classes=classes)
n_classes = y_train_bin.shape[1]

# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'bootstrap': [True, False]
}

# Select k-fold values from 2 to 21
k_values = list(range(2, 22))

def evaluate_model(k_folds):
    print(f"Starting training with {k_folds}-fold cross-validation...")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    start_time = time.time()

    # Run GridSearchCV using RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    # Get the best estimator from grid search
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_train)
    y_proba = best_estimator.predict_proba(X_train)

    # Compute confusion matrix and derive Sensitivity & Specificity
    cm = confusion_matrix(y_train, y_pred, labels=classes)
    sensitivity_list, specificity_list = [], []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    # ---------- Compute Macro-Average ROC Curve ----------
    # 1) Compute ROC for each class (one-vs-rest)
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_proba[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        roc_auc_dict[i] = auc(fpr, tpr)

    # 2) Aggregate all unique FPR points and average the TPR
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes

    # 3) Macro-average ROC curve and its AUC
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    # Store results for each candidate hyperparameter combination from GridSearchCV
    results = []
    total_models = len(grid_search.cv_results_['params'])
    for i in range(total_models):
        progress = ((i + 1) / total_models) * 100
        print(f"{i + 1}/{total_models} candidate models evaluated; {progress:.2f}% completed")
        results.append({
            'k-fold': k_folds,
            'Params': grid_search.cv_results_['params'][i],
            'Accuracy': grid_search.cv_results_['mean_test_score'][i],
            'F1-score': grid_search.cv_results_['mean_train_score'][i],
            'ROC-AUC': roc_auc_macro,
            'Sensitivity': np.nanmean(sensitivity_list),
            'Specificity': np.nanmean(specificity_list),
            'Training Time': time.time() - start_time,
            'ROC Curve': (fpr_macro, tpr_macro)
        })

    return results

# Run evaluations in parallel for different k-fold values
print("Starting parallel model evaluations...")
all_results = Parallel(n_jobs=-1)(delayed(evaluate_model)(k) for k in k_values)
print("All evaluations completed.")

# Flatten results (since each k-fold evaluation returns multiple candidate results)
flattened_results = [res for kfold_res in all_results for res in kfold_res]

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(flattened_results)
results_df.to_csv('random_forest_results.csv', index=False)
print("Training completed. Results saved to random_forest_results.csv")

# Plot and save ROC curve for the candidate with the highest Accuracy
best_candidate = max(flattened_results, key=lambda x: x['Accuracy'])
fpr_best, tpr_best = best_candidate['ROC Curve']
macro_auc = best_candidate['ROC-AUC']

plt.figure(figsize=(8, 6))
plt.plot(fpr_best, tpr_best, label=f'Macro AUC = {macro_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for Best Random Forest Model (Macro-Average)')
plt.legend()
plt.grid(True)
plt.savefig('roc_auc_curve_random_forest.png')
plt.show()
print("ROC-AUC Curve saved as roc_auc_curve_random_forest.png")
