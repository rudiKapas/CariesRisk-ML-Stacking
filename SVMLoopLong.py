import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from itertools import product
from joblib import Parallel, delayed
from sklearn.preprocessing import label_binarize

# File paths
train_file_path = r'C:\Users\user\Documents\MachineLearning\Pandas\data\vectorizeData.xlsx'

# Load training data
train_data = pd.read_excel(train_file_path)

# Separate dependent and independent variables for training data
y_train = train_data.iloc[:, 0]  # Original labels (1=Low, 2=Moderate, 3=High)
X_train = train_data.iloc[:, 1:]  # Predictive features

# Binarize labels for multi-class ROC calculation
y_train_bin = label_binarize(y_train, classes=[1, 2, 3])

# Define parameter grid
C_list = [0.1, 1, 10, 100]
kernel_list = ['linear', 'rbf', 'poly']
gamma_list = [0.01, 0.1, 1, 'scale', 'auto']
k_values = list(range(2, 22))  # k-fold values from 2 to 21


# Function to evaluate a single model configuration
def evaluate_model(k_folds, C, kernel, gamma):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies, f1_scores, roc_aucs, sensitivities, specificities = [], [], [], [], []
    best_roc_curve = None
    start_time = time.time()

    for train_index, test_index in kfold.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        y_test_bin_fold = y_train_bin[test_index]

        # Train Support Vector Machine model
        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        model.fit(X_train_fold, y_train_fold)

        # Predictions
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)

        # Metrics
        acc = accuracy_score(y_test_fold, y_pred)
        f1 = f1_score(y_test_fold, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test_bin_fold, y_proba, multi_class='ovr')

        cm = confusion_matrix(y_test_fold, y_pred, labels=[1, 2, 3])
        specificity_list, sensitivity_list = [], []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)

        accuracies.append(acc)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        sensitivities.append(np.nanmean(sensitivity_list))
        specificities.append(np.nanmean(specificity_list))

        # Compute ROC curve for macro-averaged AUC visualization
        fpr, tpr, _ = roc_curve(y_test_bin_fold.ravel(), y_proba.ravel())
        best_roc_curve = (fpr, tpr, roc_auc)

    if accuracies:
        return {
            'k-fold': k_folds,
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'Accuracy': np.mean(accuracies),
            'F1-score': np.mean(f1_scores),
            'ROC-AUC': np.mean(roc_aucs),
            'Sensitivity': np.nanmean(sensitivities),
            'Specificity': np.nanmean(specificities),
            'Training Time': time.time() - start_time,
            'ROC Curve': best_roc_curve
        }
    return None


# Run evaluations in parallel
results = Parallel(n_jobs=-1)(delayed(evaluate_model)(k_folds, C, kernel, gamma)
                              for k_folds, C, kernel, gamma
                              in product(k_values, C_list, kernel_list, gamma_list))

# Filter out None values
results = [res for res in results if res is not None]

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results).drop(columns=['ROC Curve'])  # Remove ROC curve data before saving
results_df.to_csv('svm_results.csv', index=False)
print("Training completed. Results saved to svm_results.csv")

# Plot and save ROC curve for the highest accuracy model
best_model = max(results, key=lambda x: x['Accuracy'])
plt.figure(figsize=(8, 6))
plt.plot(best_model['ROC Curve'][0], best_model['ROC Curve'][1], label=f'Macro AUC = {best_model["ROC-AUC"]:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for Best SVM Model')
plt.legend()
plt.grid(True)
plt.savefig('roc_auc_curve_svm.png')
plt.show()
print("ROC-AUC Curve saved as roc_auc_curve_svm.png")
