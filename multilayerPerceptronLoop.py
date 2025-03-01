import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler
from joblib import Parallel, delayed

# Import BayesSearchCV and Categorical from scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Categorical


# =============================================================================
# Custom wrapper for MLPClassifier
# =============================================================================
class MLPClassifierWrapper(MLPClassifier):
    def set_params(self, **params):
        if 'hidden_layer_sizes' in params:
            hls = params['hidden_layer_sizes']
            if isinstance(hls, str):
                # Convert string "50" or "50,50" or "100,50" to a tuple of ints.
                hls = tuple(int(x.strip()) for x in hls.split(','))
                params['hidden_layer_sizes'] = hls
        return super().set_params(**params)


# =============================================================================
# Data loading and preprocessing
# =============================================================================
# File paths
train_file_path = r'C:\Users\user\Documents\MachineLearning\Pandas\data\vectorizeData.xlsx'

print("Loading training data...")
train_data = pd.read_excel(train_file_path)
print("Data loaded successfully.")

# Separate dependent and independent variables
y_train = train_data.iloc[:, 0]  # Original labels (1=Low, 2=Moderate, 3=High)
X_train = train_data.iloc[:, 1:]  # Predictive features

# Scale features to help convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Binarize labels for multi-class ROC calculation
classes = [1, 2, 3]
y_train_bin = label_binarize(y_train, classes=classes)
n_classes = y_train_bin.shape[1]

# =============================================================================
# Define hyperparameter search space
# =============================================================================
# For hidden_layer_sizes, we use strings so that BayesSearchCV can handle them.
param_space = {
    'hidden_layer_sizes': Categorical(["50", "100", "50,50", "100,50"]),
    'activation': Categorical(['relu', 'tanh', 'logistic']),
    'solver': Categorical(['adam', 'sgd', 'lbfgs']),
    'alpha': Categorical([0.0001, 0.001, 0.01]),
    'learning_rate': Categorical(['constant', 'adaptive'])
}

# Total number of iterations for the Bayesian optimization run
n_iter = 216

# Define the k-fold cross-validation values (from 2 to 21)
k_values = list(range(2, 22))


# =============================================================================
# Evaluation function using BayesSearchCV
# =============================================================================
def evaluate_model(k_folds):
    print(f"\nStarting training with {k_folds}-fold cross-validation...")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    start_time = time.time()

    # Use our custom MLPClassifierWrapper as the estimator.
    bayes_search = BayesSearchCV(
        estimator=MLPClassifierWrapper(max_iter=10000, random_state=42),
        search_spaces=param_space,
        cv=kfold,
        n_jobs=-1,
        n_iter=n_iter,
        verbose=0,
        return_train_score=True
    )

    bayes_search.fit(X_train, y_train)
    print(f"BayesSearchCV progress: 100% ({n_iter}/{n_iter})")

    # Get the best estimator from the search
    best_estimator = bayes_search.best_estimator_
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
    # Compute ROC curve for each class (one-vs-rest)
    fpr_dict, tpr_dict = {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_train_bin[:, i], y_proba[:, i])

    # Aggregate all unique FPR points
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))

    # Interpolate each TPR and average for the macro-average ROC curve
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes

    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    # Store results for each candidate hyperparameter configuration
    results = []
    total_models = len(bayes_search.cv_results_['params'])
    for i in range(total_models):
        progress = ((i + 1) / total_models) * 100
        print(f"{i + 1}/{total_models} candidate models evaluated; {progress:.2f}% completed")
        results.append({
            'k-fold': k_folds,
            'Params': bayes_search.cv_results_['params'][i],
            'Accuracy': bayes_search.cv_results_['mean_test_score'][i],
            'F1-score': bayes_search.cv_results_['mean_train_score'][i],
            'ROC-AUC': roc_auc_macro,
            'Sensitivity': np.nanmean(sensitivity_list),
            'Specificity': np.nanmean(specificity_list),
            'Training Time': time.time() - start_time,
            'ROC Curve': (fpr_macro, tpr_macro)
        })

    return results


# =============================================================================
# Run evaluations in parallel for different k-fold values
# =============================================================================
print("Starting parallel model evaluations using Bayesian Optimization...")
all_results = Parallel(n_jobs=-1)(
    delayed(evaluate_model)(k) for k in k_values
)
print("All evaluations completed.")

# Flatten results (since each k-fold evaluation returns multiple candidate results)
flattened_results = [res for kfold_res in all_results for res in kfold_res]

# Save results to CSV
results_df = pd.DataFrame(flattened_results)
results_df.to_csv('mlp_results_bayes.csv', index=False)
print("Training completed. Results saved to mlp_results_bayes.csv")

# Plot and save ROC curve for the candidate with the highest Accuracy
best_candidate = max(flattened_results, key=lambda x: x['Accuracy'])
fpr_best, tpr_best = best_candidate['ROC Curve']
macro_auc = best_candidate['ROC-AUC']

plt.figure(figsize=(8, 6))
plt.plot(fpr_best, tpr_best, label=f'Macro AUC = {macro_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for Best MLP Model (Macro-Average)')
plt.legend()
plt.grid(True)
plt.savefig('roc_auc_curve_mlp_bayes.png')
plt.show()
print("ROC-AUC Curve saved as roc_auc_curve_mlp_bayes.png")
