import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
from joblib import Parallel, delayed
from sklearn.preprocessing import label_binarize

from itertools import combinations

# =============================================================================
# 1. Load Data
# =============================================================================
train_file_path = r'C:\Users\user\Documents\MachineLearning\Pandas\data\vectorizeData.xlsx'

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

# =============================================================================
# 2. Define Your 7 Best Models (Fixed Hyperparameters)
# =============================================================================
model_dt = DecisionTreeClassifier(
    criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42
)

model_knn = KNeighborsClassifier(
    n_neighbors=15, weights='distance', metric='manhattan'
)

model_svm = SVC(
    kernel='rbf', gamma='scale', C=100, probability=True, random_state=42
)

model_xgb = XGBClassifier(
    colsample_bytree=0.7, gamma=0, learning_rate=0.1,
    max_depth=5, n_estimators=100, subsample=0.5,
    use_label_encoder=False, eval_metric='mlogloss', random_state=42
)

model_lr = LogisticRegression(
    C=10, l1_ratio=0.5, penalty='elasticnet', solver='saga', max_iter=5000, random_state=42
)

model_mlp = MLPClassifier(
    activation='logistic', alpha=0.01, hidden_layer_sizes=(50, 50),
    learning_rate='adaptive', solver='adam', max_iter=10000, random_state=42
)

model_rf = RandomForestClassifier(
    bootstrap=True, max_depth=20, min_samples_leaf=1,
    min_samples_split=10, n_estimators=200, random_state=42
)

# Put them in a list for easy pairing
models = [
    ("DT", model_dt),
    ("KNN", model_knn),
    ("SVM", model_svm),
    ("XGB", model_xgb),
    ("LR", model_lr),
    ("MLP", model_mlp),
    ("RF", model_rf)
]

# Generate all unique 3-model combinations (triples)
model_triples = list(combinations(models, 3))

# =============================================================================
# 3. Define the Meta-Learner Param Grid for Random Forest Meta-Learner
# =============================================================================
# We use the proven best parameters for the meta-learner:
# {'bootstrap': True, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 1, 'n_estimators': 200}
param_grid = [
    {
        'final_estimator__bootstrap': [True],
        'final_estimator__max_depth': [20],
        'final_estimator__min_samples_split': [10],
        'final_estimator__min_samples_leaf': [1],
        'final_estimator__n_estimators': [200]
    }
]

# =============================================================================
# 4. Function to Evaluate a Single Triple with a Given k-Fold
# =============================================================================
def evaluate_stacking(triple, k_folds):
    """
    Trains a stacking ensemble for the three models in `triple` using GridSearchCV,
    with cross-validation = k_folds. Returns a list of dictionaries with results.
    """
    model_name1, model1 = triple[0]
    model_name2, model2 = triple[1]
    model_name3, model3 = triple[2]

    print(f"\nStarting stacking for triple ({model_name1}, {model_name2}, {model_name3}) with {k_folds}-fold cross-validation...")
    start_time = time.time()

    # Define KFold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Create the stacking classifier with a RandomForest meta-learner (with fixed parameters)
    stacking_clf = StackingClassifier(
        estimators=[(model_name1, model1), (model_name2, model2), (model_name3, model3)],
        final_estimator=RandomForestClassifier(
            bootstrap=True, max_depth=20, min_samples_leaf=1,
            min_samples_split=10, n_estimators=200, random_state=42
        ),
        passthrough=False,
        n_jobs=-1
    )

    # Run GridSearchCV on the stacking classifier
    grid_search = GridSearchCV(
        estimator=stacking_clf,
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1,
        verbose=0,  # set to 2 for more detailed output if desired
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
    fpr_dict, tpr_dict = {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_proba[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes

    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    # Store results for each parameter combination in grid_search (here only one candidate per run)
    results = []
    total_models = len(grid_search.cv_results_['params'])
    training_time = time.time() - start_time

    for i in range(total_models):
        progress = ((i + 1) / total_models) * 100
        print(f"{i+1}/{total_models} candidate models evaluated; {progress:.2f}% completed")
        results.append({
            'Triple': f"{model_name1}+{model_name2}+{model_name3}",
            'k-fold': k_folds,
            'Params': grid_search.cv_results_['params'][i],
            'Accuracy': grid_search.cv_results_['mean_test_score'][i],
            'F1-score': grid_search.cv_results_['mean_train_score'][i],
            'ROC-AUC': roc_auc_macro,
            'Sensitivity': np.nanmean(sensitivity_list),
            'Specificity': np.nanmean(specificity_list),
            'Training Time': training_time,
            'ROC Curve': (fpr_macro, tpr_macro)
        })

    return results

# =============================================================================
# 5. Main Loop: Evaluate All Triple + k-fold Combinations in Parallel
# =============================================================================
k_values = list(range(2, 22))  # k-fold values from 2 to 21
total_runs = len(model_triples) * len(k_values)
print("Starting parallel stacking evaluations for triples...")

# We'll keep track of overall progress
current_run = 0

def run_triple_kfold(triple, k):
    global current_run
    current_run += 1
    overall_progress = (current_run / total_runs) * 100
    print(f"{current_run}/{total_runs} models trained; {overall_progress:.2f}% completed")
    return evaluate_stacking(triple, k)

# Run everything in parallel
all_results_nested = Parallel(n_jobs=-1)(
    delayed(run_triple_kfold)(triple, k) for triple in model_triples for k in k_values
)
print("All evaluations completed for triples.")

# Flatten results (each k-fold evaluation returns a list of candidate model results)
flattened_results = [res for triple_res in all_results_nested for res in triple_res]

# Convert results to a DataFrame and save as CSV
results_df = pd.DataFrame(flattened_results)
results_df.to_csv('stacking_three_models_results.csv', index=False)
print("Training completed. Results saved to stacking_three_models_results.csv")

# =============================================================================
# 6. Find Best Candidate by Accuracy and Plot ROC Curve
# =============================================================================
best_candidate = max(flattened_results, key=lambda x: x['Accuracy'])
fpr_best, tpr_best = best_candidate['ROC Curve']
macro_auc = best_candidate['ROC-AUC']

plt.figure(figsize=(8, 6))
plt.plot(fpr_best, tpr_best, label=f'Macro AUC = {macro_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for Best Three-Model Stacking Ensemble (Macro-Average)')
plt.legend()
plt.grid(True)
plt.savefig('roc_auc_curve_stacking_three_models.png')
plt.show()
print("ROC-AUC Curve saved as roc_auc_curve_stacking_three_models.png")
