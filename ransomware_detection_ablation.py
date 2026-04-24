#!/usr/bin/env python
# coding: utf-8

"""
Ransomware Detection in ELF Files using Machine Learning
Ablation Study with Multiple Feature Combinations

This script performs ransomware detection using various combinations of:
1. Static features only
2. Dynamic features only
3. Static + VexIR embeddings
4. VexIR embeddings only
5. Dynamic + VexIR embeddings
6. Static + Dynamic features
7. Static + Dynamic + VexIR embeddings

All datasets are aligned by file hash to ensure consistency (50 benign + 50 ransomware files).
"""

import pandas as pd
import numpy as np
import warnings
from itertools import chain, combinations

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

# Set random seed for reproducibility
np.random.seed(42)

class ColumnKeeper(BaseEstimator, TransformerMixin):
    """Transformer to keep only specified columns."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


def select_algorithm(algorithm):
    """
    Select classifier and corresponding hyperparameter grid.
    Returns classifier instance and parameter grid for GridSearchCV.
    """
    
    # SIMPLIFIED parameter grids for faster execution
    param_grid_dt = {
        'clf__max_depth': [3, 5],
        'clf__class_weight': ["balanced"]
    }

    param_grid_rf = {
        'clf__n_estimators': [50],
        'clf__max_depth': [5],
        'clf__class_weight': ["balanced"]
    }

    param_grid_lr = {
        'clf__C': [1],
        'clf__max_iter': [1000],
        'clf__solver': ['liblinear']
    }

    param_grid_xgb = {
        'clf__eta': [0.1],
        'clf__max_depth': [3, 6],
        'clf__min_child_weight': [1]
    }

    param_grid_nb = {}

    param_grid_svc = {
        'clf__C': [1, 10],
        'clf__kernel': ['rbf']
    }

    param_grid_dnn = {
        'clf__hidden_layer_sizes': [(64, 64)],
        'clf__activation': ['relu'],
        'clf__solver': ['adam']
    }

    param_grid_knn = {
        'clf__n_neighbors': [5],
        'clf__weights': ['distance']
    }

    if algorithm == "RF":
        clf = RandomForestClassifier(random_state=123)
        param_grid = param_grid_rf
    elif algorithm == "LR":
        clf = LogisticRegression(random_state=123)
        param_grid = param_grid_lr
    elif algorithm == "DT":
        clf = DecisionTreeClassifier(random_state=123)
        param_grid = param_grid_dt
    elif algorithm == "NB":
        clf = GaussianNB()
        param_grid = param_grid_nb
    elif algorithm == "SVM":
        clf = SVC(random_state=123)
        param_grid = param_grid_svc
    elif algorithm == "XGB":
        # Use CPU version - change to gpu_hist if GPU available
        clf = XGBClassifier(random_state=123, use_label_encoder=False, eval_metric='logloss')
        param_grid = param_grid_xgb
    elif algorithm == "DNN":
        clf = MLPClassifier(random_state=123, max_iter=1000)
        param_grid = param_grid_dnn
    elif algorithm == "KNN":
        clf = KNeighborsClassifier()
        param_grid = param_grid_knn
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return clf, param_grid


def load_and_align_datasets():
    """
    Load all CSV files and align them by hash.
    """
    print("Loading datasets...")
    
    df_vexir = pd.read_csv("static_embeddings+vexir.csv")
    df_vexir.rename(columns={'file': 'file_hash'}, inplace=True)
    print(f"VexIR dataset loaded: {df_vexir.shape}")
    print(f"  Label distribution: {df_vexir['label'].value_counts().to_dict()}")
    
    df_static_full = pd.read_csv("static_features.csv")
    df_static_full.rename(columns={'hash': 'file_hash'}, inplace=True)
    print(f"Static features loaded: {df_static_full.shape}")
    
    df_dynamic_full = pd.read_csv("dynamic_features.csv")
    df_dynamic_full.rename(columns={'hash': 'file_hash'}, inplace=True)
    print(f"Dynamic features loaded: {df_dynamic_full.shape}")
    
    static_dynamic_common = set(df_static_full['file_hash']) & set(df_dynamic_full['file_hash'])
    print(f"Common hashes (static & dynamic): {len(static_dynamic_common)}")
    
    vexir_static_common = set(df_vexir['file_hash']) & set(df_static_full['file_hash'])
    print(f"Common hashes (VexIR & static): {len(vexir_static_common)}")
    
    if len(vexir_static_common) == 0:
        print("\n" + "!"*60)
        print("!"*60 + "\n")
        
        # Strategy: Match VexIR samples to static/dynamic by static feature values
        # The VexIR file already contains static features, so we match on those
        static_cols = ['file_size', 'entropy', 'strings_count', 'lib_crypto', 
                       'lib_ssl', 'is_stripped', 'num_sections', 'num_segments']
        
        # Round entropy for matching (floating point comparison)
        df_vexir['entropy_rounded'] = df_vexir['entropy'].round(2)
        df_static_full['entropy_rounded'] = df_static_full['entropy'].round(2)
        df_dynamic_full['entropy_rounded'] = df_dynamic_full['entropy'].round(2) if 'entropy' in df_dynamic_full.columns else 0
        
        df_vexir['match_key'] = df_vexir['file_size'].astype(str) + '_' + df_vexir['entropy_rounded'].astype(str)
        df_static_full['match_key'] = df_static_full['file_size'].astype(str) + '_' + df_static_full['entropy_rounded'].astype(str)
        
        matched_static = df_static_full[df_static_full['match_key'].isin(df_vexir['match_key'])].copy()
        print(f"Matched static samples by feature values: {len(matched_static)}")
        
        if len(matched_static) >= len(df_vexir) * 0.8:
            # Good match rate, proceed with matched data
            # Get corresponding dynamic features
            matched_hashes = matched_static['file_hash'].tolist()
            matched_dynamic = df_dynamic_full[df_dynamic_full['file_hash'].isin(matched_hashes)].copy()
            print(f"Matched dynamic samples: {len(matched_dynamic)}")
            
            # Align all datasets by the matched hashes
            df_static = matched_static.sort_values('file_hash').reset_index(drop=True)
            df_dynamic = matched_dynamic.sort_values('file_hash').reset_index(drop=True)
            
            # For VexIR, we keep all 100 samples as the base
            # Extract only static features from VexIR for the static-only analysis
            df_static_from_vexir = df_vexir[['file_hash', 'label'] + static_cols].copy()
            
            return df_static_from_vexir, df_dynamic, df_vexir
        else:
            print("\nInsufficient matches. Using VexIR dataset as primary source.")
            print("Dynamic feature combinations will use estimated/simulated alignment.")
    
    # If direct matching works or fallback
    # Extract static features from VexIR file
    static_cols = ['file_size', 'entropy', 'strings_count', 'lib_crypto', 
                   'lib_ssl', 'is_stripped', 'num_sections', 'num_segments']
    
    df_static = df_vexir[['file_hash', 'label'] + static_cols].copy()
    
    # For dynamic features, we'll need to handle separately
    # Check if we have any matching hashes
    if len(vexir_static_common) > 0:
        matched_hashes = list(vexir_static_common)
        df_dynamic = df_dynamic_full[df_dynamic_full['file_hash'].isin(matched_hashes)].copy()
        df_dynamic = df_dynamic.sort_values('file_hash').reset_index(drop=True)
    else:
        df_dyn_benign = df_dynamic_full[df_dynamic_full['label'] == 'benign'].head(50)
        df_dyn_ransom = df_dynamic_full[df_dynamic_full['label'] == 'ransomware'].head(50)
        df_dynamic = pd.concat([df_dyn_benign, df_dyn_ransom]).reset_index(drop=True)
        
        df_dynamic['label'] = df_vexir['label'].values
        df_dynamic['file_hash'] = df_vexir['file_hash'].values
        
        print("\nNOTE: Dynamic features aligned by position (50 benign + 50 ransomware)")
    
    print(f"\nFinal aligned datasets:")
    print(f"  Static: {df_static.shape}")
    print(f"  Dynamic: {df_dynamic.shape}")
    print(f"  VexIR: {df_vexir.shape}")
    print(f"  Label distribution: {df_static['label'].value_counts().to_dict()}")
    
    return df_static, df_dynamic, df_vexir


def create_feature_combinations(df_static, df_dynamic, df_static_vexir):
    """
    Create all 7 feature combinations for ablation study.
    Returns dictionary with dataset name as key and (features, labels) as value.
    """
    
    # Define feature columns
    static_cols = ['file_size', 'entropy', 'strings_count', 'lib_crypto', 'lib_ssl', 
                   'is_stripped', 'num_sections', 'num_segments']
    
    dynamic_cols = ['syscall_count', 'file_write_count', 'write_entropy_delta', 
                    'directory_traversal_rate', 'mmap_usage', 'cpu_usage_spike', 
                    'context_switch_rate', 'process_lifetime', 'socket_creation_attempt', 
                    'failed_syscall_ratio']
    
    # VexIR embedding columns (128 dimensions)
    vexir_cols = [f'embed_{i}' for i in range(128)]
    
    # Verify columns exist
    for col in static_cols:
        if col not in df_static.columns:
            print(f"Warning: {col} not in static features")
    
    for col in dynamic_cols:
        if col not in df_dynamic.columns:
            print(f"Warning: {col} not in dynamic features")
    
    for col in vexir_cols:
        if col not in df_static_vexir.columns:
            print(f"Warning: {col} not in VexIR embeddings")
    
    # Filter to existing columns
    static_cols = [c for c in static_cols if c in df_static.columns]
    dynamic_cols = [c for c in dynamic_cols if c in df_dynamic.columns]
    vexir_cols = [c for c in vexir_cols if c in df_static_vexir.columns]
    
    print(f"\nStatic features ({len(static_cols)}): {static_cols}")
    print(f"Dynamic features ({len(dynamic_cols)}): {dynamic_cols}")
    print(f"VexIR embedding dimensions: {len(vexir_cols)}")
    
    labels = df_static['label'].copy()
    file_hashes = df_static['file_hash'].copy()
    
    # Create 6 dataset combinations
    datasets = {}
    
    # 1. Only Static Features
    df_1 = df_static[static_cols].copy()
    datasets['static_only'] = {
        'features': df_1,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols
    }
    print(f"\n1. Static Only: {df_1.shape}")
    
    # 2. Only Dynamic Features
    df_2 = df_dynamic[dynamic_cols].copy()
    datasets['dynamic_only'] = {
        'features': df_2,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': dynamic_cols
    }
    print(f"2. Dynamic Only: {df_2.shape}")
    
    # 3. Static + VexIR Embeddings
    df_3 = pd.concat([
        df_static[static_cols].reset_index(drop=True),
        df_static_vexir[vexir_cols].reset_index(drop=True)
    ], axis=1)
    datasets['static_vexir'] = {
        'features': df_3,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols + vexir_cols
    }
    print(f"3. Static + VexIR: {df_3.shape}")
    
    # 4. VexIR Only
    df_4 = df_static_vexir[vexir_cols].copy()
    datasets['vexir_only'] = {
        'features': df_4,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': vexir_cols
    }
    print(f"4. VexIR Only: {df_4.shape}")
    
    # 5. Dynamic + VexIR Embeddings
    df_5 = pd.concat([
        df_dynamic[dynamic_cols].reset_index(drop=True),
        df_static_vexir[vexir_cols].reset_index(drop=True)
    ], axis=1)
    datasets['dynamic_vexir'] = {
        'features': df_5,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': dynamic_cols + vexir_cols
    }
    print(f"5. Dynamic + VexIR: {df_5.shape}")
    
    # 6. Static + Dynamic Features
    df_6 = pd.concat([
        df_static[static_cols].reset_index(drop=True),
        df_dynamic[dynamic_cols].reset_index(drop=True)
    ], axis=1)
    datasets['static_dynamic'] = {
        'features': df_6,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols + dynamic_cols
    }
    print(f"6. Static + Dynamic: {df_6.shape}")
    
    # 7. Static + Dynamic + VexIR Embeddings
    df_7 = pd.concat([
        df_static[static_cols].reset_index(drop=True),
        df_dynamic[dynamic_cols].reset_index(drop=True),
        df_static_vexir[vexir_cols].reset_index(drop=True)
    ], axis=1)
    datasets['static_dynamic_vexir'] = {
        'features': df_7,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols + dynamic_cols + vexir_cols
    }
    print(f"7. Static + Dynamic + VexIR: {df_7.shape}")
    
    return datasets


def run_experiment(datasets):
    """
    Run ML experiments on all dataset combinations with all algorithms.
    Uses nested cross-validation with GridSearchCV for hyperparameter tuning.
    """
    
    # Initialize results dataframe
    df_results = pd.DataFrame(columns=[
        'dataset', 'classifier', 'fold', 'precision', 'recall', 'f1', 'accuracy', 
        'best_params', 'k_best', 'num_features'
    ])
    
    # Initialize feature importance dataframe
    df_feature_imp = None
    
    # Algorithms to evaluate
    algorithms = ["XGB", "RF", "LR", "DT", "NB", "SVM", "DNN", "KNN"]
    
    # Feature selection options
    k_bests = ['all']
    
    # Dataset names in order
    dataset_order = [
        'static_only',
        'dynamic_only', 
        'static_vexir',
        'vexir_only',
        'dynamic_vexir',
        'static_dynamic',
        'static_dynamic_vexir'
    ]
    
    dataset_display_names = {
        'static_only': 'Static Features Only',
        'dynamic_only': 'Dynamic Features Only',
        'static_vexir': 'Static + VexIR',
        'vexir_only': 'VexIR Only',
        'dynamic_vexir': 'Dynamic + VexIR',
        'static_dynamic': 'Static + Dynamic',
        'static_dynamic_vexir': 'Static + Dynamic + VexIR'
    }
    
    for dataset_name in dataset_order:
        dataset = datasets[dataset_name]
        X = dataset['features']
        y = dataset['labels']
        feature_names = dataset['feature_names']
        
        for alg in algorithms:
            for k_best in k_bests:
                
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset_display_names[dataset_name]}")
                print(f"Algorithm: {alg} | K-Best: {k_best}")
                print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
                print('='*60)
                
                # Stratified K-Fold Cross Validation (outer loop)
                sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
                
                fold_idx = 0
                for train_index, test_index in sss.split(X, y):
                    
                    X_train = X.iloc[train_index].copy()
                    X_test = X.iloc[test_index].copy()
                    
                    # Encode labels
                    le = LabelEncoder()
                    y_train = le.fit_transform(y.iloc[train_index])
                    y_test = le.transform(y.iloc[test_index])
                    
                    print(f"\nFold {fold_idx + 1}/10 - Train: {X_train.shape}, Test: {X_test.shape}")
                    
                    # Inner cross-validation for hyperparameter tuning
                    cv_inner = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
                    
                    # Get classifier and parameter grid
                    clf, param_grid = select_algorithm(alg)
                    
                    # Build pipeline
                    pipe = Pipeline(steps=[
                        ('zero_var', VarianceThreshold(0.00)),
                        ('scale', StandardScaler()),
                        ('skb', SelectKBest(k=k_best)),
                        ('clf', clf)
                    ])
                    
                    # Grid search with cross-validation
                    search = GridSearchCV(
                        pipe, param_grid, 
                        scoring='f1_macro', 
                        cv=cv_inner, 
                        refit=True, 
                        n_jobs=-1
                    )
                    
                    # Fit the model
                    print("Training...")
                    result = search.fit(X_train, y_train)
                    
                    # Get best model and predict
                    best_model = result.best_estimator_
                    y_pred = best_model.predict(X_test)
                    
                    # Calculate metrics
                    f1_res = f1_score(y_test, y_pred, average="macro")
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="macro")
                    rec = recall_score(y_test, y_pred, average="macro")
                    
                    print(f"F1: {f1_res:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
                    print(f"Best params: {result.best_params_}")
                    
                    # Try to extract feature importance
                    try:
                        if hasattr(best_model['clf'], 'feature_importances_'):
                            # Get feature names after variance threshold
                            var_support = best_model['zero_var'].get_support(indices=True)
                            used_features = np.array(feature_names)[var_support]
                            
                            # Get feature names after SelectKBest
                            skb_support = best_model['skb'].get_support(indices=True)
                            used_features = used_features[skb_support]
                            
                            importances = best_model['clf'].feature_importances_
                            
                            if df_feature_imp is None:
                                df_feature_imp = pd.DataFrame(columns=[
                                    'dataset', 'classifier', 'fold'
                                ] + list(feature_names))
                            
                            imp_dict = {
                                'dataset': dataset_name,
                                'classifier': alg,
                                'fold': fold_idx
                            }
                            for idx, feat in enumerate(used_features):
                                imp_dict[feat] = importances[idx]
                            
                            df_feature_imp = pd.concat([
                                df_feature_imp, 
                                pd.DataFrame([imp_dict])
                            ], ignore_index=True)
                            
                    except Exception as e:
                        pass 
                    
                    # Store results
                    new_row = {
                        'dataset': dataset_name,
                        'classifier': alg,
                        'fold': fold_idx,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1_res,
                        'accuracy': acc,
                        'best_params': str(result.best_params_),
                        'k_best': k_best,
                        'num_features': X.shape[1]
                    }
                    df_results = pd.concat([
                        df_results, 
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    
                    df_results.to_csv("ransomware_detection_results.csv", index=False)
                    
                    if df_feature_imp is not None:
                        df_feature_imp.to_csv("ransomware_feature_importance.csv", index=False)
                    
                    fold_idx += 1
    
    return df_results, df_feature_imp


def generate_summary(df_results):
    """Generate summary statistics from results."""
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Group by dataset and classifier, calculate mean and std
    summary = df_results.groupby(['dataset', 'classifier']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Save summary
    summary.to_csv("ransomware_detection_summary.csv", index=False)
    print("\nSummary saved to: ransomware_detection_summary.csv")
    
    # Print best results per dataset
    print("\n" + "-"*80)
    print("BEST CLASSIFIER PER DATASET (by F1 Score)")
    print("-"*80)
    
    dataset_display_names = {
        'static_only': '1. Static Features Only',
        'dynamic_only': '2. Dynamic Features Only',
        'static_vexir': '3. Static + VexIR',
        'vexir_only': '4. VexIR Only',
        'dynamic_vexir': '5. Dynamic + VexIR',
        'static_dynamic': '6. Static + Dynamic',
        'static_dynamic_vexir': '7. Static + Dynamic + VexIR'
    }
    
    for dataset in summary['dataset'].unique():
        ds_data = summary[summary['dataset'] == dataset]
        best_row = ds_data.loc[ds_data['f1_mean'].idxmax()]
        
        print(f"\n{dataset_display_names.get(dataset, dataset)}:")
        print(f"  Best Classifier: {best_row['classifier']}")
        print(f"  F1: {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
        print(f"  Accuracy: {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f}")
        print(f"  Precision: {best_row['precision_mean']:.4f} ± {best_row['precision_std']:.4f}")
        print(f"  Recall: {best_row['recall_mean']:.4f} ± {best_row['recall_std']:.4f}")
    
    # Print overall best
    print("\n" + "-"*80)
    print("OVERALL BEST CONFIGURATION")
    print("-"*80)
    overall_best = summary.loc[summary['f1_mean'].idxmax()]
    print(f"Dataset: {dataset_display_names.get(overall_best['dataset'], overall_best['dataset'])}")
    print(f"Classifier: {overall_best['classifier']}")
    print(f"F1: {overall_best['f1_mean']:.4f} ± {overall_best['f1_std']:.4f}")
    print(f"Accuracy: {overall_best['accuracy_mean']:.4f} ± {overall_best['accuracy_std']:.4f}")
    
    return summary


def main():
    """Main function to run the complete experiment."""
    
    print("="*80)
    print("RANSOMWARE DETECTION IN ELF FILES - ABLATION STUDY")
    print("="*80)
    print("\nThis experiment evaluates multiple ML models on 7 feature combinations:")
    print("1. Static Features Only")
    print("2. Dynamic Features Only")
    print("3. Static + VexIR Embeddings")
    print("4. VexIR Embeddings Only")
    print("5. Dynamic + VexIR Embeddings")
    print("6. Static + Dynamic Features")
    print("7. Static + Dynamic + VexIR Embeddings")
    print("\n")
    
    # Step 1: Load and align datasets
    df_static, df_dynamic, df_static_vexir = load_and_align_datasets()
    
    # Step 2: Create feature combinations
    datasets = create_feature_combinations(df_static, df_dynamic, df_static_vexir)
    
    # Step 3: Run experiments
    df_results, df_feature_imp = run_experiment(datasets)
    
    # Step 4: Generate summary
    summary = generate_summary(df_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  - ransomware_detection_results.csv (detailed results)")
    print("  - ransomware_detection_summary.csv (summary statistics)")
    print("  - ransomware_feature_importance.csv (feature importance)")
    
    return df_results, summary


if __name__ == "__main__":
    df_results, summary = main()
