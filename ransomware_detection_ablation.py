#!/usr/bin/env python
# coding: utf-8

"""
Ransomware Detection in ELF Files using Machine Learning
Ablation Study with Multiple Feature Combinations and Embedding Dimensions

This script performs ransomware detection using various combinations of:
1. Static features only
2. Dynamic features only
3. VexIR embeddings only
4. Static + VexIR embeddings
5. Dynamic + VexIR embeddings
6. Static + Dynamic features
7. Static + Dynamic + VexIR embeddings

Each combination is evaluated with VexIR embeddings of dimensions: 512, 256, 128, 64, 32, 16, 8

Dataset Alignment:
- All datasets are explicitly aligned by file_hash column
- Hash intersection ensures only common samples are used
- Sorting by hash guarantees consistent row ordering
- Assertions verify alignment before experiments
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def select_algorithm(algorithm):
    """
    Returns classifier and parameter grid for the specified algorithm.
    All classifiers use RANDOM_SEED for reproducibility.
    """
    
    param_grid_dt = {
        'clf__max_depth': [3, 5, 7],
        'clf__class_weight': ["balanced"]
    }

    param_grid_rf = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [5, 10],
        'clf__class_weight': ["balanced"]
    }

    param_grid_lr = {
        'clf__C': [0.1, 1, 10],
        'clf__max_iter': [1000],
        'clf__solver': ['liblinear']
    }

    param_grid_xgb = {
        'clf__eta': [0.1, 0.3],
        'clf__max_depth': [3, 6],
        'clf__min_child_weight': [1, 3]
    }

    param_grid_nb = {}

    param_grid_svc = {
        'clf__C': [1, 10],
        'clf__kernel': ['rbf', 'linear']
    }

    param_grid_dnn = {
        'clf__hidden_layer_sizes': [(64, 64), (128, 64)],
        'clf__activation': ['relu'],
        'clf__solver': ['adam']
    }

    param_grid_knn = {
        'clf__n_neighbors': [3, 5, 7],
        'clf__weights': ['uniform', 'distance']
    }

    if algorithm == "RF":
        clf = RandomForestClassifier(random_state=RANDOM_SEED)
        param_grid = param_grid_rf
    elif algorithm == "LR":
        clf = LogisticRegression(random_state=RANDOM_SEED)
        param_grid = param_grid_lr
    elif algorithm == "DT":
        clf = DecisionTreeClassifier(random_state=RANDOM_SEED)
        param_grid = param_grid_dt
    elif algorithm == "NB":
        clf = GaussianNB()
        param_grid = param_grid_nb
    elif algorithm == "SVM":
        clf = SVC(random_state=RANDOM_SEED)
        param_grid = param_grid_svc
    elif algorithm == "XGB":
        clf = XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
        param_grid = param_grid_xgb
    elif algorithm == "DNN":
        clf = MLPClassifier(random_state=RANDOM_SEED, max_iter=1000)
        param_grid = param_grid_dnn
    elif algorithm == "KNN":
        clf = KNeighborsClassifier()
        param_grid = param_grid_knn
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return clf, param_grid


def load_datasets(vexir_dim=128):
    """
    Load all CSV files for the specified VexIR embedding dimension.
    
    CRITICAL: This function performs proper dataset alignment by file_hash:
    1. Finds intersection of hashes across all three datasets
    2. Filters each dataset to only include common hashes
    3. Sorts all datasets by hash to ensure consistent row ordering
    4. Verifies alignment with assertions
    
    Args:
        vexir_dim: Dimension of VexIR embeddings (512, 256, 128, 64, 32, 16, or 8)
    
    Returns:
        df_static: Static features DataFrame (aligned)
        df_dynamic: Dynamic features DataFrame (aligned)
        df_vexir: VexIR embeddings DataFrame (aligned)
    """
    print(f"\nLoading datasets (VexIR dim={vexir_dim})...")
    
    df_static = pd.read_csv(f"static_features_{vexir_dim}.csv")
    print(f"  Static features loaded: {df_static.shape}")
    
    df_dynamic = pd.read_csv(f"dynamic_features_{vexir_dim}.csv")
    print(f"  Dynamic features loaded: {df_dynamic.shape}")
    
    df_vexir = pd.read_csv(f"vexir_embeddings_{vexir_dim}.csv")
    print(f"  VexIR embeddings loaded: {df_vexir.shape}")
    
    # === DATASET ALIGNMENT BY HASH ===
    print("\n  Aligning datasets by file_hash...")
    
    # Step 1: Find common hashes across all datasets
    common_hashes = (
        set(df_static['file_hash'])
        & set(df_dynamic['file_hash'])
        & set(df_vexir['file_hash'])
    )
    print(f"  Common hashes found: {len(common_hashes)}")
    
    if len(common_hashes) == 0:
        raise ValueError("No common file_hash found across datasets! Check data generation.")
    
    # Step 2: Filter to only common hashes
    df_static = df_static[df_static['file_hash'].isin(common_hashes)].copy()
    df_dynamic = df_dynamic[df_dynamic['file_hash'].isin(common_hashes)].copy()
    df_vexir = df_vexir[df_vexir['file_hash'].isin(common_hashes)].copy()
    
    # Step 3: Sort by hash to ensure consistent ordering
    df_static = df_static.sort_values('file_hash').reset_index(drop=True)
    df_dynamic = df_dynamic.sort_values('file_hash').reset_index(drop=True)
    df_vexir = df_vexir.sort_values('file_hash').reset_index(drop=True)
    
    # Step 4: Verify alignment with assertions
    assert len(df_static) == len(df_dynamic) == len(df_vexir), \
        f"Dataset lengths don't match: static={len(df_static)}, dynamic={len(df_dynamic)}, vexir={len(df_vexir)}"
    
    assert all(df_static['file_hash'] == df_dynamic['file_hash']), \
        "Static and Dynamic file_hash mismatch after alignment!"
    
    assert all(df_static['file_hash'] == df_vexir['file_hash']), \
        "Static and VexIR file_hash mismatch after alignment!"
    
    # Step 5: Verify labels match across datasets
    assert all(df_static['label'] == df_dynamic['label']), \
        "Labels mismatch between Static and Dynamic datasets!"
    
    assert all(df_static['label'] == df_vexir['label']), \
        "Labels mismatch between Static and VexIR datasets!"
    
    print(f"  ✓ Alignment verified: {len(df_static)} samples")
    print(f"  ✓ All hashes match across datasets")
    print(f"  ✓ All labels match across datasets")
    print(f"  Label distribution: {df_static['label'].value_counts().to_dict()}")
    
    return df_static, df_dynamic, df_vexir


def create_feature_combinations(df_static, df_dynamic, df_vexir, vexir_dim=128):
    """
    Create all 7 feature combinations for ablation study.
    
    Returns dictionary with dataset name as key and feature data as value.
    """
    
    # Define feature columns
    static_cols = [c for c in df_static.columns if c not in ['file_hash', 'label']]
    dynamic_cols = [c for c in df_dynamic.columns if c not in ['file_hash', 'label']]
    vexir_cols = [c for c in df_vexir.columns if c.startswith('embed_')]
    
    print(f"\nFeature counts:")
    print(f"  Static features: {len(static_cols)}")
    print(f"  Dynamic features: {len(dynamic_cols)}")
    print(f"  VexIR embeddings: {len(vexir_cols)}")
    
    labels = df_static['label'].copy()
    file_hashes = df_static['file_hash'].copy()
    
    datasets = {}
    
    # 1. Static Features Only
    df_1 = df_static[static_cols].copy()
    datasets['static_only'] = {
        'features': df_1,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols
    }
    print(f"\n1. Static Only: {df_1.shape}")
    
    # 2. Dynamic Features Only
    df_2 = df_dynamic[dynamic_cols].copy()
    datasets['dynamic_only'] = {
        'features': df_2,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': dynamic_cols
    }
    print(f"2. Dynamic Only: {df_2.shape}")
    
    # 3. VexIR Only
    df_3 = df_vexir[vexir_cols].copy()
    datasets['vexir_only'] = {
        'features': df_3,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': vexir_cols
    }
    print(f"3. VexIR Only: {df_3.shape}")
    
    # 4. Static + VexIR Embeddings
    df_4 = pd.concat([
        df_static[static_cols].reset_index(drop=True),
        df_vexir[vexir_cols].reset_index(drop=True)
    ], axis=1)
    datasets['static_vexir'] = {
        'features': df_4,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols + vexir_cols
    }
    print(f"4. Static + VexIR: {df_4.shape}")
    
    # 5. Dynamic + VexIR Embeddings
    df_5 = pd.concat([
        df_dynamic[dynamic_cols].reset_index(drop=True),
        df_vexir[vexir_cols].reset_index(drop=True)
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
        df_vexir[vexir_cols].reset_index(drop=True)
    ], axis=1)
    datasets['static_dynamic_vexir'] = {
        'features': df_7,
        'labels': labels,
        'hashes': file_hashes,
        'feature_names': static_cols + dynamic_cols + vexir_cols
    }
    print(f"7. Static + Dynamic + VexIR: {df_7.shape}")
    
    return datasets


def run_experiment(datasets, vexir_dim=128):
    """
    Run ML experiments on all dataset combinations with all algorithms.
    Uses nested cross-validation with GridSearchCV for hyperparameter tuning.
    """
    
    df_results = pd.DataFrame(columns=[
        'vexir_dim', 'dataset', 'classifier', 'fold', 'precision', 'recall', 
        'f1', 'accuracy', 'best_params', 'num_features'
    ])
    
    df_feature_imp = None
    
    algorithms = ["XGB", "RF", "LR", "DT", "NB", "SVM", "DNN", "KNN"]
    
    dataset_order = [
        'static_only',
        'dynamic_only',
        'vexir_only',
        'static_vexir',
        'dynamic_vexir',
        'static_dynamic',
        'static_dynamic_vexir'
    ]
    
    dataset_display_names = {
        'static_only': '1. Static Features Only',
        'dynamic_only': '2. Dynamic Features Only',
        'vexir_only': '3. VexIR Only',
        'static_vexir': '4. Static + VexIR',
        'dynamic_vexir': '5. Dynamic + VexIR',
        'static_dynamic': '6. Static + Dynamic',
        'static_dynamic_vexir': '7. Static + Dynamic + VexIR'
    }
    
    for dataset_name in dataset_order:
        dataset = datasets[dataset_name]
        X = dataset['features']
        y = dataset['labels']
        feature_names = dataset['feature_names']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        for alg in algorithms:
            print('=' * 60)
            display_name = dataset_display_names.get(dataset_name, dataset_name)
            print(f"Dataset: {display_name}")
            print(f"Algorithm: {alg} | VexIR Dim: {vexir_dim}")
            print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
            print('=' * 60)
            
            sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
            
            fold_idx = 0
            for train_index, test_index in sss.split(X, y_encoded):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y_encoded[train_index], y_encoded[test_index]
                
                print(f"\nFold {fold_idx + 1}/10 - Train: {X_train.shape}, Test: {X_test.shape}")
                
                cv_inner = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
                
                clf, param_grid = select_algorithm(alg)
                
                pipeline = Pipeline([
                    ('var_threshold', VarianceThreshold(threshold=0.0)),
                    ('scaler', StandardScaler()),
                    ('clf', clf)
                ])
                
                result = GridSearchCV(
                    pipeline, param_grid, cv=cv_inner, 
                    scoring='f1_weighted', n_jobs=-1, refit=True
                )
                
                print("Training...")
                result.fit(X_train, y_train)
                
                y_pred = result.predict(X_test)
                
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1_res = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                acc = accuracy_score(y_test, y_pred)
                
                print(f"F1: {f1_res:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
                print(f"Best params: {result.best_params_}")
                
                # Store feature importance for tree-based models
                if alg in ["RF", "XGB", "DT"]:
                    try:
                        best_model = result.best_estimator_
                        if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
                            importances = best_model.named_steps['clf'].feature_importances_
                            
                            var_selector = best_model.named_steps['var_threshold']
                            selected_features = np.array(feature_names)[var_selector.get_support()]
                            
                            if len(selected_features) == len(importances):
                                imp_df = pd.DataFrame({
                                    'vexir_dim': vexir_dim,
                                    'dataset': dataset_name,
                                    'classifier': alg,
                                    'fold': fold_idx,
                                    'feature': selected_features,
                                    'importance': importances
                                })
                                
                                if df_feature_imp is None:
                                    df_feature_imp = imp_df
                                else:
                                    df_feature_imp = pd.concat([df_feature_imp, imp_df], ignore_index=True)
                    except Exception as e:
                        print(f"  Warning: Feature importance extraction failed: {e}")
                
                new_row = {
                    'vexir_dim': vexir_dim,
                    'dataset': dataset_name,
                    'classifier': alg,
                    'fold': fold_idx,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1_res,
                    'accuracy': acc,
                    'best_params': str(result.best_params_),
                    'num_features': X.shape[1]
                }
                df_results = pd.concat([
                    df_results, 
                    pd.DataFrame([new_row])
                ], ignore_index=True)
                
                fold_idx += 1
    
    return df_results, df_feature_imp


def generate_summary(df_results):
    """Generate summary statistics from results."""
    
    summary = df_results.groupby(['vexir_dim', 'dataset', 'classifier']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).reset_index()
    
    summary.columns = [
        'vexir_dim', 'dataset', 'classifier',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std',
        'f1_mean', 'f1_std',
        'accuracy_mean', 'accuracy_std'
    ]
    
    summary.to_csv("ransomware_detection_summary.csv", index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"\nSummary saved to: ransomware_detection_summary.csv")
    
    print("\n" + "-" * 60)
    print("BEST CLASSIFIER PER DATASET (by F1 Score)")
    print("-" * 60)
    
    dataset_display_names = {
        'static_only': '1. Static Features Only',
        'dynamic_only': '2. Dynamic Features Only',
        'vexir_only': '3. VexIR Only',
        'static_vexir': '4. Static + VexIR',
        'dynamic_vexir': '5. Dynamic + VexIR',
        'static_dynamic': '6. Static + Dynamic',
        'static_dynamic_vexir': '7. Static + Dynamic + VexIR'
    }
    
    for vexir_dim in summary['vexir_dim'].unique():
        print(f"\n--- VexIR Dimension: {vexir_dim} ---")
        dim_summary = summary[summary['vexir_dim'] == vexir_dim]
        
        for dataset in dim_summary['dataset'].unique():
            ds_data = dim_summary[dim_summary['dataset'] == dataset]
            best_row = ds_data.loc[ds_data['f1_mean'].idxmax()]
            
            print(f"\n{dataset_display_names.get(dataset, dataset)}:")
            print(f"  Best Classifier: {best_row['classifier']}")
            print(f"  F1: {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
            print(f"  Accuracy: {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f}")
    
    # Print overall best
    print("\n" + "-" * 60)
    print("OVERALL BEST CONFIGURATION")
    print("-" * 60)
    best_overall = summary.loc[summary['f1_mean'].idxmax()]
    print(f"VexIR Dim: {best_overall['vexir_dim']}")
    print(f"Dataset: {dataset_display_names.get(best_overall['dataset'], best_overall['dataset'])}")
    print(f"Classifier: {best_overall['classifier']}")
    print(f"F1: {best_overall['f1_mean']:.4f} ± {best_overall['f1_std']:.4f}")
    print(f"Accuracy: {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f}")
    
    return summary


def main():
    """Main function to run the complete experiment."""
    
    print("=" * 80)
    print("RANSOMWARE DETECTION IN ELF FILES")
    print("=" * 80)
    print("\nThis experiment evaluates 8 ML classifiers on:")
    print("  - 7 feature combinations")
    print("  - 5 VexIR embedding dimensions")
    print("\nFeature Combinations:")
    print("  1. Static Features Only")
    print("  2. Dynamic Features Only")
    print("  3. VexIR Embeddings Only")
    print("  4. Static + VexIR Embeddings")
    print("  5. Dynamic + VexIR Embeddings")
    print("  6. Static + Dynamic Features")
    print("  7. Static + Dynamic + VexIR Embeddings")
    print("\nVexIR embedding dimensions: 512, 256, 128, 64, 32, 16, 8")
    print("\nClassifiers: XGB, RF, LR, DT, NB, SVM, DNN, KNN")
    print("\n")
    
    vexir_dimensions = [512, 256, 128, 64, 32, 16, 8]
    
    all_results = None
    all_feature_imp = None
    
    for vexir_dim in vexir_dimensions:
        print("\n" + "#" * 80)
        print(f"# RUNNING EXPERIMENTS WITH VexIR DIMENSION: {vexir_dim}")
        print("#" * 80)
        
        # Load and align datasets by hash
        df_static, df_dynamic, df_vexir = load_datasets(vexir_dim)
        
        # Create feature combinations
        datasets = create_feature_combinations(df_static, df_dynamic, df_vexir, vexir_dim)
        
        # Run experiments
        df_results, df_feature_imp = run_experiment(datasets, vexir_dim)
        
        # Aggregate results
        if all_results is None:
            all_results = df_results
        else:
            all_results = pd.concat([all_results, df_results], ignore_index=True)
        
        if df_feature_imp is not None:
            if all_feature_imp is None:
                all_feature_imp = df_feature_imp
            else:
                all_feature_imp = pd.concat([all_feature_imp, df_feature_imp], ignore_index=True)
    
    # Save all results
    all_results.to_csv("ransomware_detection_results.csv", index=False)
    
    # Generate summary
    summary = generate_summary(all_results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - ransomware_detection_results.csv (detailed results)")
    print("  - ransomware_detection_summary.csv (summary statistics)")
    
    return all_results, summary


if __name__ == "__main__":
    df_results, summary = main()
