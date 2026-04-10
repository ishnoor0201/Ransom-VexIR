"""
ML Model Training Script with Normalized Dataset
Normalizes data before training XGBoost, CatBoost, Random Forest, VotingEnsemble, BaggingEnsemble, and CNN
Evaluates: Accuracy, Precision, Recall, F1 Score, AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def load_and_preprocess_data(filepath):
    """Load dataset, normalize features, and preprocess for training."""
    print("=" * 60)
    print("Loading and Preprocessing Data (with Normalization)")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    
    # Drop 'file' column if present (it's an identifier, not a feature)
    if 'file' in df.columns:
        df = df.drop('file', axis=1)
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Normalize features using MinMaxScaler (scales to [0, 1])
    print("\nNormalizing features using MinMaxScaler...")
    normalizer = MinMaxScaler()
    X_normalized = pd.DataFrame(
        normalizer.fit_transform(X),
        columns=X.columns
    )
    
    print(f"Feature ranges after normalization:")
    print(f"  Min: {X_normalized.min().min():.4f}")
    print(f"  Max: {X_normalized.max().max():.4f}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Scale features (StandardScaler for zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def evaluate_model(y_true, y_pred, y_prob=None):
    """Calculate evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['AUC'] = roc_auc_score(y_true, y_pred)
    
    return metrics


def print_metrics(model_name, metrics):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"{metric:12}: {value:.4f}")


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost classifier."""
    print("\n" + "-"*40)
    print("Training XGBoost...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics("XGBoost", metrics)
    
    return model, metrics


def train_catboost(X_train, X_test, y_train, y_test):
    """Train CatBoost classifier."""
    print("\n" + "-"*40)
    print("Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics("CatBoost", metrics)
    
    return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest classifier."""
    print("\n" + "-"*40)
    print("Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics("Random Forest", metrics)
    
    return model, metrics


def train_voting_ensemble(X_train, X_test, y_train, y_test):
    """Train Voting Ensemble classifier."""
    print("\n" + "-"*40)
    print("Training Voting Ensemble...")
    
    # Create base estimators
    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    cat = CatBoostClassifier(
        iterations=50,
        depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    model = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('cat', cat)],
        voting='soft'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics("Voting Ensemble", metrics)
    
    return model, metrics


def train_bagging_ensemble(X_train, X_test, y_train, y_test):
    """Train Bagging Ensemble classifier."""
    print("\n" + "-"*40)
    print("Training Bagging Ensemble...")
    
    base_estimator = RandomForestClassifier(
        n_estimators=10,
        max_depth=6,
        random_state=RANDOM_STATE
    )
    
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics("Bagging Ensemble", metrics)
    
    return model, metrics


class CNN1D(nn.Module):
    """1D CNN model for tabular data."""
    def __init__(self, input_shape):
        super(CNN1D, self).__init__()
        
        # Conv blocks
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Calculate flattened size
        self.flat_size = 64 * (input_shape // 4)
        
        # Dense layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Conv blocks
        x = self.pool1(self.dropout(self.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.dropout(self.relu(self.bn4(self.fc1(x))))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x


def train_cnn(X_train, X_test, y_train, y_test):
    """Train CNN classifier using PyTorch."""
    print("\n" + "-"*40)
    print("Training CNN...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_shape = X_train.shape[1]
    model = CNN1D(input_shape).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Predict
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_tensor).squeeze().cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics("CNN", metrics)
    
    return model, metrics


def create_results_summary(all_results):
    """Create a summary DataFrame of all results."""
    summary_df = pd.DataFrame(all_results).T
    summary_df.index.name = 'Model'
    return summary_df


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("ML MODEL TRAINING PIPELINE (NORMALIZED DATA)")
    print("="*60)
    
    # Load data
    filepath = 'static_embeddings.csv'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    
    # Dictionary to store all results
    all_results = {}
    
    # Train all models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # 1. XGBoost
    _, metrics = train_xgboost(X_train, X_test, y_train, y_test)
    all_results['XGBoost'] = metrics
    
    # 2. CatBoost
    _, metrics = train_catboost(X_train, X_test, y_train, y_test)
    all_results['CatBoost'] = metrics
    
    # 3. Random Forest
    _, metrics = train_random_forest(X_train, X_test, y_train, y_test)
    all_results['Random Forest'] = metrics
    
    # 4. Voting Ensemble
    _, metrics = train_voting_ensemble(X_train, X_test, y_train, y_test)
    all_results['Voting Ensemble'] = metrics
    
    # 5. Bagging Ensemble
    _, metrics = train_bagging_ensemble(X_train, X_test, y_train, y_test)
    all_results['Bagging Ensemble'] = metrics
    
    # 6. CNN
    _, metrics = train_cnn(X_train, X_test, y_train, y_test)
    all_results['CNN'] = metrics
    
    # Create and display summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY ")
    print("="*60)
    
    summary_df = create_results_summary(all_results)
    print("\n")
    print(summary_df.round(4).to_string())
    
    # Save results to CSV
    summary_df.to_csv('model_results.csv')
    print("\n\nResults saved to 'model_results.csv'")
    
    # Find best model for each metric
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    for col in summary_df.columns:
        best_model = summary_df[col].idxmax()
        best_value = summary_df[col].max()
        print(f"{col:12}: {best_model} ({best_value:.4f})")
    
    return summary_df


if __name__ == "__main__":
    results = main()
