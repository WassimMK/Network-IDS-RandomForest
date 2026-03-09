import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


def load_data(file_path, sample_fraction=0.5):
    """
    Load dataset from CSV file

    Args:
        file_path: Path to CSV file
        sample_fraction: Fraction of data to sample (for faster testing)

    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Take only part of the dataset for faster processing
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"  Sampled to {len(df)} rows ({sample_fraction * 100}%)")

    return df


def prepare_features_and_target(df):
    """
    Prepare features (X) and target (y) from dataframe

    Args:
        df: Input DataFrame

    Returns:
        X (features), y (target), feature_columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Drop identifier columns
    columns_to_drop = ['id', 'srcip', 'dstip', 'sport', 'dsport']

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
            print(f"  Dropped identifier column: {col}")

    # Check for label column
    label_column = None
    possible_labels = ['label', 'Label', 'attack_cat', 'Attack']

    for label in possible_labels:
        if label in df.columns:
            label_column = label
            print(f"  Using label column: {label_column}")
            break

    if label_column is None:
        raise ValueError("No suitable label column found in dataset")

    # Encode label if it's categorical
    if df[label_column].dtype == 'object':
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])
        print(f"  Encoded labels with {len(le.classes_)} classes")

    # Create next-step target
    df['Label_next'] = df[label_column].shift(-1)
    df = df.dropna()

    # Prepare features and target
    X = df.drop([label_column, 'Label_next'], axis=1)
    y = df['Label_next']

    print(f"  Features shape: X={X.shape}, Target shape: y={y.shape}")

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if categorical_cols:
        print(f"  Categorical columns to encode: {categorical_cols}")
        print(f"  Numerical columns: {len(numerical_cols)}")

    return X, y, categorical_cols, numerical_cols


def encode_categorical_features(X_train, X_test, categorical_cols):
    """
    Encode categorical features using OneHotEncoding

    Args:
        X_train: Training features
        X_test: Testing features
        categorical_cols: List of categorical column names

    Returns:
        Encoded X_train, X_test, feature_names
    """
    if not categorical_cols:
        return X_train, X_test, X_train.columns.tolist()

    print(f"\nEncoding categorical features: {categorical_cols}")

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', [col for col in X_train.columns if col not in categorical_cols]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    # Fit and transform training data
    X_train_encoded = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_encoded = preprocessor.transform(X_test)

    # Get feature names
    feature_names = []

    # Numerical feature names
    for col in X_train.columns:
        if col not in categorical_cols:
            feature_names.append(col)

    # Categorical feature names from one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    for i, col in enumerate(categorical_cols):
        for category in cat_encoder.categories_[i]:
            feature_names.append(f"{col}_{category}")

    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Encoded features: {X_train_encoded.shape[1]}")

    return X_train_encoded, X_test_encoded, feature_names


def preprocess_data(train_df, test_df):
    """
    Preprocess both train and test dataframes for next-step prediction

    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\nPreprocessing training data...")
    X_train, y_train, cat_cols_train, num_cols_train = prepare_features_and_target(train_df)

    print("\nPreprocessing testing data...")
    X_test, y_test, cat_cols_test, num_cols_test = prepare_features_and_target(test_df)

    # Ensure categorical columns are the same
    categorical_cols = list(set(cat_cols_train).union(set(cat_cols_test)))

    # Encode categorical features
    X_train_encoded, X_test_encoded, feature_names = encode_categorical_features(
        X_train, X_test, categorical_cols
    )

    print(f"\nFinal dataset sizes:")
    print(f"  X_train shape: {X_train_encoded.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test_encoded.shape}, y_test shape: {y_test.shape}")
    print(f"  Number of features: {X_train_encoded.shape[1]}")

    # Convert to DataFrames with proper column names
    X_train_final = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_encoded, columns=feature_names)

    return X_train_final, X_test_final, y_train, y_test


def preprocess_single_file(df):
    """
    Preprocess a single dataframe (when using same file for train/test)

    Args:
        df: Input DataFrame

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\nPreprocessing data from single file...")
    X, y, categorical_cols, numerical_cols = prepare_features_and_target(df)

    # Encode categorical features
    if categorical_cols:
        X_encoded, _, feature_names = encode_categorical_features(X, X, categorical_cols)
        X = pd.DataFrame(X_encoded, columns=feature_names)

    # Time-series split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"\nSplit results:")
    print(f"  Train size: {len(X_train)} samples")
    print(f"  Test size: {len(X_test)} samples")
    print(f"  Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test