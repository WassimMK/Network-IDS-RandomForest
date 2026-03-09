from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import joblib
import os


def train_model(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model for next-step prediction

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data

    Returns:
        Trained model
    """
    print("Training Random Forest model...")

    # Initialize model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced for faster training
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print(f"  Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"  Test set has {X_test.shape[0]} samples")

    # Train model
    model.fit(X_train, y_train)
    print("  Model training completed")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root MSE (RMSE): {mse ** 0.5:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")

    # For binary classification interpretation
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = (y_pred_class == y_test).mean()
    print(f"  Binary accuracy (threshold 0.5): {accuracy:.4f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        )
        feature_importance = feature_importance.sort_values(ascending=False)

        print("\nTop 15 Most Important Features:")
        print("-" * 40)
        for i, (feature, importance) in enumerate(feature_importance.head(15).items(), 1):
            print(f"{i:2}. {feature}: {importance:.4f}")
    else:
        print("\nFeature importance not available for this model type")

    return model


def save_model(model, path):
    """
    Save trained model to disk

    Args:
        model: Trained model
        path: Path to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")
    print(f"Model details:")
    print(f"  Type: {type(model).__name__}")
    print(f"  Features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}")
    print(f"  Estimators: {model.n_estimators if hasattr(model, 'n_estimators') else 'N/A'}")