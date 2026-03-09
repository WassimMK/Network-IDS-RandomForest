import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def visualize_model():
    """
    Load and visualize the trained model
    """
    print("=" * 60)
    print("MODEL VISUALIZATION")
    print("=" * 60)

    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "randomForestRegression_model.pkl")

    print(f"Looking for model at: {MODEL_PATH}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n✗ Model not found at {MODEL_PATH}")
        print("\nPlease run 'python src/main.py' first to train the model")
        return

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Number of trees: {model.n_estimators}")
        print(f"  Number of features: {model.n_features_in_}")
    except Exception as e:
        print(f"\n✗ Error loading model: {str(e)}")
        return

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\n[1/2] Feature Importance Analysis")
        print("-" * 40)

        # Create feature importance DataFrame
        feature_names = [f"Feature_{i + 1}" for i in range(model.n_features_in_)]
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)

        # Display top features
        print("Top 20 Most Important Features:")
        for i, (feature, importance) in enumerate(importances.head(20).items(), 1):
            print(f"{i:2}. {feature}: {importance:.4f}")

        # Create visualization
        plt.figure(figsize=(14, 8))

        # Bar plot for top 20 features
        plt.subplot(1, 2, 1)
        top_features = importances.head(20)
        colors = plt.cm.viridis(top_features.values / top_features.values.max())
        bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score')
        plt.title('Top 20 Feature Importance')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_features.values)):
            plt.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)

        # Pie chart for importance distribution
        plt.subplot(1, 2, 2)
        top_10_sum = importances.head(10).sum()
        others_sum = importances[10:].sum()
        sizes = [top_10_sum, others_sum]
        labels = ['Top 10 Features', 'Other Features']
        colors_pie = ['#ff9999', '#66b3ff']
        plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Importance Distribution')

        plt.tight_layout()
        plt.show()

        # Save the plot
        plot_path = os.path.join(BASE_DIR, "models", "feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: {plot_path}")

    # Model information
    print("\n[2/2] Model Information")
    print("-" * 40)
    print(f"Model parameters:")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  random_state: {model.random_state}")
    print(f"  n_jobs: {model.n_jobs}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    visualize_model()