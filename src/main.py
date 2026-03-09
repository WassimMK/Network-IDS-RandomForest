import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_data, preprocess_data, preprocess_single_file
from randomForestReggression_model import train_model, save_model


def check_and_fix_filenames():
    """
    Check for files with double .csv.csv extension and fix them
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")

    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        return False

    # Files that might have double extension
    files_to_check = [
        ("UNSW_NB15_train.csv.csv", "UNSW_NB15_train.csv"),
        ("UNSW_NB15_test.csv.csv", "UNSW_NB15_test.csv"),
        ("UNSW_NB15_train.csv", "UNSW_NB15_train.csv"),  # Already correct
        ("UNSW_NB15_test.csv", "UNSW_NB15_test.csv")  # Already correct
    ]

    fixed = False
    for old_name, new_name in files_to_check:
        old_path = os.path.join(data_dir, old_name)
        new_path = os.path.join(data_dir, new_name)

        # Only rename if old exists and new doesn't
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"✓ Renamed: {old_name} -> {new_name}")
            fixed = True
        elif os.path.exists(old_path) and old_name != new_name:
            # Old file exists but new also exists - remove old
            os.remove(old_path)
            print(f"✓ Removed duplicate: {old_name}")
            fixed = True

    return fixed


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("NEXT-STEP CYBER ATTACK PREDICTION TRAINING")
    print("=" * 60)

    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project directory: {BASE_DIR}")

    # Check and fix file names
    print("\n[1/4] Checking data files...")
    if check_and_fix_filenames():
        print("  Fixed file naming issues")

    # Define paths
    TRAIN_PATH = os.path.join(BASE_DIR, "data", "UNSW_NB15_train.csv")
    TEST_PATH = os.path.join(BASE_DIR, "data", "UNSW_NB15_test.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "randomForestRegression_model.pkl")

    print(f"  Train path: {TRAIN_PATH}")
    print(f"  Test path: {TEST_PATH}")

    # Check if files exist
    if not os.path.exists(TRAIN_PATH):
        print(f"\n✗ ERROR: Train file not found at {TRAIN_PATH}")
        print("\nPlease ensure:")
        print("1. UNSW_NB15_train.csv is in the 'data' folder")
        print("2. The file name is exactly 'UNSW_NB15_train.csv'")
        print("\nFiles in data folder:")
        data_dir = os.path.join(BASE_DIR, "data")
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                print(f"  - {f}")
        return

    # Load data
    print("\n[2/4] Loading data...")
    try:
        # Load with small fraction for initial testing
        train_df = load_data(TRAIN_PATH, sample_fraction=0.2)  # Reduced to 20% for faster processing

        if os.path.exists(TEST_PATH):
            test_df = load_data(TEST_PATH, sample_fraction=0.1)  # Reduced to 10% for faster processing
            print("\n✓ Both train and test files loaded")
            use_separate_files = True
        else:
            print(f"\n⚠ Test file not found at {TEST_PATH}")
            print("Using train file for both training and testing")
            test_df = train_df
            use_separate_files = False

        print(f"\nDataset Information:")
        print(f"  Train shape: {train_df.shape}")
        print(f"  Test shape: {test_df.shape}")

        # Show data types
        print(f"\nTrain data types:")
        for col, dtype in train_df.dtypes.items():
            if dtype == 'object':
                unique_vals = train_df[col].nunique()
                print(f"    {col}: {dtype} (unique: {unique_vals})")

    except Exception as e:
        print(f"\n✗ ERROR loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Preprocess data
    print("\n[3/4] Preprocessing data...")
    try:
        if use_separate_files:
            # Use separate train and test files
            X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
        else:
            # Use single file and split it
            X_train, X_test, y_train, y_test = preprocess_single_file(train_df)

        # Show preprocessing results
        print(f"\nPreprocessing completed:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")

        # Show some feature names
        print(f"\nFirst 10 feature names:")
        for i, col in enumerate(X_train.columns[:10]):
            print(f"  {i + 1}. {col}")

    except Exception as e:
        print(f"\n✗ ERROR preprocessing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Train model
    print("\n[4/4] Training Random Forest model...")
    try:
        model = train_model(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"\n✗ ERROR training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Save model
    print("\n[5/5] Saving model...")
    try:
        save_model(model, MODEL_PATH)
    except Exception as e:
        print(f"\n✗ ERROR saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python src/visualize.py' to see feature importance")
    print("2. Run 'python src/gui.py' to launch prediction interface")
    print(f"3. Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()