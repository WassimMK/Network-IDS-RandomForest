import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_model():
    """
    Load the trained model
    """
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "randomForestRegression_model.pkl")

    print(f"Loading model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Model Not Found",
            f"Model not found at:\n{MODEL_PATH}\n\n"
            "Please run 'python src/main.py' first to train the model."
        )
        return None

    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded successfully")
        print(f"  Number of features expected: {model.n_features_in_}")
        return model
    except Exception as e:
        messagebox.showerror(
            "Error Loading Model",
            f"Error loading model:\n{str(e)}"
        )
        return None


def create_gui():
    """
    Create the prediction GUI
    """
    # Load model
    model = load_model()
    if model is None:
        sys.exit(1)

    FEATURE_COUNT = model.n_features_in_

    # Create main window
    root = tk.Tk()
    root.title("Next-Step Cyber Attack Prediction System")
    root.geometry("800x600")
    root.configure(bg='#f0f0f0')

    # Set icon (optional)
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass

    # Header
    header_frame = tk.Frame(root, bg='#2c3e50', height=80)
    header_frame.pack(fill='x')
    header_frame.pack_propagate(False)

    title_label = tk.Label(
        header_frame,
        text="Next-Step Cyber Attack Prediction",
        font=('Arial', 20, 'bold'),
        fg='white',
        bg='#2c3e50'
    )
    title_label.pack(expand=True)

    sub_title = tk.Label(
        header_frame,
        text="Random Forest Regression Model",
        font=('Arial', 12),
        fg='#ecf0f1',
        bg='#2c3e50'
    )
    sub_title.pack()

    # Main content frame
    main_frame = tk.Frame(root, bg='#f0f0f0', padx=20, pady=20)
    main_frame.pack(fill='both', expand=True)

    # Input section
    input_frame = tk.LabelFrame(
        main_frame,
        text=f"Input Features (Enter {FEATURE_COUNT} comma-separated values)",
        font=('Arial', 11, 'bold'),
        bg='#f0f0f0',
        padx=15,
        pady=15
    )
    input_frame.pack(fill='x', pady=(0, 15))

    # Input label and entry
    tk.Label(
        input_frame,
        text="Features:",
        font=('Arial', 10),
        bg='#f0f0f0'
    ).pack(anchor='w', pady=(0, 5))

    entry = tk.Entry(
        input_frame,
        width=100,
        font=('Courier', 10)
    )
    entry.pack(fill='x', pady=(0, 10))

    # Example input
    example_text = f"Example: {', '.join(['0.5'] * min(5, FEATURE_COUNT))}"
    if FEATURE_COUNT > 5:
        example_text += f", ... (total {FEATURE_COUNT} values)"

    example_label = tk.Label(
        input_frame,
        text=example_text,
        font=('Arial', 9, 'italic'),
        fg='#7f8c8d',
        bg='#f0f0f0'
    )
    example_label.pack(anchor='w')

    # Quick fill buttons
    button_frame = tk.Frame(input_frame, bg='#f0f0f0')
    button_frame.pack(fill='x', pady=(10, 0))

    def fill_normal():
        """Fill with normal traffic pattern"""
        values = [str(round(np.random.normal(0, 0.1), 3)) for _ in range(FEATURE_COUNT)]
        entry.delete(0, tk.END)
        entry.insert(0, ', '.join(values))

    def fill_attack():
        """Fill with attack traffic pattern"""
        values = [str(round(np.random.normal(1, 0.3), 3)) for _ in range(FEATURE_COUNT)]
        entry.delete(0, tk.END)
        entry.insert(0, ', '.join(values))

    def fill_random():
        """Fill with random values"""
        values = [str(round(np.random.random(), 3)) for _ in range(FEATURE_COUNT)]
        entry.delete(0, tk.END)
        entry.insert(0, ', '.join(values))

    tk.Button(
        button_frame,
        text="Normal Pattern",
        command=fill_normal,
        bg='#27ae60',
        fg='white',
        font=('Arial', 9, 'bold'),
        padx=10,
        pady=5
    ).pack(side='left', padx=(0, 5))

    tk.Button(
        button_frame,
        text="Attack Pattern",
        command=fill_attack,
        bg='#e74c3c',
        fg='white',
        font=('Arial', 9, 'bold'),
        padx=10,
        pady=5
    ).pack(side='left', padx=(0, 5))

    tk.Button(
        button_frame,
        text="Random Values",
        command=fill_random,
        bg='#3498db',
        fg='white',
        font=('Arial', 9, 'bold'),
        padx=10,
        pady=5
    ).pack(side='left')

    # Prediction function
    def predict():
        values_text = entry.get().strip()

        if not values_text:
            messagebox.showwarning("Input Required", "Please enter feature values")
            return

        # Parse input
        values = [v.strip() for v in values_text.split(',')]

        if len(values) != FEATURE_COUNT:
            messagebox.showerror(
                "Input Error",
                f"Expected {FEATURE_COUNT} values, got {len(values)}\n\n"
                f"Please enter exactly {FEATURE_COUNT} comma-separated numbers."
            )
            return

        # Convert to float
        try:
            values_array = np.array(values, dtype=float).reshape(1, -1)
        except ValueError as e:
            messagebox.showerror(
                "Input Error",
                f"Invalid input: {str(e)}\n\n"
                "All values must be numbers (e.g., 0.5, 1.2, -0.3)"
            )
            return

        # Make prediction
        try:
            prediction = model.predict(values_array)[0]

            # Determine risk level
            if prediction < 0.3:
                risk_level = "LOW"
                color = "#27ae60"
                description = "Normal traffic pattern detected"
            elif prediction < 0.7:
                risk_level = "MEDIUM"
                color = "#f39c12"
                description = "Suspicious activity detected"
            else:
                risk_level = "HIGH"
                color = "#e74c3c"
                description = "Attack pattern detected"

            # Update result display
            result_text.set(
                f"PREDICTION: {risk_level}\n"
                f"Score: {prediction:.4f}\n"
                f"{description}"
            )
            result_label.config(fg=color)

            # Update progress bar
            progress_value = min(100, int(prediction * 100))
            progress_bar['value'] = progress_value

            if risk_level == "LOW":
                progress_bar['style'] = 'green.Horizontal.TProgressbar'
            elif risk_level == "MEDIUM":
                progress_bar['style'] = 'orange.Horizontal.TProgressbar'
            else:
                progress_bar['style'] = 'red.Horizontal.TProgressbar'

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction:\n{str(e)}")

    # Predict button
    predict_button = tk.Button(
        main_frame,
        text="PREDICT NEXT-STEP ATTACK RISK",
        command=predict,
        bg='#2c3e50',
        fg='white',
        font=('Arial', 12, 'bold'),
        padx=20,
        pady=10,
        relief='raised',
        borderwidth=3
    )
    predict_button.pack(pady=(10, 20))

    # Results section
    results_frame = tk.LabelFrame(
        main_frame,
        text="Prediction Results",
        font=('Arial', 11, 'bold'),
        bg='#f0f0f0',
        padx=15,
        pady=15
    )
    results_frame.pack(fill='both', expand=True)

    # Progress bar
    progress_style = ttk.Style()
    progress_style.theme_use('clam')

    # Define styles for different risk levels
    for color_name, color_code in [('green', '#27ae60'), ('orange', '#f39c12'), ('red', '#e74c3c')]:
        progress_style.configure(
            f'{color_name}.Horizontal.TProgressbar',
            background=color_code,
            troughcolor='#ecf0f1',
            bordercolor='#bdc3c7',
            lightcolor=color_code,
            darkcolor=color_code
        )

    progress_bar = ttk.Progressbar(
        results_frame,
        orient='horizontal',
        length=400,
        mode='determinate',
        style='green.Horizontal.TProgressbar'
    )
    progress_bar.pack(pady=(0, 15))

    # Result label
    result_text = tk.StringVar()
    result_text.set("Enter features and click PREDICT")

    result_label = tk.Label(
        results_frame,
        textvariable=result_text,
        font=('Arial', 14, 'bold'),
        bg='#f0f0f0',
        justify='center'
    )
    result_label.pack(expand=True)

    # Status bar
    status_bar = tk.Frame(root, bg='#2c3e50', height=30)
    status_bar.pack(side='bottom', fill='x')
    status_bar.pack_propagate(False)

    status_text = tk.StringVar()
    status_text.set(f"Model: Random Forest | Features: {FEATURE_COUNT} | Ready")

    status_label = tk.Label(
        status_bar,
        textvariable=status_text,
        font=('Arial', 9),
        fg='white',
        bg='#2c3e50'
    )
    status_label.pack(side='left', padx=10)

    # Clear button
    def clear_input():
        entry.delete(0, tk.END)
        result_text.set("Enter features and click PREDICT")
        result_label.config(fg='black')
        progress_bar['value'] = 0

    clear_button = tk.Button(
        status_bar,
        text="Clear",
        command=clear_input,
        bg='#7f8c8d',
        fg='white',
        font=('Arial', 9),
        padx=10
    )
    clear_button.pack(side='right', padx=10, pady=5)

    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    create_gui()