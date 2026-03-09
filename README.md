Network IDS - Random Forest Classifier
This is a machine learning-based Intrusion Detection System (IDS). I built this to demonstrate how ensemble learning can be used to detect modern network threats by analyzing traffic patterns.

** Project Overview
The system is designed to classify network traffic as either normal or malicious. I chose the UNSW-NB15 dataset for this project because it includes modern attack types (like Fuzzers, Analysis, and Backdoors) that older datasets often miss.

** How it Works
Preprocessing: The code handles missing values, encodes categorical features (like protocol and service), and scales numerical data to ensure the model trains accurately.

Model: I used a Random Forest Regressor (interpreted for classification) with 100 estimators. This approach is great for network data because it reduces the risk of overfitting on specific traffic noise.

GUI: I included a Tkinter-based interface (gui.py) that allows you to manually input network features and see the model's prediction in real-time.

Visualization: The project includes scripts to plot feature importance, showing exactly which parts of a network packet (like byte count or TTL) are the biggest "red flags" for an attack.

** Technical Stack
Language: Python

Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and Joblib.

** Structure
/src: Python scripts for the full pipeline (preprocess, train, visualize, and GUI).

/models: Contains the trained .pkl files and feature importance plots.

/data: Includes the train and test CSV files from the UNSW-NB15 set.
