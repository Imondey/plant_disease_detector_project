🌿 Vegetable Plant Disease Detector
A simple web-based application built with Streamlit that detects diseases in vegetable plant leaves using a Decision Tree classifier.

📂 Project Structure

PLANT_DISEASE_DETECTOR/
│
├── app.py                  # Main Streamlit app
├── logo.jpg                # Logo displayed on the UI
├── requirements.txt        # Python dependencies
│
├── data/                   # Folder for training data (images)
│
├── models/
│   ├── decision_tree.py    # Code for training Decision Tree model
│   └── lstm_model.py       # (Optional) LSTM model (not used in app.py yet)
│
├── utils/
│   └── preprocess.py       # Helper functions (e.g., image loading)

🚀 Features
Upload leaf images of vegetable plants (JPG/PNG).

Detect disease using a trained Decision Tree model.

Compact and beginner-friendly code structure.

Streamlit-based responsive UI.

Logo aligned to top-right corner.

🧠 Tech Stack
Python

Streamlit

OpenCV

NumPy

Scikit-learn

