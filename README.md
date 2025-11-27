# Training Scripts

These scripts are self-contained and can be run independently to train the quantum ML model.

## Setup Instructions

1. **Folder Structure**: The scripts expect this structure:
   ```
   project_root/
   ├── scripts/
   │   ├── extract_frames.py
   │   ├── extract_features.py
   │   ├── auto_label_risk.py
   │   └── train_quantum.py
   ├── dataset/
   │   ├── videos/          # Put your video files here
   │   ├── images/          # Frames will be extracted here
   │   └── features.csv     # Features will be saved here
   └── models/
       └── quantum/
           └── qml_model.pkl  # Trained model will be saved here
   ```

2. **Environment Variables**: Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent
   ```
   Or set these as environment variables.

3. **Install Dependencies**:
   ```bash
   pip install pandas numpy opencv-python requests pennylane scikit-learn joblib
   ```

## Usage

Run the scripts in this order:

1. **Extract frames from videos**:
   ```bash
   python scripts/extract_frames.py --interval 10
   ```

2. **Extract features from images** (requires Gemini API):
   ```bash
   python scripts/extract_features.py
   ```

3. **Auto-label risk values**:
   ```bash
   python scripts/auto_label_risk.py
   ```

4. **Train the quantum model**:
   ```bash
   python scripts/train_quantum.py
   ```

The trained model will be saved to `models/quantum/qml_model.pkl`.

## Notes

- All scripts are self-contained and don't depend on other project files
- The scripts automatically create necessary directories
- Make sure you have video files in `dataset/videos/` before running step 1
