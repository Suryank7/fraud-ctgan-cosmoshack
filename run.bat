@echo off
echo ========================================================
echo AI-Based Fraud Detection System (CTGAN Edition)
echo ========================================================

echo [1/6] Setting up environment...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/6] Generating Simulated Data...
python utils/helpers.py --action generate_data
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/6] Training CTGAN on Fraud Data...
python ctgan/train_ctgan.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/6] Generating Synthetic Samples...
python ctgan/generate_synthetic.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/6] Training & Evaluating Models...
python model/train.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/6] Launching Streamlit App...
streamlit run app/streamlit_app.py
