# app.py - Heart Disease Prediction System

from flask import Flask, render_template, request
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import os
import json
import base64
from openai import OpenAI
import pytesseract
from PIL import Image
import cv2
import traceback

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load trained models
try:
    artifact = joblib.load("model.pkl")
    models = artifact["models"]
    scaler = artifact["scaler"]
    FEATURE_NAMES = artifact["features"]
    accuracies = artifact["accuracies"]
    print("✅ Models loaded successfully!")
except FileNotFoundError:
    print("❌ model.pkl not found! Please run train_model.py first")
    exit(1)

# Feature names
REQUIRED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# Mappings for display
MAPPINGS = {
    "sex": {0: "Female", 1: "Male"},
    "fbs": {0: "No", 1: "Yes"},
    "exang": {0: "No", 1: "Yes"},
    "cp": {0: "Asymptomatic", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Typical Angina"},
    "restecg": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "LV Hypertrophy"},
    "slope": {0: "Downsloping", 1: "Flat", 2: "Upsloping"},
    "thal": {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
}

DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Gender",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia"
}

# OCR Function
def extract_text_from_image(image_file):
    try:
        print("📄 Starting OCR extraction...")
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        processed_image = Image.fromarray(denoised)
        extracted_text = pytesseract.image_to_string(processed_image, config='--psm 6')
        
        print("📄 Extracted text:")
        print(extracted_text)
        return extracted_text
    except Exception as e:
        raise ValueError(f"OCR Error: {str(e)}")

# ECG Analysis Function
def analyze_ecg_image(image_file):
    try:
        print("🫀 Analyzing ECG image...")
        image_bytes = image_file.read()
        image_file.seek(0)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = """Analyze this ECG image and extract:
        - thalach (Maximum Heart Rate in bpm)
        - oldpeak (ST Depression in mm, 0-6 range)
        - slope (0=Downsloping, 1=Flat, 2=Upsloping)
        - restecg (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)
        
        Return ONLY JSON: {"thalach": 150, "oldpeak": 1.5, "slope": 1, "restecg": 0}
        Use defaults if unsure: thalach=140, oldpeak=0.0, slope=2, restecg=0"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=500,
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.strip("`").replace("json", "").strip()
        
        ecg_features = json.loads(result_text)
        print("🫀 ECG Results:", ecg_features)
        return ecg_features
    except Exception as e:
        raise ValueError(f"ECG Analysis Error: {str(e)}")

# Text Parsing Function
def extract_features_from_report(report_text):
    try:
        print("🤖 Parsing medical report...")
        prompt = f"""Extract medical features from this report as JSON with these keys:
        {FEATURE_NAMES}
        
        Mappings:
        - age (number), sex (1=Male, 0=Female)
        - cp (0-3), trestbps (mm Hg), chol (mg/dl)
        - fbs (1=Yes, 0=No), restecg (0-2)
        - thalach (bpm), exang (1=Yes, 0=No)
        - oldpeak (mm), slope (0-2), ca (0-3), thal (1-3)
        
        Return ONLY JSON, no markdown.
        
        Report: {report_text}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract medical data as JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.strip("`").replace("json", "").strip()
        
        features = json.loads(result_text)
        if isinstance(features, list):
            features = features[0]
        
        print("🤖 Extracted features:", features)
        return {f: features[f] for f in FEATURE_NAMES}
    except Exception as e:
        raise ValueError(f"Parsing Error: {str(e)}")

# Helper function
def coerce_features(features):
    coerced = {}
    for f in FEATURE_NAMES:
        raw = features.get(f, 0)
        if f == "oldpeak":
            coerced[f] = float(raw) if raw else 0.0
        else:
            coerced[f] = int(float(raw)) if raw else 0
    return coerced

@app.route("/", methods=["GET"])
def index():
    print("📍 Homepage accessed")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("\n" + "="*60)
    print("🔍 PREDICTION REQUEST")
    print("="*60)
    
    action = request.form.get("action")
    print(f"📋 Action: {action}")
    
    inputs = {}
    data_source = "Unknown"
    
    try:
        # ECG Image
        if action == "ecg_image":
            ecg_file = request.files.get("ecg_image")
            if not ecg_file or ecg_file.filename == "":
                return "No ECG image uploaded.", 400
            
            ecg_features = analyze_ecg_image(ecg_file)
            
            for f in FEATURE_NAMES:
                if f in ecg_features:
                    val = ecg_features[f]
                else:
                    val = request.form.get(f, 50 if f == "age" else 0)
                inputs[f] = float(val) if f == "oldpeak" else int(float(val))
            
            data_source = "ECG Image Analysis"
        
        # OCR Image
        elif action == "ocr_image":
            ocr_file = request.files.get("ocr_image")
            if not ocr_file or ocr_file.filename == "":
                return "No image uploaded.", 400
            
            extracted_text = extract_text_from_image(ocr_file)
            features = extract_features_from_report(extracted_text)
            inputs = coerce_features(features)
            data_source = "OCR from Image"
        
        # Text File
        elif action == "file":
            file = request.files.get("report_file")
            if not file or file.filename == "":
                return "No file uploaded.", 400
            
            report_text = file.read().decode("utf-8")
            features = extract_features_from_report(report_text)
            inputs = coerce_features(features)
            data_source = "Text Report"
        
        # Manual Input
        elif action == "manual":
            for f in FEATURE_NAMES:
                val = request.form.get(f)
                if val is None or val == "":
                    return f"Missing value for {f}", 400
                inputs[f] = float(val) if f == "oldpeak" else int(float(val))
            data_source = "Manual Input"
        
        else:
            return "Unknown action.", 400
        
        # Prediction
        print("🤖 Running prediction...")
        X = pd.DataFrame([inputs], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X)
        
        details = {}
        chart_labels = []
        chart_values = []
        
        for name, clf in models.items():
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_scaled)[0][1]
            else:
                prob = float(clf.predict(X_scaled)[0])
            
            percentage = prob * 100
            details[name] = f"{percentage:.1f}%"
            chart_labels.append(name)
            chart_values.append(round(percentage, 1))
            print(f"  {name}: {percentage:.1f}%")
        
        percent = round(sum(chart_values) / len(chart_values), 1)
        print(f"✅ Final Risk: {percent}%")
        print("="*60 + "\n")
        
        # Convert to readable
        readable_inputs = {}
        for k, v in inputs.items():
            label = DISPLAY_NAMES.get(k, k)
            if k in MAPPINGS:
                readable_inputs[label] = MAPPINGS[k].get(v, v)
            else:
                readable_inputs[label] = v
        readable_inputs["📊 Data Source"] = data_source
        
        acc_labels = list(accuracies.keys())
        acc_values = list(accuracies.values())
        
        return render_template(
            "result.html",
            inputs=readable_inputs,
            percent=percent,
            details=details,
            chart_labels=chart_labels,
            chart_values=chart_values,
            acc_labels=acc_labels,
            acc_values=acc_values
        )
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        return f"<h1>Error: {str(e)}</h1><a href='/'>Back</a>", 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🫀 HEART DISEASE PREDICTION SYSTEM")
    print("="*60)
    print("🚀 Starting server...")
    print("📍 Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)