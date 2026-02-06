from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model/pipeline.joblib")

@app.route("/")
def home():
    return "Hello, AI Phishing Detector is running, model loaded successfully"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if "text" not in data:
        return jsonify({"error": "No text"}), 400
    
    text = data["text"]
    if not data["text"].strip():
        return jsonify({"error": "Text is empty"}), 400
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0][1]
    
    return jsonify({
        "prediction": int(pred),
        "probability": round(float(proba), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)