import os
from flask import Flask, request, jsonify, render_template
from src.predict import predict_email
from src.features import validate_text

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        is_valid, result = validate_text(data)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        text = result
        threshold = data.get("threshold", 0.4)
        
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            return jsonify({"error": "Threshold must be a number between 0 and 1."}), 400
        
        result = predict_email(text, threshold=threshold)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Train the model first."}), 503
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500
    
    
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)