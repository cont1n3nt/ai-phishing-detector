from flask import Flask, request, jsonify, render_template
from utils.predict import predict_email
from utils.utils import validate_text, get_top_words, load_model

app = Flask(__name__)
MODEL = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    is_valid, result = validate_text(data)
    if not is_valid:
        return jsonify({"error": result}), 400

    text = result
    threshold = data.get("threshold", 0.4)
    prediction, probability = predict_email(text, threshold=threshold)
    top_words = get_top_words(MODEL, text)
    prediction_label = "PHISHING" if probability >= threshold else "LEGITIMATE"

    return jsonify({
        "prediction": prediction,
        "label": prediction_label,
        "probability": round(float(probability), 3),
        "top_words": top_words
    })

if __name__ == "__main__":
    app.run(debug=True)