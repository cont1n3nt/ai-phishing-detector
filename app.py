from flask import Flask, request, jsonify, render_template
from src.predict import predict_email
from src.features import validate_text

app = Flask(__name__)

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
    result = predict_email(text, threshold=threshold)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)