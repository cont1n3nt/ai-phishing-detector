import joblib

model = joblib.load("../model/pipeline.joblib")

def predict_email(text: str):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0][1]
    return pred, proba

if __name__ == "__main__":
    test_email = "Verify your account now by clicking the link"
    pred, proba = predict_email(test_email)
    
    print("Prediction:", pred)
    print("Phishing probability:", round(proba,3))