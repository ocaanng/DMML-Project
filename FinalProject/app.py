from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Memuat model dan vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# Fungsi untuk memprediksi sentimen berdasarkan teks ulasan
def predict_sentiment(text):
    text = text.lower()  # Konversi teks menjadi huruf kecil
    text_vectorized = vectorizer.transform([text])  # Konversi teks menjadi fitur TF-IDF
    prediction = model.predict(text_vectorized)
    sentiment = 'positive' if prediction == 1 else 'negative'
    return sentiment

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json(force=True)
            review_text = data['review']
        else:
            review_text = request.form['review']
        
        predicted_sentiment = predict_sentiment(review_text)
        return jsonify({'sentiment': predicted_sentiment})
    else:
        return 'Only POST requests are allowed for this endpoint'
if __name__ == '__main__':
    app.run(debug=True)
