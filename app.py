from flask import Flask, request, render_template
import pickle


with open('spam_ham_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('feature_extractor.pkl', 'rb') as extractor_file:
    feature_extract = pickle.load(extractor_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        
        input_data = feature_extract.transform([message])
        input_data = input_data.toarray().reshape(1, -1)
        
        
        prediction = model.predict(input_data)[0]

        
        result = "Spam" if prediction == 1 else "Ham"
        
        return render_template('index.html', prediction_text=f'This  is {result} Email')

if __name__ == "__main__":
    app.run(debug=True)
