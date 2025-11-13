from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            s_len = float(request.form.get('s_len'))
            s_width = float(request.form.get('s_width'))

            p_len = float(request.form.get('p_len'))
            p_width = float(request.form.get('p_width'))
            
            input_data = np.array([[s_len, s_width, p_len, p_width]])
            X_scaled = scaler.transform(input_data)

            pred = model.predict(X_scaled)
            print(pred)
            return render_template("index.html", result=f"Predicted Flower Species: {pred}")
        except Exception as e:
            return render_template("index.html", result = f"Error: {e}")
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)