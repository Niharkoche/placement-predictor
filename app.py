from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Train ML model
data = pd.read_csv("placement.csv")

X = data[["cgpa"]]
y = data["package"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Creating Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            cgpa = float(request.form["cgpa"])
            pred = model.predict([[cgpa]])[0]
            prediction = round(pred, 2)
        except:
            prediction = "Invalid Input"
    return render_template("index.html", prediction=prediction)

# ==============================
# 3. Run server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
