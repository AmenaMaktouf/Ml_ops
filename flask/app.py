from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# URL de l'API FastAPI
FASTAPI_URL = "http://localhost:8000/predict/"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  # initialisation de la variable
    if request.method == "POST":
        input_data = {
            "account_length": float(request.form["account_length"]),
            "international_plan": int(request.form["international_plan"]),
            "voice_mail_plan": int(request.form["voice_mail_plan"]),
            "number_vmail_messages": float(request.form["number_vmail_messages"]),
            "total_day_minutes": float(request.form["total_day_minutes"]),
            "total_day_calls": float(request.form["total_day_calls"]),
            "total_day_charge": float(request.form["total_day_charge"]),
            "total_eve_minutes": float(request.form["total_eve_minutes"]),
            "total_eve_calls": float(request.form["total_eve_calls"]),
            "total_night_minutes": float(request.form["total_night_minutes"]),
            "total_night_calls": float(request.form["total_night_calls"]),
            "total_intl_minutes": float(request.form["total_intl_minutes"]),
            "total_intl_calls": float(request.form["total_intl_calls"]),
            "customer_service_calls": float(request.form["customer_service_calls"]),
        }

        # Envoi des données à FastAPI pour obtenir la prédiction
        response = requests.post(FASTAPI_URL, json=input_data)
        try:
            prediction = response.json().get("prediction", "Erreur dans la prédiction")
        except Exception as e:
            prediction = f"Erreur lors de l'analyse de la réponse : {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7000)
