from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from nutrition import NUTRITION
from main import ensure_model_ready, predict_image

# Opretter Flask-app
app = Flask(__name__)

# Sikrer at modellen er klar
svm, label_map = ensure_model_ready()

# Mapping fra forskellige labels til vores nutrition-nøgler
CANON = {
    "apple": "apple",
    "æble": "apple",
    "banana": "banana",
    "banan": "banana",
    "orange": "orange",
    "appelsin": "orange",
}

# Forside render
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

#  billedgenkendelse og returnere næringsdata
@app.route("/predict", methods=["POST"])
def predict():
    # Tjek om der er uploadet en fil
    if "image" not in request.files:
        return jsonify({"error": "Ingen fil modtaget."}), 400

    file = request.files["image"]

    # Læs billedet fra hukommelsen
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Kunne ikke læse billedet."}), 400

    # Kør billedet igennem modellen for at få label
    label = predict_image(svm, label_map, img)
    key = CANON.get(label.lower(), label.lower())
    nutrition = NUTRITION.get(key)

    # Returner label og næringsdata som JSON
    return jsonify({
        "label": label,
        "nutrition": nutrition
    })

# Starter serveren hvis filen køres direkte
if __name__ == "__main__":
    app.run(debug=True)
