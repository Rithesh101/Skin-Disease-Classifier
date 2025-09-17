from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("my_model.keras")


classes = {
    4: ('nv', 'melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', 'basal cell carcinoma'),
    5: ('vasc', 'pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28))  
    img_array = image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        
        if confidence < 0.6:  
            result = "Uncertain / No clear disease detected. Please consult a doctor."
        else:
            result = f"{classes[predicted_class][1]} ({round(confidence*100,2)}%)"
        image_path = file_path  

    return render_template("index.html", result=result, image_path=image_path)


@app.route("/disease-info")
def disease_info():
    return render_template("disease_info.html")


if __name__ == "__main__":
    app.run(debug=True)
