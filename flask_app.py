import os
from flask import render_template, Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#model = tf.keras.models.load_model('Leaves_Classification_Model.h5', compile=False) 

class_names = ['cat', 'dog']
@app.route("/", methods = ["POST", "GET"])
def predict_image():
    if request.method == 'POST':
        image_file = request.files["file"]
        if image_file and allowed_file(image_file.filename):
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)


            from keras.models import load_model
            new_model = load_model('dog_cat_classification_model.h5', compile = False)
            #new_model.summary()
            test_image = image.load_img('static/images/'+image_file.filename,target_size=(180,180))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            predictions = new_model.predict(test_image)
            score = tf.nn.softmax(predictions[0])
            
            #class_names[np.argmax(score)], 100 * np.max(score)

            return render_template("index.html", text = class_names[np.argmax(score)], prediction = (100 * np.max(score)) ,image_loc = image_file.filename)

    return render_template("index.html", prediction = 0, image_loc=None)

@app.route("/model.html", methods=["GET", "POST"])
def model():
    return render_template('model.html')

if __name__ == '__main__':
    app.run(port = 5000, debug = True)


