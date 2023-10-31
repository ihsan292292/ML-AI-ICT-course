# Library imports
from flask import Flask, request, render_template
import cv2
import numpy as np
from keras.models import load_model
import tensorflow.keras.backend as K
import json
import base64

# Create the app object
app = Flask(__name__)

with open("character_encoding.json", "r") as json_file:
    char_list = json.load(json_file)
json_file.close()

saved_model = load_model("full_model.h5")

def process_image(img): #Converts image to shape (32, 128, 1) & normalize
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255
    return img

def predict_image(img):
    pred = saved_model.predict(np.expand_dims(img, axis=0))
    out = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
    letters = ''
    for x in out[0]:
        if int(x) != -1:
            letters += char_list[int(x)]
    return letters

# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    result = ""
    try:
        # Get the uploaded image data from the request
        image_data = request.files['image'].read()
        image_data_bs = base64.b64encode(image_data).decode('utf-8')

        # Convert the image data to a NumPy array
        nparr = np.fromstring(image_data, np.uint8)
        
        # Decode the image array using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        #print("image :", image.shape)
        if image is not None:
            processed_image = process_image(image)
            result = "Predicted word : " + str(predict_image(processed_image))
        else:
            result = "No image found"
    except Exception as e:
        result = "An error occured"
    return render_template('result.html', prediction_text=result, image_data_web=image_data_bs)

if __name__ == "__main__":
    app.run(debug=True)
