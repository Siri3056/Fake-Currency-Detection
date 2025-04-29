from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from tensorflow_docs.vis import embed
from tensorflow import keras
#from imutils import paths
# import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array, load_img
from flask import Flask,render_template,session,flash,redirect,request,send_from_directory,url_for
import mysql.connector, os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import time

# Load the pre-trained model
model = keras.models.load_model("mobilenet.keras")

# Load and prepare the image
def prepare_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Load the image
    img_array = img_to_array(img)  # Convert the image to numpy array
    img_array = img_array / 255.0  # Scale the image (if your model requires normalization)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='fake_indian_currency'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('login.html', message="This email ID is already exists!")
        return render_template('login.html', message="Conform password is not match!")
    return render_template('login.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')





@app.route('/home')
def home():
    return render_template('home.html')



@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        file_extension = fn.split('.')[-1].lower()

        accepted_formats = ['jpg', 'png', 'jpeg', 'jfif']
        if file_extension not in accepted_formats:
            flash("Image formats only Accepted", "danger")
            return render_template("upload.html")

        mypath = os.path.join('static/uploaded_images', fn)
        myfile.save(mypath)

        rev = relevent_finder(mypath)
        if rev != "relevent":
            prediction = "This image is not relevant to this project! Please provide a valid Indian Currency image."
            return render_template("upload.html", mypath = mypath, prediction=prediction)

        new_model = load_model(r"mobilenet.keras")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        classes = ['Fake', 'Real']
        prediction = classes[np.argmax(result)]
        print(prediction)

        return render_template("upload.html", mypath = mypath, prediction=prediction)
    
    return render_template('upload.html')




def relevent_finder(img):

    import torch
    from torchvision import transforms
    from PIL import Image
    import torch.nn as nn
    from torchvision import models
    import os

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the model class (same as the one used during training)
    class MobileNetModel(nn.Module):
        def __init__(self, num_classes):
            super(MobileNetModel, self).__init__()
            self.mobilenet = models.mobilenet_v2(pretrained=True)
            num_features = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

        def forward(self, x):
            return self.mobilenet(x)

    # Load the trained model
    model = MobileNetModel(num_classes=2)
    model.load_state_dict(torch.load(r"mobilenet.pt"))
    model = model.to(device)
    model.eval()

    def predict_image(image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = image_transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Perform the prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    # Helper function to map the prediction to label
    def map_prediction_to_label(prediction):
        label_mapping = {0: "relevent", 1: "irrelevent"}
        return label_mapping.get(prediction, "Unknown")

    # Example usage
    image_path = img
    prediction = predict_image(image_path)
    predicted_label = map_prediction_to_label(prediction)
    print(55555555555555555, predicted_label)

    return predicted_label

if __name__ == '__main__':
    app.run(debug = True)

    