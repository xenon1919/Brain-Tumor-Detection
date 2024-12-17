import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define class labels
class_labels = ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary Tumor']

# Load the trained model
model_path = 'braintumor.h5'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to load and preprocess the image
def load_image(img_path):
    try:
        img = Image.open(img_path).resize((224, 224))  # Resize as per your model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Function to make prediction
def predict(img_path):
    img_array = load_image(img_path)
    if img_array is not None:
        a = model.predict(img_array)
        indices = a.argmax()
        predicted_tumor = class_labels[indices]
        result_label.config(text=f'Predicted Tumor Type: {predicted_tumor}')
    else:
        result_label.config(text="Error in image processing. Please try again.")

# Function to open a file dialog and select an image
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = ImageTk.PhotoImage(img.resize((250, 250)))  # Resize for display in the GUI
        panel.config(image=img)
        panel.image = img
        predict(file_path)

# Create the main window
root = tk.Tk()
root.title("Brain Tumor Detection")

# Create a button to open the file dialog
btn = tk.Button(root, text="Select Image", command=open_file)
btn.pack(pady=10)

# Create a label to display the selected image
panel = Label(root)
panel.pack(pady=10)

# Create a label to display the prediction result
result_label = Label(root, text="Predicted Tumor Type: None")
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
