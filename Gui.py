
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Path to the classification model
classification_model_path = r"C:\Users\betti\Downloads\Mini_project_Image\pixel_classification_model.h5"

# Load classification model
classification_model = load_model(classification_model_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification")
        self.root.attributes('-fullscreen', True)
        
        # Background Image
        bg_image = Image.open(r"C:\Users\betti\Downloads\Mini_project_Image\6114100.jpg")
        bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label = tk.Label(root, image=bg_photo)
        self.bg_label.image = bg_photo
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Buttons
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Helvetica", 18))
        self.upload_btn.place(relx=0.02, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.classify_btn = tk.Button(root, text="Classify", command=self.classify_image, font=("Helvetica", 18))
        self.classify_btn.place(relx=0.24, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.correct_btn = tk.Button(root, text="Correct Image", command=self.correct_image, font=("Helvetica", 18))
        self.correct_btn.place(relx=0.46, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.close_btn = tk.Button(root, text="Close", command=root.quit, font=("Helvetica", 18))
        self.close_btn.place(relx=0.68, rely=0.02, relwidth=0.2, relheight=0.08)
        
        # Result Label
        self.result_label = tk.Label(root, text="", font=("Helvetica", 24), bg='white')
        self.result_label.place(relx=0.27, rely=0.12, relwidth=0.46, relheight=0.08)
        
        # Image Display Label
        self.image_label = tk.Label(root, bg='white')
        self.image_label.place(relx=0.27, rely=0.22, relwidth=0.71, relheight=0.76)
        
        self.image_path = None
        self.original_image = None
    
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            img = Image.open(self.image_path)
            self.original_image = img  # Save original image
            img.thumbnail((800, 600))  # Resize image to fit display
            img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img)
            self.image_label.image = img
            self.result_label.config(text="")  # Clear previous result
    
    def classify_image(self):
        if self.image_path:
            result = classify_image(self.image_path, classification_model)
            self.result_label.config(text=f"Classification: {result}")
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
    
    def correct_image(self):
        if self.image_path:
            if self.original_image:
                corrected_image = self.correct_pixelation(self.original_image)
                corrected_image.thumbnail((800, 600))  # Resize corrected image to fit display
                corrected_image = ImageTk.PhotoImage(corrected_image)
                self.image_label.configure(image=corrected_image)
                self.image_label.image = corrected_image
                self.result_label.config(text="Image Corrected")
            else:
                messagebox.showwarning("Warning", "Please upload an image first.")
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def correct_pixelation(self, img):
        # Apply a basic filter to reduce pixelation (for demonstration purposes)
        corrected_img = img.filter(ImageFilter.SMOOTH_MORE)
        return corrected_img

def classify_image(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]
    threshold = 0.8
    return "Pixelated" if prediction >= threshold else "High resolution"

def preprocess_image(image_path, img_height=224, img_width=224):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
