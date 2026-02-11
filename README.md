1. Clone the repository
## Bash
# git clone 
# cd virtual-background-studio
## 2. Create and Activate Virtual Environment
## PowerShell
 # Windows
# py -3.11 -m venv venv
# .\venv\Scripts\activate
3. Install Dependencies
# PowerShell
# .\venv\Scripts\python.exe -m pip install mediapipe opencv-python streamlit numpy requests Pillow
# if meadiapipe is not working then try this 
## .\venv\Scripts\python.exe -m pip uninstall mediapipe -y

## .\venv\Scripts\python.exe -m pip install mediapipe==0.10.11
# Usage
# To launch the application, run the following command in your terminal:

# PowerShell

# .\venv\Scripts\python.exe -m streamlit run main.py

## How it Works
# The application uses a Semantic Segmentation model to predict a "mask" for every pixel in the image.

# Input: The image is converted to RGB.

# AI Inference: MediaPipe analyzes the frame and produces a probability mask (where 1.0 is the person and 0.0 is the background).

# Thresholding: We apply a user-defined threshold to create a clean binary mask.

# Compositing: Using numpy.where(), we combine the original pixels of the person with the pixels of the chosen background.