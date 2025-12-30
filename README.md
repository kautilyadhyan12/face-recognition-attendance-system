This project uses GPU-intensive models.Due to hardware limitations on 
free cloud platforms, live deployment is not provided.

A complete video demonstration and system walkthrough 
is available below.
demo: https://youtu.be/Hct65Vwb2tM 

# face-recognition-attendance-system
A modern, GPU-accelerated face recognition attendance system built using **RetinaFace** for face detection and **ArcFace** for face recognition. Designed with educational institutions in mind, the system supports **multi-role access** for Departments, Professors, and Students through a secure web dashboard.

Setup Guide:
python:3.10

Hardware:Webcam,8GB RAM,GPU

Create Virtual Environment:  1)python3.10 -m venv venv  
2)venv\Scripts\activate

Install Dependencies:pip install -r requirements.txt

Check GPU: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Start Server: python app.py
