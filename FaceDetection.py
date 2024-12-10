import requests
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage  
from email import encoders
import boto3
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
import numpy as np
import cv2
#from picamera2 import Picamera2
import time
import shutil


# Email configuration
SMTP_SERVER = 'smtp.gmail.com'  
SMTP_PORT = 587
EMAIL_ADDRESS = 'finalproject831@gmail.com'  # Sender email address
EMAIL_PASSWORD = 'nlfnkgfhujelykmm'  # Sender email password
target_email = 'galranel22@gmail.com'
#target_email= 'ofekpa@hotmail.com'
AWS_ACCESS_KEY = 'AKIA4RCAOF3EJLMIPLKF'
AWS_SECRET_KEY = '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'
BUCKET_NAME = "ofekpa"
DOWNLOAD_FOLDER = "downloaded_images"

# Path to the input image
image_path = 'Capture.JPG'

def send_email(subject, body, attachment_path=None, inline_image_path=None):
    try:
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = target_email
        msg['Subject'] = subject

        # Add the email body
        msg.attach(MIMEText(body, 'plain'))

        # Attach the file (if provided)
        if attachment_path:
            attachment = MIMEBase('application', 'octet-stream')
            with open(attachment_path, 'rb') as f:
                attachment.set_payload(f.read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename="{attachment_path.split("/")[-1]}"')
            msg.attach(attachment)

        # Attach the inline image (if provided)
        if inline_image_path:
            with open(inline_image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-ID', '<inline_image>')
                img.add_header('Content-Disposition', 'inline', filename=inline_image_path.split("/")[-1])
                msg.attach(img)

        # Connect to SMTP server and send the email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, target_email, msg.as_string())
        server.quit()
        print("Email sent successfully.")

    except Exception as e:
        print(f"Error sending email: {e}")


def process_image(image_path):
    try:
        # Read and encode the image
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')

        # Send the image to the Lambda function
        response = requests.post(api_url, json={'body': encoded_image})
        response_data = response.json()
        print("Response from Lambda:", response_data)

        # Process the server response
        if response_data.get('match') is True:
            # Send success email
            send_email(
                subject="Gate Access Granted",
                body="A valid person crossed the gate."
            )
        else:
            # Send failure email with image attached and inline
            send_email(
                subject="Unauthorized Gate Access Attempt",
                body="An unauthorized person tried to cross the gate. See the attached image.",
                attachment_path=image_path,
                inline_image_path=image_path
            )
    except Exception as e:
        print(f"Error processing image: {e}")

def clear_local_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    

def download_face_images():
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME)

    folders = set()
    files_downloaded = []

    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(".jpg"):
                    # Extract folder name and file name
                    folder_name = os.path.dirname(key)
                    file_name = os.path.basename(key)

                    # Ensure local folder exists
                    local_folder_path = os.path.join(DOWNLOAD_FOLDER, folder_name)
                    os.makedirs(local_folder_path, exist_ok=True)
                    folders.add(local_folder_path)

                    # Download file
                    local_file_path = os.path.join(local_folder_path, file_name)
                    s3.download_file(BUCKET_NAME, key, local_file_path)
                    files_downloaded.append(local_file_path)

    return list(folders), files_downloaded



def is_face_match(target_image_path, checked_image_path, threshold=0.35):

    try:
        # Extract embeddings for the target image
        target_embedding_result = DeepFace.represent(img_path=target_image_path, model_name='Facenet', enforce_detection=False)
        if not isinstance(target_embedding_result, list) or len(target_embedding_result) == 0:
            print(f"Failed to extract embedding for target image: {target_image_path}")
            return False
        target_embedding = np.array(target_embedding_result[0]['embedding'])

        # Extract embeddings for the checked image
        checked_embedding_result = DeepFace.represent(img_path=checked_image_path, model_name='Facenet', enforce_detection=False)
        if not isinstance(checked_embedding_result, list) or len(checked_embedding_result) == 0:
            print(f"Failed to extract embedding for checked image: {checked_image_path}")
            return False
        checked_embedding = np.array(checked_embedding_result[0]['embedding'])

        # Compute cosine distance between the embeddings
        distance = cosine(target_embedding, checked_embedding)

        # Determine if the distance is below the threshold
        is_match = distance < threshold
        print(f"Distance: {distance:.4f}, Match: {is_match}")
        return is_match

    except Exception as e:
        print(f"Error processing images: {e}")
        return False

def detect_face_in_video(frame):

    if frame is None:
        print("Image is empty!!")
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("FILE IS EMPTY!!")
        return False
    faces = face_cascade.detectMultiScale(gray, 1.1,4)
    return len(faces) > 0
   
def face_detection():
    result = False
    print("Start compare with authorized photos:")
    for folder in folders:
        folder_name = os.path.basename(folder)  # Get the folder name
        print(f"Processing folder: {folder_name}")
        for file in os.listdir(folder):
            if file.endswith(".jpg"):
                file_path = os.path.join(folder, file)
                print(f"Processing file: {file_path} in folder: {folder_name}")
                result = is_face_match(image_path,file_path)
                if(result == True):
                    print(f'Authorized Entrance has been made by {folder_name}')
                break
    if result == False:
        print("UnAuthorized entrance was tried. Send Email with details")
        subject = 'UnAuthorized access'
        message = 'An attempt to make unauthorized entrance was done. Image of the person is attached'
        send_email(subject,message,None,image_path)

# Send the image and process the response
def main():
    download_face_images()
    print("Finish download images")
    # Initialize Picamera2
    os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
    picam2 = Picamera2()

    # Set up the camera configuration
    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)
   
    #SHOW VIDEO

    # Start the camera
    picam2.start()

    try:
        # Loop to continuously check for faces in the video feed
        print("start photo for face detection...")
        while True:
            # Capture a frame from the camera
            os.remove(image_path)
            picam2.start_and_capture_file(image_path)
            frame = cv2.imread(image_path)
           
            #show the
            #cv2.imshow("Video",frame)
            # Call the function to detect faces in the frame
            if detect_face_in_video(frame):
                print("Face detected!")
                face_detection() #COMPARE PHOTO TO Authorized photos
            #else:
               # print("No face detected.")
            time.sleep(0.5)
            #cv2.destroyWindow("Video")
    except KeyboardInterrupt:
        # Stop the camera gracefully when interrupted
        picam2.close()
        picam2.stop()
   
