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
from picamera2 import Picamera2
import time

# API Gateway URL (replace with your actual URL)
API_URL = ' https://86ifarstkg.execute-api.us-east-1.amazonaws.com/FaceDetect/'

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'  # Use your SMTP server (e.g., Gmail)
SMTP_PORT = 587
EMAIL_ADDRESS = 'finalproject831@gmail.com'  # Sender email address
EMAIL_PASSWORD = 'iufryabbtpprizrc'  # Sender email password
target_email = 'galranel22@gmail.com'
target_email2= 'ofekpa@hotmail.com'
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

def download_face_images():
    # Create the download folder if it doesn't exist
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
    else:
        for file_name in os.listdir(DOWNLOAD_FOLDER):
            file_path = os.path.join(DOWNLOAD_FOLDER,file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    # Initialize S3 client
    s3 = boto3.client('s3')  # Credentials will be picked up from environment variables or AWS config

    try:
        # List all objects in the bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)

        # Check if the bucket is empty
        if 'Contents' not in response:
            print("No files found in the bucket.")
            return

        # Download files ending with .jpg
        for obj in response['Contents']:
            file_key = obj['Key']
            if file_key.lower().endswith('.jpg'):
                local_file_path = os.path.join(DOWNLOAD_FOLDER, os.path.basename(file_key))
                if not os.path.exists(local_file_path):
                    s3.download_file(BUCKET_NAME, file_key, local_file_path)
                    print(f"Downloaded: {file_key} -> {local_file_path}")
        print("All .jpg files downloaded successfully!")
    except Exception as e:
        print(f"Error: {e}")

def is_face_match(target_image_path, checked_image_path, threshold=0.6):
    """
    Checks if the face in the checked image matches the face in the target image.

    Parameters:
        target_image_path (str): Path to the target image.
        checked_image_path (str): Path to the image to be checked.
        threshold (float): Cosine distance threshold for determining a match.

    Returns:
        bool: True if the faces match, False otherwise.
    """
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
    """
    Detects if a face is present in the given video frame.

    Args:
        frame: The current video frame (image) from the camera.
        face_detector: The face detection model (e.g., Haar cascade).

    Returns:
        bool: True if a face is detected, False otherwise.
    """
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
    for file_name in os.listdir(DOWNLOAD_FOLDER):
        if file_name.lower().endswith(".jpg"): 
            result = is_face_match(image_path,f'{DOWNLOAD_FOLDER}//{file_name}')
            if(result == True):
                break
    if result == False:
        subject = 'UnAuthorized access'
        message = 'An attempt to make unauthorized entrance was done. Image of the person is attached'
        send_email(subject,message,None,image_path)

# Send the image and process the response
def main():
    download_face_images()
    print("Finish download images")
    # Initialize Picamera2
    picam2 = Picamera2()

    # Set up the camera configuration
    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)
    
    #SHOW VIDEO

    # Start the camera
    picam2.start()

    try:
        # Loop to continuously check for faces in the video feed
        while True:
            # Capture a frame from the camera
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
    
if __name__ == "__main__":
    main()
