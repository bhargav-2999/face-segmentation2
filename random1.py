import cv2
import numpy as np
from tensorflow.keras.models import load_model

def validate_face(image_path):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Validate the number of faces detected
    if len(faces) != 1:
        return False, None
    return True, faces[0]

def segment_face(image_path, model):
    # Load the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))  # Resize to model input size
    img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize
    
    # Perform segmentation
    segmented = model.predict(img_array)
    segmented = (segmented[0] > 0.5).astype(np.uint8)  # Binarize the output
    
    # Create a transparent background
    output = np.zeros((*segmented.shape[0:2], 4), dtype=np.uint8)  # RGBA
    output[..., :3] = img_resized * segmented[..., np.newaxis]  # Apply mask
    output[..., 3] = segmented * 255  # Alpha channel
    
    return output

# Load your pre-trained segmentation model
model = load_model('path_to_your_model.h5')

# Example usage
image_path = 'path_to_input_image.jpg'
is_valid, face_coords = validate_face(image_path)

if is_valid:
    segmented_face = segment_face(image_path, model)
    cv2.imwrite('segmented_face.png', segmented_face)
else:
    print("Error: Please upload an image with exactly one face.")
