# import cv2
# import numpy as np
# import mediapipe as mp

# # Load the input image (ensure it’s in BGR format for OpenCV)
# input_path = "Input-Reference-2.png"
# image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
# if image is None:
#     raise FileNotFoundError(f"Could not load image at {input_path}")

# # If the image has an alpha channel, drop it for processing (we'll create our own later)
# if image.shape[2] == 4:
#     bgr = image[:, :, :3]
# else:
#     bgr = image

# # Initialize MediaPipe selfie segmentation (model_selection=1 for landscape model – higher quality)
# segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
# # Run segmentation to get the person mask
# result = segmentor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
# mask = result.segmentation_mask  # This is a 2D array of floats [0.0,1.0] with 1=person, 0=background
# segmentor.close()  # close the model to free resources

# # Convert the mask to a NumPy float array (if not already) and ensure same size as image
# mask = np.array(mask, dtype=np.float32)
# mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]))  # in case segmentation mask is scaled

# # **Remove the body below the face using face detection**
# face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
# face_results = face_detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
# face_detector.close()
# if face_results.detections:
#     # Assume the first detected face is our subject
#     det = face_results.detections[0]
#     bbox = det.location_data.relative_bounding_box
#     ih, iw, _ = bgr.shape
#     # Convert relative bbox to pixel coordinates
#     x = int(bbox.xmin * iw)
#     y = int(bbox.ymin * ih)
#     w = int(bbox.width * iw)
#     h = int(bbox.height * ih)
#     face_bottom = y + h  # bottom y-coordinate of face bounding box
#     # Define a cutoff line a bit below the face bottom to remove neck/shoulders
#     cutoff_y = int(face_bottom + 0.1 * h)  # 10% of face height below the face box
#     if cutoff_y < bgr.shape[0]:
#         mask[cutoff_y: , :] = 0.0  # set everything below this line to background (transparent)
# # If no face detected, we skip body removal (fallback: keep entire person mask)

# # **Refine mask edges for smoother transparency**
# # Apply a small Gaussian blur to soften edges (reduces jaggies and helps semi-transparency on hair edges)
# mask = cv2.GaussianBlur(mask, (5, 5), 0)

# # (Optional) You could also apply morphological ops here if needed, e.g.:
# # kernel = np.ones((3,3), np.uint8)
# # mask = cv2.erode(mask, kernel, iterations=1)
# # mask = cv2.dilate(mask, kernel, iterations=1)

# # **Prepare the RGBA output image**
# # Create an alpha channel from the mask (scale 0-1 to 0-255)
# alpha = (mask * 255).astype(np.uint8)
# # Apply the mask to the original BGR image to remove background color from translucent regions
# # We multiply each color channel by the mask (which is 0 to 1 float) to black-out the background
# foreground = (bgr.astype(np.float32) * mask[..., np.newaxis]).astype(np.uint8)

# # Combine BGR and alpha into a 4-channel BGRA image
# output_rgba = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)  # start with BGR and add empty alpha channel
# output_rgba[:, :, 3] = alpha  # set the alpha channel

# # Save the result as a PNG (which supports transparency)
# cv2.imwrite("output.png", output_rgba)
# print("Saved segmented image to output.png")




# import cv2
# import numpy as np
# import mediapipe as mp

# # Load the input image
# input_path = "Input-Reference-2.png"
# image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
# if image is None:
#     raise FileNotFoundError(f"Could not load image at {input_path}")

# # Ensure image is in BGR format
# if image.shape[2] == 4:
#     bgr = image[:, :, :3]  # Drop existing alpha channel
# else:
#     bgr = image

# # Initialize MediaPipe Selfie Segmentation
# segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
# result = segmentor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
# segmentor.close()

# # Convert segmentation mask to NumPy format
# mask = np.array(result.segmentation_mask, dtype=np.float32)
# mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

# # Convert mask to binary: Foreground (face + hair) = 255, Background = 0
# mask = (mask > 0.5).astype(np.uint8) * 255

# # ---- STEP 1: Remove Everything Below the Face ----
# face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
# face_results = face_detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
# face_detector.close()

# if face_results.detections:
#     # Assume first detected face is the subject
#     det = face_results.detections[0]
#     bbox = det.location_data.relative_bounding_box
#     ih, iw, _ = bgr.shape
#     x = int(bbox.xmin * iw)
#     y = int(bbox.ymin * ih)
#     w = int(bbox.width * iw)
#     h = int(bbox.height * ih)
#     face_bottom = y + h  # Bottom of face

#     # Define a cutoff line slightly below the chin
#     cutoff_y = int(face_bottom + 0.08 * h)  # Remove 8% more below face
#     mask[cutoff_y:, :] = 0  # Set everything below face to transparent (0)

# # ---- STEP 2: Smooth and Feather the Mask ----
# # Apply a slight Gaussian blur to smooth edges
# mask = cv2.GaussianBlur(mask, (5, 5), 0)

# # Use morphology operations to refine edges further
# kernel = np.ones((3, 3), np.uint8)
# mask = cv2.erode(mask, kernel, iterations=1)
# mask = cv2.dilate(mask, kernel, iterations=1)

# # ---- STEP 3: Preserve Hair Details and Blend Transparency ----
# # Convert binary mask into a softer alpha mask for better blending
# alpha = (cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)).astype(np.uint8)

# # Ensure hair region is slightly translucent to preserve details
# alpha = np.clip(alpha, 50, 255)  # Avoid fully transparent edges in hair

# # ---- STEP 4: Create Final RGBA Image ----
# foreground = (bgr.astype(np.float32) * (alpha[..., np.newaxis] / 255.0)).astype(np.uint8)
# output_rgba = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
# output_rgba[:, :, 3] = alpha  # Set alpha channel

# # Save the output as a transparent PNG
# output_path = "output.png"
# cv2.imwrite(output_path, output_rgba)

# print(f"Saved segmented image to {output_path}")


















































import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from rembg import remove

# *** 1. LOAD INPUT IMAGE ***
input_path = "Input-Reference-2.png"  # <-- replace with your input image path
output_path = "output.png"
# Load image using PIL and convert to RGB
input_pil = Image.open(input_path).convert("RGB")
orig_np = np.array(input_pil)  # RGB image as numpy array
height, width = orig_np.shape[0], orig_np.shape[1]

# *** 2. MEDIAPIPE SEGMENTATION (Selfie Segmentation) ***
mp_selfie = mp.solutions.selfie_segmentation
with mp_selfie.SelfieSegmentation(model_selection=1) as segmentor:
    # MediaPipe expects RGB uint8 image
    results = segmentor.process(orig_np)
    mask = results.segmentation_mask  # float32 2D array
# Ensure mask matches original size (MediaPipe may use 256x256 internally)
if mask.shape[0] != height or mask.shape[1] != width:
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
mask = mask.astype('float32')

# *** 3. MEDIAPIPE FACE DETECTION ***
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_results = face_detector.process(orig_np)
face_box = None
if face_results.detections:
    # Take the first detected face
    det = face_results.detections[0]
    bb = det.location_data.relative_bounding_box
    # Convert relative coords to absolute pixel values
    x, y, w, h = bb.xmin, bb.ymin, bb.width, bb.height
    face_box = (
        int(x * width),
        int(y * height),
        int((x + w) * width),
        int((y + h) * height)
    )
# (face_box is (x_min, y_min, x_max, y_max) in pixel coordinates if a face is detected)

# *** 4. BACKGROUND COLOR ESTIMATION (for de-haloing) ***
# Use the segmentation mask to find definite background regions.
# We'll take pixels with mask value very low (e.g., < 0.1) as background.
bg_mask = mask < 0.1
if np.any(bg_mask):
    # Compute average color of those background pixels from the original image
    # orig_np is RGB; take mean along first two axes
    background_color = orig_np[bg_mask].mean(axis=0)
else:
    # Fallback: if no sure background found, assume a default neutral background (black)
    background_color = np.array([0.0, 0.0, 0.0])
background_color = background_color.astype(np.float32)

# *** 5. U²-Net (rembg) BACKGROUND REMOVAL ***
# Perform background removal with U²-Net model using rembg, enabling alpha matting for quality.
# This returns a PIL Image with an alpha channel.
output_pil = remove(input_pil, alpha_matting=True, 
                    alpha_matting_foreground_threshold=270,
                    alpha_matting_background_threshold=20,
                    alpha_matting_erode_size=11)
# Convert result to numpy array (RGBA)
output_np = np.array(output_pil)
# Split channels
alpha = output_np[:, :, 3].astype(np.float32) / 255.0  # Alpha channel as 0.0-1.0
rgb = output_np[:, :, :3].astype(np.float32)           # RGB color channels

# *** 6. REFINE ALPHA MASK (FACE AND SHOULDERS) ***
# a. Ensure face region is fully opaque
if face_box:
    x_min, y_min, x_max, y_max = face_box
    # Slightly expand the face box by 5% for safety (include hairline or ears)
    y_expansion = int(0.05 * (y_max - y_min))
    x_expansion = int(0.05 * (x_max - x_min))
    y1 = max(0, y_min - y_expansion)
    y2 = min(height, y_max + y_expansion)
    x1 = max(0, x_min - x_expansion)
    x2 = min(width, x_max + x_expansion)
    alpha[y1:y2, x1:x2] = 1.0  # set face & surrounding area to opaque

# b. Determine cutoff for shoulders (neck line)
cutoff_y = None
if face_box:
    # Use face_box bottom plus some margin
    _, _, _, face_y_max = face_box
    # Start at bottom of face, search downwards for neck (min mask width)
    start_y = face_y_max
    end_y = min(height, face_y_max + int((face_y_max - y_min) * 0.5) if face_box else height)
    # Compute horizontal span of person mask at each row (using MediaPipe mask)
    # Use a binary mask with a high threshold to focus on definite person region
    person_bin = mask > 0.5
    min_width = width + 1
    neck_y = start_y
    for yy in range(start_y, end_y):
        row_width = np.count_nonzero(person_bin[yy, :])
        if 0 < row_width < min_width:
            min_width = row_width
            neck_y = yy
    cutoff_y = neck_y
else:
    # If no face detected, default cutoff: e.g., 70% of mask height from top
    fg_rows = np.any(mask > 0.5, axis=1)
    y_indices = np.where(fg_rows)[0]
    if y_indices.size > 0:
        top_y, bottom_y = y_indices[0], y_indices[-1]
        cutoff_y = int(top_y + 0.7 * (bottom_y - top_y))
# If determined, apply cutoff
if cutoff_y is not None:
    cutoff_y = min(height - 1, cutoff_y)
    # Feather size (e.g., 10 pixels)
    feather = 10
    for yy in range(cutoff_y, height):
        if yy < cutoff_y + feather:
            # Linearly interpolate alpha from 1 at cutoff_y to 0 at cutoff_y+feather
            alpha_factor = 1.0 - float(yy - cutoff_y + 1) / float(feather)
            alpha_factor = np.clip(alpha_factor, 0.0, 1.0)
            # Multiply existing alpha by this factor to taper it off
            alpha[yy, :] *= alpha_factor
        else:
            # Fully transparent beyond the feather region
            alpha[yy, :] = 0.0

# c. Clip alpha to [0,1] just in case
alpha = np.clip(alpha, 0.0, 1.0)

# *** 7. COLOR DECONTAMINATION (remove halos) ***
# Adjust RGB colors where alpha is not 0 or 1 to remove background contribution.
# new_rgb = original_rgb - background_color * (1 - alpha)
# We apply this per channel.
# Prepare background color arrays for subtraction
bg_r, bg_g, bg_h = background_color  # (Using h instead of b to avoid confusion with blue)
# original RGB as float (from orig_np)
orig_float = orig_np.astype(np.float32)
# Compute new RGB values
# For each channel: new = orig - bg_color * (1 - alpha)
new_R = orig_float[:, :, 0] - bg_r * (1.0 - alpha)
new_G = orig_float[:, :, 1] - bg_g * (1.0 - alpha)
new_B = orig_float[:, :, 2] - bg_h * (1.0 - alpha)
# Clip to valid range [0, 255]
new_R = np.clip(new_R, 0, 255)
new_G = np.clip(new_G, 0, 255)
new_B = np.clip(new_B, 0, 255)
# Combine channels back and convert to uint8
new_rgb = np.dstack((new_R, new_G, new_B)).astype(np.uint8)
# Also convert alpha back to 0-255 uint8
alpha_out = (alpha * 255).astype(np.uint8)
# Compose final output image (BGRA for OpenCV or RGBA for PIL)
output_bgra = np.dstack((new_rgb[:, :, 2], new_rgb[:, :, 1], new_rgb[:, :, 0], alpha_out))

# *** 8. PREVIEW RESULT ***
# Create a preview by compositing the output onto a solid background (e.g., gray)
preview_bg_color = (192, 192, 192)  # light gray for contrast
# Create a full background image
background_layer = np.zeros_like(new_rgb)
background_layer[:, :, 0] = preview_bg_color[0]
background_layer[:, :, 1] = preview_bg_color[1]
background_layer[:, :, 2] = preview_bg_color[2]
# Normalize alpha for blending
alpha_f = alpha[..., None]  # shape (H,W,1)
# Composite: out_rgb * alpha + bg * (1 - alpha)
preview_rgb = (new_rgb.astype(float) * alpha_f + background_layer.astype(float) * (1 - alpha_f))
preview_rgb = preview_rgb.astype(np.uint8)
# Show preview window
cv2.imshow("Segmentation Preview", preview_rgb)
cv2.waitKey(0)  # wait for key press to close
cv2.destroyAllWindows()

# *** 9. SAVE OUTPUT PNG ***
cv2.imwrite(output_path, output_bgra)
print(f"Saved output to {output_path}")
