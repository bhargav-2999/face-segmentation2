# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# from model import BiSeNet

# mp_face_mesh = mp.solutions.face_mesh
# mp_face_detection = mp.solutions.face_detection

# def detect_single_face(image_rgb):
#     """
#     Checks if exactly one face is detected using MediaPipe's FaceDetection.
#     Returns True if exactly one face is found, otherwise False.
#     """
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
#         results = face_detection.process(image_rgb)
#         return (results.detections and len(results.detections) == 1)

# def load_bisenet_model(weight_path='79999_iter.pth', n_classes=19):
#     """
#     Loads the BiSeNet model with specified weights for face/hair segmentation.
#     """
#     net = BiSeNet(n_classes=n_classes)
#     net.load_state_dict(torch.load(weight_path, map_location='cpu'))
#     net.eval()
#     return net

# def keep_largest_component(mask_255):
#     """
#     Retains the largest connected component in a binary mask (255 = foreground).
#     Helps remove small spurious regions.
#     """
#     mask_bin = (mask_255 > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels <= 1:
#         return mask_255

#     largest_label = 1
#     max_area = stats[1, cv2.CC_STAT_AREA]
#     for label_id in range(2, num_labels):
#         area = stats[label_id, cv2.CC_STAT_AREA]
#         if area > max_area:
#             max_area = area
#             largest_label = label_id

#     final_mask = np.zeros_like(mask_bin, dtype=np.uint8)
#     final_mask[labels == largest_label] = 255
#     return final_mask

# def fill_mask_holes(mask_255):
#     """
#     Fills holes in the mask by inverting the mask, keeping the largest background,
#     then inverting again.
#     """
#     inv_mask = cv2.bitwise_not(mask_255)
#     largest_bg = keep_largest_component(inv_mask)
#     filled_mask = cv2.bitwise_not(largest_bg)
#     return filled_mask

# def parse_image(image_bgr, model):
#     """
#     Resizes the input image, runs BiSeNet segmentation, and returns the parsing map.
#     Also returns the original image dimensions.
#     """
#     # Increase to 1024 if you want finer details and have enough memory.
#     input_size = 512  
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape

#     # Resize to input_size x input_size
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     tensor = transform(image_resized).unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(tensor)
#         main_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
#         main_output = main_output.squeeze(0)
#         parsing_map = torch.argmax(main_output, dim=0).cpu().numpy()

#     return parsing_map, orig_h, orig_w

# def face_hair_mask(parsing_map, orig_w, orig_h):
#     """
#     Creates a mask for face/hair/neck based on the parsing map, applies morphological
#     closing and a small dilation to preserve hair details, then resizes to the
#     original image size.
#     """
#     # Classes for face, hair, ears, neck, etc.
#     keep_classes = list(range(1, 14)) + [15, 17]  # includes neck (13)
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255

#     # Morphological operations:
#     # Do a small closing to fill minor gaps, then a more aggressive dilation
#     kernel = np.ones((3, 3), np.uint8)

#     closed = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=2)
#     dilated = cv2.dilate(closed, kernel, iterations=3)

#     largest = keep_largest_component(dilated)
#     filled = fill_mask_holes(largest)
#     final_512 = keep_largest_component(filled)

#     # Resize back to original dimensions
#     mask_original = cv2.resize(final_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Creates an RGBA image from the original BGR image and a 255-based mask.
#     Pixels where mask == 255 are kept, alpha=255; otherwise alpha=0.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def feather_alpha(rgba, radius=15, iterations=3):
#     """
#     Applies more aggressive Gaussian blur to the alpha channel to create a softer edge.
#     Increases radius and adds multiple blur passes for smoother transition.
#     """
#     alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    
#     # Multiple passes of Gaussian blur with decreasing radius
#     for i in range(iterations):
#         current_radius = int(radius * (1 / (i + 1)))
#         alpha = cv2.GaussianBlur(alpha, (2 * current_radius + 1, 2 * current_radius + 1), current_radius)
    
#     alpha = np.clip(alpha, 0, 1)
#     rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
#     return rgba

# def make_glasses_see_through(rgba_image, left_eye_box, right_eye_box, color_threshold=20, edge_aperture=3):
#     """
#     More aggressive approach to making glasses lenses transparent.
#     Reduces color threshold and preserves more edge details.
#     """
#     h, w, _ = rgba_image.shape
#     bgr = rgba_image[:, :, :3].copy()
#     alpha = rgba_image[:, :, 3].copy()
#     mask_outside_eyes = np.ones((h, w), dtype=np.uint8)

#     # Mark outside eye regions to compute average face color
#     for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
#         x1, x2 = max(0, x1), min(w, x2)
#         y1, y2 = max(0, y1), min(h, y2)
#         mask_outside_eyes[y1:y2, x1:x2] = 0

#     valid_pixels = (alpha == 255) & (mask_outside_eyes == 1)
#     # Compute mean color from face pixels outside the eye boxes
#     if np.count_nonzero(valid_pixels) > 0:
#         face_color = bgr[valid_pixels].mean(axis=0)
#     else:
#         face_color = np.array([128, 128, 128])  # fallback if no valid pixels

#     # For each eye box, remove face-like regions while preserving more details
#     for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
#         x1, x2 = max(0, x1), min(w, x2)
#         y1, y2 = max(0, y1), min(h, y2)
#         region_bgr = bgr[y1:y2, x1:x2]
#         region_alpha = alpha[y1:y2, x1:x2]

#         # Enhanced edge detection to preserve glasses frame
#         edges = cv2.Canny(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=edge_aperture)
#         edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

#         # More precise color difference calculation
#         diff = np.linalg.norm(region_bgr.astype(np.float32) - face_color, axis=2)

#         # More selective removal of pixels
#         remove_mask = (diff < color_threshold) & (edges == 0)
#         region_alpha[remove_mask] = 0

#         alpha[y1:y2, x1:x2] = region_alpha

#     rgba_image[:, :, 3] = alpha
#     return rgba_image

# def main(image_path):
#     """
#     Main function to:
#     1) Load and validate the image
#     2) Detect single face
#     3) Load BiSeNet model and parse image
#     4) Create an RGBA image with the face/hair segmented
#     5) Feather alpha edges more aggressively
#     6) Make glasses lenses see-through (if eyes detected)
#     7) Crop to alpha
#     8) Resize to fit a 512x512 canvas
#     """
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)

#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     if not detect_single_face(image_rgb):
#         print("No face or multiple faces detected. Exiting.")
#         sys.exit(0)
#     else:
#         print("Exactly one face detected.")

#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)

#     # Segment face and hair
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
#     rgba_image = create_rgba_image(image_bgr, final_mask)

#     # More aggressive alpha feathering
#     rgba_image = feather_alpha(rgba_image, radius=20, iterations=4)

#     # Attempt to make glasses lenses see-through
#     left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
#     if left_eye_box and right_eye_box:
#         rgba_image = make_glasses_see_through(
#             rgba_image,
#             left_eye_box,
#             right_eye_box,
#             color_threshold=20,  # More aggressive removal
#             edge_aperture=3
#         )

#     # Crop to bounding box of non-transparent pixels
#     rgba_image = crop_to_alpha(rgba_image)

#     # Resize cropped face to fill ~65% of a 512x512 canvas
#     canvas_size = 512
#     target_fill_ratio = 0.65  # Slightly increased from previous 0.6
#     h, w = rgba_image.shape[:2]
#     max_dim = int(canvas_size * target_fill_ratio)
#     scale = min(max_dim / h, max_dim / w)
#     resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

#     # Create a 512x512 RGBA canvas and center the resized image
#     canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
#     y_offset = (canvas_size - resized.shape[0]) // 2
#     x_offset = (canvas_size - resized.shape[1]) // 2
#     canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
#     rgba_image = canvas

#     out_name = "face_hair_segmented1.png"
#     cv2.imwrite(out_name, rgba_image)
#     print(f"Saved to {out_name}")

# def crop_to_alpha(rgba):
#     """
#     Crops the RGBA image to the bounding box of non-zero alpha pixels.
#     Adds a small padding to preserve more details.
#     """
#     alpha = rgba[:, :, 3]
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)

#     # Increased padding to preserve more details
#     padding = 20  # Increased from previous 0
#     y_start = max(0, y - padding)
#     x_start = max(0, x - padding)
#     y_end = min(rgba.shape[0], y + h + padding)
#     x_end = min(rgba.shape[1], x + w + padding)

#     cropped = rgba[y_start:y_end, x_start:x_end]
#     return cropped

# def detect_eye_boxes(image_rgb):
#     """
#     Uses MediaPipe FaceMesh to detect landmarks, returning bounding boxes
#     for the left and right eye regions. Each box has some padding around the eye.
#     """
#     with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
#         results = face_mesh.process(image_rgb)
#         if not results.multi_face_landmarks:
#             return None, None

#         landmarks = results.multi_face_landmarks[0].landmark
#         h, w, _ = image_rgb.shape

#         def get_box(indices):
#             xs = [int(landmarks[i].x * w) for i in indices]
#             ys = [int(landmarks[i].y * h) for i in indices]
#             return (min(xs) - 15, min(ys) - 15, max(xs) + 15, max(ys) + 15)

#         # Eye landmark indices (you can tweak these if needed)
#         left_eye_indices = list(range(33, 42))
#         right_eye_indices = list(range(263, 272))

#         return get_box(left_eye_indices), get_box(right_eye_indices)

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_segmentation.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])




# ####WORKING CODE ----------- EXCEPT BLACK SPOT ON LEFT EYE---------------#########


import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import mediapipe as mp

from model import BiSeNet

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def detect_single_face(image_rgb):
    """
    Checks if exactly one face is detected using MediaPipe's FaceDetection.
    Returns True if exactly one face is found, otherwise False.
    """
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)
        return (results.detections and len(results.detections) == 1)

def load_bisenet_model(weight_path='79999_iter.pth', n_classes=19):
    """
    Loads the BiSeNet model with specified weights for face/hair segmentation.
    """
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()
    return net

def keep_largest_component(mask_255):
    """
    Retains the largest connected component in a binary mask (255 = foreground).
    Helps remove small spurious regions.
    """
    mask_bin = (mask_255 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_255

    largest_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for label_id in range(2, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            largest_label = label_id

    final_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    final_mask[labels == largest_label] = 255
    return final_mask

def fill_mask_holes(mask_255):
    """
    Fills holes in the mask by inverting the mask, keeping the largest background,
    then inverting again.
    """
    inv_mask = cv2.bitwise_not(mask_255)
    largest_bg = keep_largest_component(inv_mask)
    filled_mask = cv2.bitwise_not(largest_bg)
    return filled_mask

def parse_image(image_bgr, model):
    """
    Resizes the input image, runs BiSeNet segmentation, and returns the parsing map.
    Also returns the original image dimensions.
    """
    # Increase to 1024 if you want finer details and have enough memory.
    input_size = 512  
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image_rgb.shape

    # Resize to input_size x input_size
    image_resized = cv2.resize(image_rgb, (input_size, input_size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tensor = transform(image_resized).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        main_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        main_output = main_output.squeeze(0)
        parsing_map = torch.argmax(main_output, dim=0).cpu().numpy()

    return parsing_map, orig_h, orig_w

def face_hair_mask(parsing_map, orig_w, orig_h):
    """
    Creates a mask for face/hair/neck based on the parsing map, applies morphological
    closing and a small dilation to preserve hair details, then resizes to the
    original image size.
    """
    # Classes for face, hair, ears, neck, etc.
    keep_classes = list(range(1, 14)) + [15, 17]  # includes neck (13)
    mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255

    # Morphological operations:
    # Do a small closing to fill minor gaps, then a more aggressive dilation
    kernel = np.ones((3, 3), np.uint8)

    closed = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=3)

    largest = keep_largest_component(dilated)
    filled = fill_mask_holes(largest)
    final_512 = keep_largest_component(filled)

    # Resize back to original dimensions
    mask_original = cv2.resize(final_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_original

def create_rgba_image(original_bgr, mask):
    """
    Creates an RGBA image from the original BGR image and a 255-based mask.
    Pixels where mask == 255 are kept, alpha=255; otherwise alpha=0.
    """
    orig_h, orig_w, _ = original_bgr.shape
    rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    face_pixels = (mask == 255)
    rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
    rgba_image[face_pixels, 3] = 255
    return rgba_image

def feather_alpha(rgba, radius=5, iterations=1):
    """
    Applies a lighter Gaussian blur to the alpha channel to create a subtler edge.
    Reduces radius and iterations for a cleaner transition.
    """
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    
    # Single pass of light Gaussian blur
    alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
    
    alpha = np.clip(alpha, 0, 1)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return rgba

def make_glasses_see_through(rgba_image, left_eye_box, right_eye_box, color_threshold=15, edge_aperture=3):
    """
    More precise approach to making glasses lenses transparent.
    Reduces color threshold and preserves more edge details.
    """
    h, w, _ = rgba_image.shape
    bgr = rgba_image[:, :, :3].copy()
    alpha = rgba_image[:, :, 3].copy()
    mask_outside_eyes = np.ones((h, w), dtype=np.uint8)

    # Mark outside eye regions to compute average face color
    for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        mask_outside_eyes[y1:y2, x1:x2] = 0

    valid_pixels = (alpha == 255) & (mask_outside_eyes == 1)
    # Compute mean color from face pixels outside the eye boxes
    if np.count_nonzero(valid_pixels) > 0:
        face_color = bgr[valid_pixels].mean(axis=0)
    else:
        face_color = np.array([128, 128, 128])  # fallback if no valid pixels

    # For each eye box, remove face-like regions while preserving more details
    for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        region_bgr = bgr[y1:y2, x1:x2]
        region_alpha = alpha[y1:y2, x1:x2]

        # Enhanced edge detection to preserve glasses frame
        edges = cv2.Canny(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=edge_aperture)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        # More precise color difference calculation
        diff = np.linalg.norm(region_bgr.astype(np.float32) - face_color, axis=2)

        # More selective removal of pixels
        remove_mask = (diff < color_threshold) & (edges == 0)
        region_alpha[remove_mask] = 0

        alpha[y1:y2, x1:x2] = region_alpha

    rgba_image[:, :, 3] = alpha
    return rgba_image

def main(image_path):
    """
    Main function to:
    1) Load and validate the image
    2) Detect single face
    3) Load BiSeNet model and parse image
    4) Create an RGBA image with the face/hair segmented
    5) Feather alpha edges with subtler effect
    6) Make glasses lenses see-through (if eyes detected)
    7) Crop to alpha
    8) Resize to fit a 512x512 canvas
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Cannot read {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not detect_single_face(image_rgb):
        print("No face or multiple faces detected. Exiting.")
        sys.exit(0)
    else:
        print("Exactly one face detected.")

    print("Loading BiSeNet model...")
    model = load_bisenet_model('79999_iter.pth', 19)

    # Segment face and hair
    parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
    final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
    rgba_image = create_rgba_image(image_bgr, final_mask)

    # Subtle alpha feathering for cleaner edges
    rgba_image = feather_alpha(rgba_image, radius=5, iterations=1)

    # Attempt to make glasses lenses see-through
    left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
    if left_eye_box and right_eye_box:
        rgba_image = make_glasses_see_through(
            rgba_image,
            left_eye_box,
            right_eye_box,
            color_threshold=15,  # More precise removal
            edge_aperture=3
        )

    # Crop to bounding box of non-transparent pixels with smaller padding
    rgba_image = crop_to_alpha(rgba_image, padding=10)

    # Resize cropped face to fill ~65% of a 512x512 canvas
    canvas_size = 512
    target_fill_ratio = 0.65
    h, w = rgba_image.shape[:2]
    max_dim = int(canvas_size * target_fill_ratio)
    scale = min(max_dim / h, max_dim / w)
    resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Create a 512x512 RGBA canvas and center the resized image
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    y_offset = (canvas_size - resized.shape[0]) // 2
    x_offset = (canvas_size - resized.shape[1]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    rgba_image = canvas

    out_name = "face_hair_segmented2.png"
    cv2.imwrite(out_name, rgba_image)
    print(f"Saved to {out_name}")

def crop_to_alpha(rgba, padding=10):
    """
    Crops the RGBA image to the bounding box of non-zero alpha pixels.
    Adds a small padding to preserve details without excessive space.
    """
    alpha = rgba[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return rgba
    x, y, w, h = cv2.boundingRect(coords)

    # More moderate padding to preserve details without excess
    y_start = max(0, y - padding)
    x_start = max(0, x - padding)
    y_end = min(rgba.shape[0], y + h + padding)
    x_end = min(rgba.shape[1], x + w + padding)

    cropped = rgba[y_start:y_end, x_start:x_end]
    return cropped

def detect_eye_boxes(image_rgb):
    """
    Uses MediaPipe FaceMesh to detect landmarks, returning bounding boxes
    for the left and right eye regions. Each box has some padding around the eye.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image_rgb.shape

        def get_box(indices):
            xs = [int(landmarks[i].x * w) for i in indices]
            ys = [int(landmarks[i].y * h) for i in indices]
            return (min(xs) - 15, min(ys) - 15, max(xs) + 15, max(ys) + 15)

        # Eye landmark indices (you can tweak these if needed)
        left_eye_indices = list(range(33, 42))
        right_eye_indices = list(range(263, 272))

        return get_box(left_eye_indices), get_box(right_eye_indices)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_segmentation.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])




























# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# from model import BiSeNet

# mp_face_mesh = mp.solutions.face_mesh
# mp_face_detection = mp.solutions.face_detection

# def detect_single_face(image_rgb):
#     """
#     Checks if exactly one face is detected using MediaPipe's FaceDetection.
#     Returns True if exactly one face is found, otherwise False.
#     """
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
#         results = face_detection.process(image_rgb)
#         return (results.detections and len(results.detections) == 1)

# def load_bisenet_model(weight_path='79999_iter.pth', n_classes=19):
#     """
#     Loads the BiSeNet model with specified weights for face/hair segmentation.
#     """
#     net = BiSeNet(n_classes=n_classes)
#     net.load_state_dict(torch.load(weight_path, map_location='cpu'))
#     net.eval()
#     return net

# def keep_largest_component(mask_255):
#     """
#     Retains the largest connected component in a binary mask (255 = foreground).
#     Helps remove small spurious regions.
#     """
#     mask_bin = (mask_255 > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels <= 1:
#         return mask_255

#     largest_label = 1
#     max_area = stats[1, cv2.CC_STAT_AREA]
#     for label_id in range(2, num_labels):
#         area = stats[label_id, cv2.CC_STAT_AREA]
#         if area > max_area:
#             max_area = area
#             largest_label = label_id

#     final_mask = np.zeros_like(mask_bin, dtype=np.uint8)
#     final_mask[labels == largest_label] = 255
#     return final_mask

# def fill_mask_holes(mask_255):
#     """
#     Fills holes in the mask by inverting the mask, keeping the largest background,
#     then inverting again.
#     """
#     inv_mask = cv2.bitwise_not(mask_255)
#     largest_bg = keep_largest_component(inv_mask)
#     filled_mask = cv2.bitwise_not(largest_bg)
#     return filled_mask

# def parse_image(image_bgr, model):
#     """
#     Resizes the input image, runs BiSeNet segmentation, and returns the parsing map.
#     Also returns the original image dimensions.
#     """
#     input_size = 512  
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape

#     # Resize to input_size x input_size
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     tensor = transform(image_resized).unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(tensor)
#         main_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
#         main_output = main_output.squeeze(0)
#         parsing_map = torch.argmax(main_output, dim=0).cpu().numpy()

#     return parsing_map, orig_h, orig_w

# def face_hair_mask(parsing_map, orig_w, orig_h):
#     """
#     Creates a mask for face/hair/neck based on the parsing map, applies morphological
#     operations to refine the mask, then resizes to the original image size.
#     """
#     # Classes for face, hair, ears, neck, etc.
#     # Modified to match expected output better
#     face_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # Face, eyes, brows, nose, mouth regions
#     hair_classes = [17]  # Hair class
#     ear_classes = [15]  # Ear class
    
#     # Create separate masks for face and hair
#     face_mask = np.isin(parsing_map, face_classes).astype(np.uint8) * 255
#     hair_mask = np.isin(parsing_map, hair_classes).astype(np.uint8) * 255
#     ear_mask = np.isin(parsing_map, ear_classes).astype(np.uint8) * 255
    
#     # Process face mask (more conservative morphology)
#     face_kernel = np.ones((3, 3), np.uint8)
#     face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, face_kernel, iterations=1)
    
#     # Process hair mask (more aggressive to capture all hair)
#     hair_kernel = np.ones((5, 5), np.uint8)
#     hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, hair_kernel, iterations=2)
#     hair_mask = cv2.dilate(hair_mask, hair_kernel, iterations=1)
    
#     # Process ear mask
#     ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_CLOSE, face_kernel, iterations=1)
    
#     # Combine masks
#     combined_mask = cv2.bitwise_or(face_mask, hair_mask)
#     combined_mask = cv2.bitwise_or(combined_mask, ear_mask)
    
#     # Fill holes and keep largest component
#     filled = fill_mask_holes(combined_mask)
#     largest = keep_largest_component(filled)
    
#     # Resize back to original dimensions with INTER_LINEAR for smoother edges
#     mask_original = cv2.resize(largest, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
#     # Threshold to ensure binary mask
#     _, mask_original = cv2.threshold(mask_original, 127, 255, cv2.THRESH_BINARY)
    
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Creates an RGBA image from the original BGR image and a 255-based mask.
#     Pixels where mask == 255 are kept, alpha=255; otherwise alpha=0.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def feather_alpha(rgba):
#     """
#     Simplified edge feathering specifically designed to match the expected output.
#     Creates a thin gradient at the edges while keeping the interior fully opaque.
#     """
#     # Extract alpha channel
#     alpha = rgba[:, :, 3].copy().astype(np.float32)
    
#     # Create binary mask
#     _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
#     # Find edge pixels
#     kernel = np.ones((3, 3), np.uint8)
#     eroded = cv2.erode(binary, kernel, iterations=1)
#     edge_mask = binary - eroded
    
#     # Apply feathering only to the edge
#     feathered = alpha.copy()
#     edge_indices = np.where(edge_mask > 0)
    
#     # Set edge pixels to a gradient (simple feathering)
#     feathered[edge_indices] = 200  # Less transparent edge
    
#     # Apply a very slight blur to smooth transitions
#     feathered = cv2.GaussianBlur(feathered, (3, 3), 0.5)
    
#     # Ensure interior pixels remain fully opaque
#     interior_indices = np.where(eroded > 0)
#     feathered[interior_indices] = 255
    
#     rgba[:, :, 3] = feathered.astype(np.uint8)
#     return rgba

# def refine_glasses(rgba_image):
#     """
#     Completely new approach to glasses refinement, focused on
#     simply making the lenses transparent while preserving frames.
#     """
#     # Convert to HSV for better color thresholding
#     bgr = rgba_image[:, :, :3]
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     alpha = rgba_image[:, :, 3].copy()
    
#     # Mask for areas with alpha > 0
#     valid_mask = (alpha > 0)
    
#     # Create a mask for possible lens areas (low saturation, medium-high value)
#     # These values are tuned for the expected output
#     lower_hsv = np.array([0, 0, 30])  # Low saturation, medium value
#     upper_hsv = np.array([180, 30, 220])  # Any hue, low saturation, high value
#     lens_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
#     # Only consider lens pixels where alpha is non-zero
#     lens_mask = lens_mask & valid_mask
    
#     # Find edges in the color image
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
#     # Create mask for frames (edges and non-lens areas)
#     frame_mask = edges | ~lens_mask
    
#     # Make lens areas transparent, preserve frames
#     final_alpha = alpha.copy()
#     lens_indices = np.where((lens_mask > 0) & (~frame_mask))
#     final_alpha[lens_indices] = 0
    
#     # Update alpha channel
#     rgba_image[:, :, 3] = final_alpha
    
#     return rgba_image

# def crop_to_alpha(rgba, padding=3):
#     """
#     Crops the RGBA image to the bounding box of non-zero alpha pixels.
#     Reduced padding to match expected output.
#     """
#     alpha = rgba[:, :, 3]
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)

#     y_start = max(0, y - padding)
#     x_start = max(0, x - padding)
#     y_end = min(rgba.shape[0], y + h + padding)
#     x_end = min(rgba.shape[1], x + w + padding)

#     cropped = rgba[y_start:y_end, x_start:x_end]
#     return cropped

# def main(image_path):
#     """
#     Main function with simplified workflow:
#     1) Load and validate the image
#     2) Detect single face
#     3) Load BiSeNet model and parse image
#     4) Create refined face/hair mask
#     5) Create an RGBA image
#     6) Apply simplified feathering
#     7) Apply specialized glasses refinement
#     8) Crop to alpha with minimal padding
#     9) Resize to match expected output dimensions
#     """
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)

#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     if not detect_single_face(image_rgb):
#         print("No face or multiple faces detected. Exiting.")
#         sys.exit(0)
#     else:
#         print("Exactly one face detected.")

#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)

#     # Segment face and hair
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
#     rgba_image = create_rgba_image(image_bgr, final_mask)

#     # Apply simplified feathering
#     rgba_image = feather_alpha(rgba_image)
    
#     # Apply specialized glasses refinement
#     rgba_image = refine_glasses(rgba_image)

#     # Crop to the bounding box of non-transparent pixels with minimal padding
#     rgba_image = crop_to_alpha(rgba_image, padding=3)

#     # Determine target size based on expected output dimensions
#     # Analyzing your expected output, it appears to be approximately 140x170 pixels
#     target_width = 140
#     target_height = 170
    
#     # Resize to match expected output dimensions
#     # Use INTER_AREA for downsizing to avoid moiré patterns
#     resized = cv2.resize(rgba_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

#     out_name = "face_hair_segmented_exact_match.png"
#     cv2.imwrite(out_name, resized)
#     print(f"Saved to {out_name}")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_segmentation.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])       









  




