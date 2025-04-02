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
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
#         results = face_detection.process(image_rgb)
#         return (results.detections and len(results.detections) == 1)

# def load_bisenet_model(weight_path='79999_iter.pth', n_classes=19):
#     net = BiSeNet(n_classes=n_classes)
#     net.load_state_dict(torch.load(weight_path, map_location='cpu'))
#     net.eval()
#     return net

# def keep_largest_component(mask_255):
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
#     inv_mask = cv2.bitwise_not(mask_255)
#     largest_bg = keep_largest_component(inv_mask)
#     filled_mask = cv2.bitwise_not(largest_bg)
#     return filled_mask

# def parse_image(image_bgr, model):
#     input_size = 512
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape

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
#     keep_classes = list(range(1, 14)) + [15, 17]
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255

#     kernel = np.ones((3, 3), np.uint8)
#     closed = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=2)
#     dilated = cv2.dilate(closed, kernel, iterations=3)

#     largest = keep_largest_component(dilated)
#     filled = fill_mask_holes(largest)
#     final_512 = keep_largest_component(filled)

#     mask_original = cv2.resize(final_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def feather_alpha(rgba, radius=5, iterations=1):
#     alpha = rgba[:, :, 3].astype(np.float32) / 255.0
#     alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
#     alpha = np.clip(alpha, 0, 1)
#     rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
#     return rgba

# def make_glasses_see_through(rgba_image, left_eye_box, right_eye_box):
#     # Disabled to avoid black spot artifacts under glasses
#     return rgba_image

# def crop_to_alpha(rgba, padding=10):
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

# def detect_eye_boxes(image_rgb):
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

#         left_eye_indices = list(range(33, 42))
#         right_eye_indices = list(range(263, 272))

#         return get_box(left_eye_indices), get_box(right_eye_indices)

# # def main(image_path):
# #     image_bgr = cv2.imread(image_path)
# #     if image_bgr is None:
# #         print(f"Error: Cannot read {image_path}")
# #         sys.exit(1)

# #     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# #     if not detect_single_face(image_rgb):
# #         print("No face or multiple faces detected. Exiting.")
# #         sys.exit(0)
# #     else:
# #         print("Exactly one face detected.")

# #     print("Loading BiSeNet model...")
# #     model = load_bisenet_model('79999_iter.pth', 19)

# #     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
# #     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
# #     rgba_image = create_rgba_image(image_bgr, final_mask)

# #     rgba_image = feather_alpha(rgba_image, radius=5, iterations=1)

# #     # Eye box detection only (transparency disabled)
# #     left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
# #     if left_eye_box and right_eye_box:
# #         rgba_image = make_glasses_see_through(rgba_image, left_eye_box, right_eye_box)

# #     rgba_image = crop_to_alpha(rgba_image, padding=10)

# #     canvas_size = 512
# #     target_fill_ratio = 0.65
# #     h, w = rgba_image.shape[:2]
# #     max_dim = int(canvas_size * target_fill_ratio)
# #     scale = min(max_dim / h, max_dim / w)
# #     resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# #     canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
# #     y_offset = (canvas_size - resized.shape[0]) // 2
# #     x_offset = (canvas_size - resized.shape[1]) // 2
# #     canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
# #     rgba_image = canvas

# #     out_name = "face_hair_segmented2.png"
# #     cv2.imwrite(out_name, rgba_image)
# #     print(f"Saved to {out_name}")



# def process_image(image_bgr):
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#     # Validate face
#     if not detect_single_face(image_rgb):
#         raise ValueError("No face or multiple faces detected.")

#     # Load model (load only once outside in real app for performance)
#     model = load_bisenet_model('79999_iter.pth', 19)

#     # Run BiSeNet segmentation
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
#     rgba_image = create_rgba_image(image_bgr, final_mask)
#     rgba_image = feather_alpha(rgba_image, radius=5, iterations=1)

#     # Optional: handle glasses
#     left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
#     if left_eye_box and right_eye_box:
#         rgba_image = make_glasses_see_through(rgba_image, left_eye_box, right_eye_box)

#     # Crop and center
#     rgba_image = crop_to_alpha(rgba_image, padding=10)

#     # Resize and center on transparent canvas
#     canvas_size = 512
#     target_fill_ratio = 0.65
#     h, w = rgba_image.shape[:2]
#     max_dim = int(canvas_size * target_fill_ratio)
#     scale = min(max_dim / h, max_dim / w)
#     resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

#     canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
#     y_offset = (canvas_size - resized.shape[0]) // 2
#     x_offset = (canvas_size - resized.shape[1]) // 2
#     canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

#     return canvas  # final RGBA NumPy array

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_segmentation.py <image_path>")
#         sys.exit(1)

#     image_path = sys.argv[1]
#     image_bgr = cv2.imread(image_path)

#     try:
#         output_rgba = process_image(image_bgr)
#         cv2.imwrite("segmented_face.png", output_rgba)
#         print("Saved segmented image to segmented_face.png")
#     except ValueError as e:
#         print(f"Error: {e}")











import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import mediapipe as mp
from model import BiSeNet

# MediaPipe models
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def detect_single_face(image_rgb):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)
        return (results.detections and len(results.detections) == 1)

def load_bisenet_model(weight_path='79999_iter.pth', n_classes=19):
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()
    return net

def keep_largest_component(mask_255):
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
    inv_mask = cv2.bitwise_not(mask_255)
    largest_bg = keep_largest_component(inv_mask)
    filled_mask = cv2.bitwise_not(largest_bg)
    return filled_mask

def parse_image(image_bgr, model):
    input_size = 512
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image_rgb.shape

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
    keep_classes = list(range(1, 14)) + [15, 17]
    mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=3)

    largest = keep_largest_component(dilated)
    filled = fill_mask_holes(largest)
    final_512 = keep_largest_component(filled)

    mask_original = cv2.resize(final_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_original

def create_rgba_image(original_bgr, mask):
    orig_h, orig_w, _ = original_bgr.shape
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)  # Fix: Convert to RGB
    rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    face_pixels = (mask == 255)
    rgba_image[face_pixels, 0:3] = original_rgb[face_pixels, :]  # Use RGB instead of BGR
    rgba_image[face_pixels, 3] = 255
    return rgba_image

def feather_alpha(rgba, radius=5, iterations=1):
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
    alpha = np.clip(alpha, 0, 1)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return rgba

def make_glasses_see_through(rgba_image, left_eye_box, right_eye_box):
    # Currently disabled
    return rgba_image

def crop_to_alpha(rgba, padding=10):
    alpha = rgba[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return rgba
    x, y, w, h = cv2.boundingRect(coords)

    y_start = max(0, y - padding)
    x_start = max(0, x - padding)
    y_end = min(rgba.shape[0], y + h + padding)
    x_end = min(rgba.shape[1], x + w + padding)

    cropped = rgba[y_start:y_end, x_start:x_end]
    return cropped

def detect_eye_boxes(image_rgb):
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

        left_eye_indices = list(range(33, 42))
        right_eye_indices = list(range(263, 272))

        return get_box(left_eye_indices), get_box(right_eye_indices)

# Core processing function used by the Streamlit UI
def process_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if not detect_single_face(image_rgb):
        raise ValueError("No face or multiple faces detected.")

    model = load_bisenet_model('79999_iter.pth', 19)
    parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
    final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
    rgba_image = create_rgba_image(image_bgr, final_mask)
    rgba_image = feather_alpha(rgba_image, radius=5, iterations=1)

    left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
    if left_eye_box and right_eye_box:
        rgba_image = make_glasses_see_through(rgba_image, left_eye_box, right_eye_box)

    rgba_image = crop_to_alpha(rgba_image, padding=10)

    canvas_size = 512
    target_fill_ratio = 0.65
    h, w = rgba_image.shape[:2]
    max_dim = int(canvas_size * target_fill_ratio)
    scale = min(max_dim / h, max_dim / w)
    resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    y_offset = (canvas_size - resized.shape[0]) // 2
    x_offset = (canvas_size - resized.shape[1]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    return canvas

# Optional CLI for testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_segmentation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image_bgr = cv2.imread(image_path)

    try:
        output_rgba = process_image(image_bgr)
        cv2.imwrite("segmented_face.png", output_rgba)
        print("Saved segmented image to segmented_face.png")
    except ValueError as e:
        print(f"Error: {e}")


