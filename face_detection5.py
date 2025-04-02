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
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
#     input_size = 512
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
#     keep_classes = list(range(1, 14)) + [15, 17]  # include neck (13)
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255
#     opened = cv2.morphologyEx(mask_512, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
#     largest = keep_largest_component(closed)
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


# def feather_alpha(rgba, radius=5):
#     alpha = rgba[:, :, 3].astype(np.float32) / 255.0
#     alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
#     alpha = np.clip(alpha, 0, 1)
#     rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
#     return rgba


# def crop_to_alpha(rgba):
#     alpha = rgba[:, :, 3]
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)
#     cropped = rgba[y:y + h, x:x + w]
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
#             return (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

#         left_eye_indices = list(range(33, 42))
#         right_eye_indices = list(range(263, 272))

#         return get_box(left_eye_indices), get_box(right_eye_indices)


# def make_glasses_see_through(rgba_image, left_eye_box, right_eye_box, color_threshold=50, edge_aperture=3):
#     h, w, _ = rgba_image.shape
#     bgr = rgba_image[:, :, :3].copy()
#     alpha = rgba_image[:, :, 3].copy()
#     mask_outside_eyes = np.ones((h, w), dtype=np.uint8)

#     for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
#         x1, x2 = max(0, x1), min(w, x2)
#         y1, y2 = max(0, y1), min(h, y2)
#         mask_outside_eyes[y1:y2, x1:x2] = 0

#     valid_pixels = (alpha == 255) & (mask_outside_eyes == 1)
#     face_color = bgr[valid_pixels].mean(axis=0) if np.count_nonzero(valid_pixels) > 0 else np.array([128, 128, 128])

#     for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
#         x1, x2 = max(0, x1), min(w, x2)
#         y1, y2 = max(0, y1), min(h, y2)
#         region_bgr = bgr[y1:y2, x1:x2]
#         region_alpha = alpha[y1:y2, x1:x2]
#         edges = cv2.Canny(region_bgr, 100, 200, apertureSize=edge_aperture)
#         diff = np.linalg.norm(region_bgr.astype(np.float32) - face_color, axis=2)
#         remove_mask = (diff < color_threshold) & (edges == 0)
#         region_alpha[remove_mask] = 0
#         alpha[y1:y2, x1:x2] = region_alpha

#     rgba_image[:, :, 3] = alpha
#     return rgba_image


# def main(image_path):
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
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
#     rgba_image = create_rgba_image(image_bgr, final_mask)
#     rgba_image = feather_alpha(rgba_image, radius=5)

#     left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
#     if left_eye_box and right_eye_box:
#         rgba_image = make_glasses_see_through(rgba_image, left_eye_box, right_eye_box, color_threshold=50, edge_aperture=3)

#     rgba_image = crop_to_alpha(rgba_image)

#     # Resize cropped face to fill ~60% of canvas and center in 512x512
#     canvas_size = 512
#     target_fill_ratio = 0.6  # Adjusted for better proportion
#     max_dim = int(canvas_size * target_fill_ratio)

#     h, w = rgba_image.shape[:2]
#     scale = min(max_dim / h, max_dim / w)
#     resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

#     canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
#     y_offset = (canvas_size - resized.shape[0]) // 2
#     x_offset = (canvas_size - resized.shape[1]) // 2
#     canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
#     rgba_image = canvas

#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, rgba_image)
#     print(f"Saved to {out_name}")


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_segmentation.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])















# import sys
# import os
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# # Import pyMatting modules
# from pymatting import (
#     cutout,
#     estimate_alpha_cf,
#     estimate_alpha_knn,
#     estimate_foreground_ml
# )

# # Import your BiSeNet model definition
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

# def parse_image(image_bgr, model, input_size=512):
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

# def create_initial_mask(parsing_map, orig_w, orig_h):
#     keep_classes = list(range(1, 14)) + [15, 17]  # includes neck (13)
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255
#     mask_original = cv2.resize(mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_trimap(mask_255, kernel_size=10):
#     mask_bin = (mask_255 == 255).astype(np.uint8)

#     # definite foreground
#     fg = cv2.erode(mask_bin, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
#     # definite background
#     bg = cv2.dilate(mask_bin, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

#     trimap = np.zeros_like(mask_bin, dtype=np.uint8)
#     trimap[fg == 1] = 255
#     trimap[bg == 0] = 0
#     trimap[(fg == 0) & (bg == 1)] = 128
#     return trimap

# def apply_matting(original_bgr, mask_255, matting_method="cf", kernel_size=10):
#     """
#     Uses pyMatting to refine edges.
#     'matting_method' can be "cf" (closed-form) or "knn".
#     """
#     trimap = create_trimap(mask_255, kernel_size=kernel_size)

#     temp_image_path = "temp_image.png"
#     temp_trimap_path = "temp_trimap.png"
#     temp_output_path = "temp_matted.png"

#     cv2.imwrite(temp_image_path, original_bgr)
#     cv2.imwrite(temp_trimap_path, trimap)

#     # Select alpha estimator
#     if matting_method.lower() == "cf":
#         alpha_estimator = estimate_alpha_cf
#     elif matting_method.lower() == "knn":
#         alpha_estimator = estimate_alpha_knn
#     else:
#         raise ValueError(f"Unknown matting method: {matting_method}")

#     cutout(
#         temp_image_path,
#         temp_trimap_path,
#         temp_output_path,
#         alpha_estimator=alpha_estimator,
#         foreground_method=estimate_foreground_ml
#     )

#     rgba = cv2.imread(temp_output_path, cv2.IMREAD_UNCHANGED)

#     # Cleanup if desired
#     if os.path.exists(temp_image_path):
#         os.remove(temp_image_path)
#     if os.path.exists(temp_trimap_path):
#         os.remove(temp_trimap_path)
#     if os.path.exists(temp_output_path):
#         os.remove(temp_output_path)

#     return rgba

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
#             return (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

#         left_eye_indices = list(range(33, 42))
#         right_eye_indices = list(range(263, 272))

#         return get_box(left_eye_indices), get_box(right_eye_indices)

# def make_glasses_see_through(rgba_image, left_eye_box, right_eye_box, color_threshold=30, edge_aperture=3):
#     h, w, _ = rgba_image.shape
#     bgr = rgba_image[:, :, :3].copy()
#     alpha = rgba_image[:, :, 3].copy()
#     mask_outside_eyes = np.ones((h, w), dtype=np.uint8)

#     for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
#         x1, x2 = max(0, x1), min(w, x2)
#         y1, y2 = max(0, y1), min(h, y2)
#         mask_outside_eyes[y1:y2, x1:x2] = 0

#     valid_pixels = (alpha == 255) & (mask_outside_eyes == 1)
#     if np.count_nonzero(valid_pixels) > 0:
#         face_color = bgr[valid_pixels].mean(axis=0)
#     else:
#         face_color = np.array([128, 128, 128])

#     for (x1, y1, x2, y2) in [left_eye_box, right_eye_box]:
#         x1, x2 = max(0, x1), min(w, x2)
#         y1, y2 = max(0, y1), min(h, y2)
#         region_bgr = bgr[y1:y2, x1:x2]
#         region_alpha = alpha[y1:y2, x1:x2]

#         edges = cv2.Canny(region_bgr, 100, 200, apertureSize=edge_aperture)
#         diff = np.linalg.norm(region_bgr.astype(np.float32) - face_color, axis=2)

#         remove_mask = (diff < color_threshold) & (edges == 0)
#         region_alpha[remove_mask] = 0
#         alpha[y1:y2, x1:x2] = region_alpha

#     rgba_image[:, :, 3] = alpha
#     return rgba_image

# def crop_to_alpha(rgba):
#     alpha = rgba[:, :, 3]
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)
#     return rgba[y:y+h, x:x+w]

# def feather_alpha(rgba, radius=5):
#     alpha = rgba[:, :, 3].astype(np.float32) / 255.0
#     alpha = cv2.GaussianBlur(alpha, (2*radius+1, 2*radius+1), radius)
#     alpha = np.clip(alpha, 0, 1)
#     rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
#     return rgba

# def main(image_path):
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

#     # Get parsing map
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model, input_size=512)
#     mask_255 = create_initial_mask(parsing_map, orig_w, orig_h)

#     # Apply matting (closed-form or knn)
#     rgba_image = apply_matting(image_bgr, mask_255, matting_method="cf", kernel_size=10)

#     # Make glasses transparent if eyes are detected
#     left_eye_box, right_eye_box = detect_eye_boxes(image_rgb)
#     if left_eye_box and right_eye_box:
#         rgba_image = make_glasses_see_through(rgba_image, left_eye_box, right_eye_box,
#                                               color_threshold=30, edge_aperture=3)

#     # Crop to alpha
#     rgba_image = crop_to_alpha(rgba_image)

#     # Optional feather
#     rgba_image = feather_alpha(rgba_image, radius=5)

#     # Resize cropped face to fill ~60% of a 512x512 canvas
#     canvas_size = 512
#     target_fill_ratio = 0.6
#     h, w = rgba_image.shape[:2]
#     max_dim = int(canvas_size * target_fill_ratio)
#     scale = min(max_dim / h, max_dim / w)
#     resized = cv2.resize(rgba_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

#     canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
#     y_offset = (canvas_size - resized.shape[0]) // 2
#     x_offset = (canvas_size - resized.shape[1]) // 2
#     canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, canvas)
#     print(f"Saved to {out_name}")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection5.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])









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
#     face_classes = [1,2,3,4,5,6,7,8,9,10,11,12,13]
#     hair_classes = [17]
#     ear_classes = [15]

#     face_mask = np.isin(parsing_map, face_classes).astype(np.uint8) * 255
#     hair_mask = np.isin(parsing_map, hair_classes).astype(np.uint8) * 255
#     ear_mask = np.isin(parsing_map, ear_classes).astype(np.uint8) * 255

#     face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
#     hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
#     hair_mask = cv2.dilate(hair_mask, np.ones((5, 5), np.uint8), iterations=1)
#     ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

#     combined_mask = cv2.bitwise_or(face_mask, hair_mask)
#     combined_mask = cv2.bitwise_or(combined_mask, ear_mask)

#     filled = fill_mask_holes(combined_mask)
#     largest = keep_largest_component(filled)
#     largest = cv2.dilate(largest, np.ones((3, 3), np.uint8), iterations=1)

#     mask_original = cv2.resize(largest, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
#     _, mask_original = cv2.threshold(mask_original, 127, 255, cv2.THRESH_BINARY)

#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)

#     # Use premultiplied alpha blending to avoid halos
#     for c in range(3):
#         rgba_image[:, :, c] = np.where(face_pixels, original_bgr[:, :, c], 0)
#     rgba_image[:, :, 3] = np.where(face_pixels, 255, 0)

#     return rgba_image

# def feather_alpha_distance(rgba):
#     alpha = rgba[:, :, 3].copy()
#     mask = (alpha > 0).astype(np.uint8)
#     dist = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
#     dist = np.clip(dist / np.max(dist), 0, 1)
#     feathered_alpha = (dist * 255).astype(np.uint8)
#     rgba[:, :, 3] = feathered_alpha
#     return rgba

# def refine_glasses(rgba_image):
#     bgr = rgba_image[:, :, :3]
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     alpha = rgba_image[:, :, 3].copy()
#     valid_mask = (alpha > 0)

#     lower_hsv = np.array([0, 0, 40])
#     upper_hsv = np.array([180, 25, 230])
#     lens_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
#     lens_mask = lens_mask & valid_mask

#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 40, 120)
#     edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

#     frame_mask = edges | ~lens_mask
#     final_alpha = alpha.copy()
#     lens_indices = np.where((lens_mask > 0) & (~frame_mask))
#     final_alpha[lens_indices] = 40

#     rgba_image[:, :, 3] = final_alpha
#     return rgba_image

# def crop_to_alpha(rgba, padding=2):
#     alpha = rgba[:, :, 3]
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)
#     y_start = max(0, y - padding)
#     x_start = max(0, x - padding)
#     y_end = min(rgba.shape[0], y + h + padding - 1)
#     x_end = min(rgba.shape[1], x + w + padding)
#     cropped = rgba[y_start:y_end, x_start:x_end]
#     return cropped

# def main(image_path):
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

#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)
#     rgba_image = create_rgba_image(image_bgr, final_mask)

#     rgba_image = feather_alpha_distance(rgba_image)
#     rgba_image = refine_glasses(rgba_image)
#     rgba_image = crop_to_alpha(rgba_image, padding=2)

#     # Do not blur entire image before resizing (causes halos)
#     # Resize directly using high-quality interpolation
#     resized = cv2.resize(rgba_image, (143, 168), interpolation=cv2.INTER_AREA)
#     out_name = "face_hair_segmented_exact_match.png"
#     cv2.imwrite(out_name, resized)
#     print(f"Saved to {out_name}")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_segmentation.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])  




###--------------------WORKING WITH LITTLE ISSUES --------------------------------------------------##########


# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp
# from model import BiSeNet

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

# def keep_largest_component(mask):
#     mask_bin = (mask > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels <= 1:
#         return mask
#     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     final_mask = np.zeros_like(mask_bin, dtype=np.uint8)
#     final_mask[labels == largest_label] = 255
#     return final_mask

# def fill_mask_holes(mask):
#     inv_mask = cv2.bitwise_not(mask)
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
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
#     with torch.no_grad():
#         outputs = model(tensor)[0].squeeze(0)
#         parsing_map = torch.argmax(outputs, dim=0).cpu().numpy()
#     return parsing_map, orig_h, orig_w

# def face_hair_mask(parsing_map, orig_w, orig_h):
#     mask = np.isin(parsing_map, [1,2,3,4,5,6,7,8,9,10,11,12,13,15,17]).astype(np.uint8)*255
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
#     mask = fill_mask_holes(mask)
#     mask = keep_largest_component(mask)
#     mask_original = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
#     _, mask_original = cv2.threshold(mask_original,127,255,cv2.THRESH_BINARY)
#     return mask_original

# def guided_alpha_refinement(image_bgr, initial_mask, radius=8, eps=1e-3):
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     mask = initial_mask.astype(np.float32) / 255.0

#     mean_gray = cv2.boxFilter(gray, -1, (radius, radius))
#     mean_mask = cv2.boxFilter(mask, -1, (radius, radius))
#     corr_gray = cv2.boxFilter(gray * gray, -1, (radius, radius))
#     corr_gray_mask = cv2.boxFilter(gray * mask, -1, (radius, radius))

#     var_gray = corr_gray - mean_gray ** 2
#     cov_gray_mask = corr_gray_mask - mean_gray * mean_mask

#     a = cov_gray_mask / (var_gray + eps)
#     b = mean_mask - a * mean_gray

#     mean_a = cv2.boxFilter(a, -1, (radius, radius))
#     mean_b = cv2.boxFilter(b, -1, (radius, radius))

#     refined_alpha = mean_a * gray + mean_b
#     refined_alpha = np.clip(refined_alpha * 255, 0, 255).astype(np.uint8)
#     return refined_alpha


# def create_rgba_image(original_bgr, mask):
#     rgba = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
#     rgba[:,:,3] = mask
#     return rgba

# def feather_alpha(rgba):
#     alpha = rgba[:,:,3].astype(np.float32)
#     alpha = cv2.GaussianBlur(alpha, (7,7), 2)
#     alpha[alpha>200]=255
#     alpha[alpha<5]=0
#     rgba[:,:,3] = alpha.astype(np.uint8)
#     return rgba

# def refine_glasses(rgba):
#     bgr, alpha = rgba[:,:,:3], rgba[:,:,3]
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     lens_mask = cv2.inRange(hsv, np.array([0,0,60]), np.array([180,30,220]))
#     lens_mask = cv2.GaussianBlur(lens_mask,(5,5),1)
#     alpha[lens_mask>100]=80
#     rgba[:,:,3]=alpha
#     return rgba

# def crop_to_alpha(rgba, padding=4):
#     alpha = rgba[:,:,3]
#     coords = cv2.findNonZero(alpha)
#     if coords is None: return rgba
#     x,y,w,h = cv2.boundingRect(coords)
#     y0=max(0,y-padding);x0=max(0,x-padding)
#     y1=min(rgba.shape[0],y+h+padding);x1=min(rgba.shape[1],x+w+padding)
#     return rgba[y0:y1,x0:x1]

# def main(image_path):
#     image_bgr=cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error reading {image_path}")
#         sys.exit(1)
#     if not detect_single_face(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)):
#         print("Face detection error");sys.exit(0)

#     model=load_bisenet_model('79999_iter.pth',19)
#     parsing_map,h,w=parse_image(image_bgr,model)
#     final_mask=face_hair_mask(parsing_map,w,h)

#     refined_alpha=guided_alpha_refinement(image_bgr,final_mask)
#     rgba=create_rgba_image(image_bgr,refined_alpha)

#     rgba=feather_alpha(rgba)
#     rgba=refine_glasses(rgba)
#     rgba=crop_to_alpha(rgba, padding=4)

#     final=cv2.resize(rgba,(143,168),interpolation=cv2.INTER_AREA)
#     out_name="face_hair_segmented_exact_match.png"
#     cv2.imwrite(out_name,final)
#     print(f"Saved {out_name}")

# if __name__=="__main__":
#     if len(sys.argv)<2:
#         print("Usage: python face_segmentation.py <image>")
#         sys.exit(1)
#     main(sys.argv[1])

























import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import mediapipe as mp
from model import BiSeNet

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

def keep_largest_component(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    final_mask[labels == largest_label] = 255
    return final_mask

def fill_mask_holes(mask):
    inv_mask = cv2.bitwise_not(mask)
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
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(image_resized).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)[0].squeeze(0)
        parsing_map = torch.argmax(outputs, dim=0).cpu().numpy()
    return parsing_map, orig_h, orig_w

def face_hair_mask(parsing_map, orig_w, orig_h):
    # Retain only face, hair, ears clearly (exclude neck, shoulders explicitly)
    face_classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,17]
    mask = np.isin(parsing_map, face_classes).astype(np.uint8)*255

    # Remove lower parts explicitly
    mask[400:, :] = 0  # Remove bottom area (adjust as needed if input is 512x512)

    # Morphology operations to clean edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)

    # Hole filling and largest component extraction
    mask = fill_mask_holes(mask)
    mask = keep_largest_component(mask)

    mask_original = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    _, mask_original = cv2.threshold(mask_original,127,255,cv2.THRESH_BINARY)
    return mask_original


def guided_alpha_refinement(image_bgr, initial_mask):
    refined_mask = cv2.GaussianBlur(initial_mask, (7, 7), 2)
    refined_mask = np.clip(refined_mask, 0, 255).astype(np.uint8)
    return refined_mask

def create_rgba_image(original_bgr, mask):
    rgba = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = mask
    return rgba

def feather_alpha(rgba):
    alpha = rgba[:, :, 3].astype(np.float32)
    feathered_alpha = cv2.GaussianBlur(alpha, (11, 11), 4)  # Increased blur for smoother edges
    feathered_alpha = np.clip(feathered_alpha, 0, 255)
    rgba[:, :, 3] = feathered_alpha.astype(np.uint8)
    return rgba


def refine_glasses(rgba_image):
    bgr = rgba_image[:, :, :3]
    alpha = rgba_image[:, :, 3].copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lens_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 40, 200]))
    lens_mask = cv2.GaussianBlur(lens_mask, (9, 9), 2)

    alpha = np.where(lens_mask > 100, 60, alpha)
    rgba_image[:, :, 3] = alpha.astype(np.uint8)

    return rgba_image


def crop_to_alpha(rgba, padding=10):  # increased padding for better symmetry
    alpha = rgba[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return rgba
    x, y, w, h = cv2.boundingRect(coords)

    y0 = max(0, y - padding)
    x0 = max(0, x - padding)
    y1 = min(rgba.shape[0], y + h + padding)
    x1 = min(rgba.shape[1], x + w + padding)

    cropped = rgba[y0:y1, x0:x1]
    return cropped


def main(image_path):
    image_bgr=cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error reading {image_path}")
        sys.exit(1)
    if not detect_single_face(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)):
        print("Face detection error");sys.exit(0)

    model=load_bisenet_model('79999_iter.pth',19)
    parsing_map,h,w=parse_image(image_bgr,model)
    final_mask=face_hair_mask(parsing_map,w,h)

    refined_alpha=guided_alpha_refinement(image_bgr,final_mask)
    rgba=create_rgba_image(image_bgr,refined_alpha)

    rgba=feather_alpha(rgba)
    rgba=refine_glasses(rgba)
    rgba=crop_to_alpha(rgba, padding=6)

    final=cv2.resize(rgba,(143,168),interpolation=cv2.INTER_AREA)
    out_name="face_hair_segmented_exact_match.png"
    cv2.imwrite(out_name,final)
    print(f"Saved {out_name}")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python face_segmentation.py <image>")
        sys.exit(1)
    main(sys.argv[1])