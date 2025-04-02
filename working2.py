# import cv2
# import mediapipe as mp
# import sys

# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# def expand_and_shift_bbox(bbox, scale=1.7, shift_up=0.18, shift_right=0.03):
#     """
#     Expands and shifts the bounding box to capture more of the head (hair, ears, forehead).
    
#     :param bbox: A dictionary with keys: xmin, ymin, width, height (all relative coords in [0,1]).
#     :param scale: How much to scale the bounding box around its center (1.0 = no expansion).
#     :param shift_up: Moves the box upward (fraction of the face's original height).
#     :param shift_right: Moves the box to the right (fraction of the face's original width).
#     :return: A new bounding box dict with updated xmin, ymin, width, height.
#     """
#     x, y, w, h = bbox['xmin'], bbox['ymin'], bbox['width'], bbox['height']
    
#     # Center of the original bounding box
#     center_x = x + w / 2.0
#     center_y = y + h / 2.0
    
#     # Scale the bounding box around the center
#     new_w = w * scale
#     new_h = h * scale
    
#     # Shift upward (negative in y-direction)
#     center_y = center_y - (shift_up * h)
#     # Shift right (positive in x-direction)
#     center_x = center_x + (shift_right * w)
    
#     # Calculate new top-left
#     new_x = center_x - (new_w / 2.0)
#     new_y = center_y - (new_h / 2.0)
    
#     # Clamp to [0, 1] so the box doesn't go outside the image
#     new_x = max(0.0, new_x)
#     new_y = max(0.0, new_y)
#     # Adjust width/height if shifting or scaling goes beyond image bounds
#     new_w = min(1.0 - new_x, new_w)
#     new_h = min(1.0 - new_y, new_h)
    
#     return {
#         'xmin': new_x,
#         'ymin': new_y,
#         'width': new_w,
#         'height': new_h
#     }

# def detect_faces(image_rgb, expand=True):
#     """
#     Detect faces in the input image using Mediapipe.
#     Returns a list of bounding boxes in relative coordinates.
#     If expand=True, expands each bounding box to capture more of the head.
#     """
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
#         results = face_detection.process(image_rgb)
#         face_bboxes = []
#         if results.detections:
#             for detection in results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 bbox_dict = {
#                     'xmin': bbox.xmin,
#                     'ymin': bbox.ymin,
#                     'width': bbox.width,
#                     'height': bbox.height
#                 }
#                 if expand:
#                     # Tweak scale, shift_up, and shift_right as needed
#                     bbox_dict = expand_and_shift_bbox(
#                         bbox_dict,
#                         scale=1.7,      # Increase to capture more area
#                         shift_up=0.18,  # Increase to move further up
#                         shift_right=0.03  # Increase to move further to the right
#                     )
#                 face_bboxes.append(bbox_dict)
#         return face_bboxes

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Unable to read image at {image_path}")
#         sys.exit(1)
    
#     # Convert to RGB for Mediapipe
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
#     # Detect faces with expanded & shifted bounding box
#     faces = detect_faces(image_rgb, expand=True)
#     num_faces = len(faces)
#     print("Detected faces:", num_faces)
    
#     if num_faces == 0:
#         print("No faces detected. Please try another image.")
#     elif num_faces > 1:
#         print("Multiple faces detected. Please use an image with exactly one face.")
#     else:
#         print("One face detected. Proceeding with segmentation in later steps.")
    
#     # Draw the expanded, shifted bounding box
#     h, w, _ = image_bgr.shape
#     for bbox in faces:
#         x1 = int(bbox['xmin'] * w)
#         y1 = int(bbox['ymin'] * h)
#         x2 = int((bbox['xmin'] + bbox['width']) * w)
#         y2 = int((bbox['ymin'] + bbox['height']) * h)
#         cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     # Show the image with the bounding box
#     cv2.imshow("Expanded & Shifted Face Detection", image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection_expanded_shifted.py <image_path>")
#         sys.exit(1)
#     image_path = sys.argv[1]
#     main(image_path)


























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

# def keep_largest_component(mask_255):
#     """
#     Keep only the largest white region in the mask (0 or 255).
#     """
#     mask_bin = (mask_255 > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels <= 1:
#         return mask_255  # no white region or just background

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
#     Fill any holes fully enclosed by the mask. This removes
#     black spots inside the face region IF they're fully enclosed.
#     """
#     inv_mask = cv2.bitwise_not(mask_255)  # invert => background becomes white
#     largest_bg = keep_largest_component(inv_mask)
#     filled_mask = cv2.bitwise_not(largest_bg)
#     return filled_mask

# def unify_hair_if_mislabeled(parsing_map):
#     """
#     Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
#     Also unify adjacent neck(13) or clothes(14) if they were mislabeled as hair.
#     """
#     hair_mask = ((parsing_map == 15) | (parsing_map == 17))
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Slight dilation to catch adjacent neck/clothes mislabeled as hair
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8),
#                               np.ones((3,3), np.uint8), iterations=1)
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def parse_image(image_bgr, model):
#     """
#     Forward pass through BiSeNet, returning:
#       - parsing_map (the segmentation classes)
#       - (orig_h, orig_w) for reference
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape

#     # 1) Resize to 512
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))

#     # 2) Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)

#     # 3) Forward pass
#     with torch.no_grad():
#         outputs = model(tensor)
#         if isinstance(outputs, (tuple, list)):
#             main_output = outputs[0]
#         else:
#             main_output = outputs
        
#         if main_output.ndim == 4:
#             main_output = main_output.squeeze(0)

#         # Argmax
#         if main_output.shape[0] == 19:  # channels-first
#             parsing_map = torch.argmax(main_output, dim=0).cpu().numpy()
#         elif main_output.shape[-1] == 19:  # channels-last
#             parsing_map = torch.argmax(main_output, dim=-1).cpu().numpy()
#         else:
#             print(f"[ERROR] Unexpected shape: {main_output.shape}")
#             sys.exit(1)

#     return parsing_map, orig_h, orig_w

# def face_hair_mask(parsing_map, orig_w, orig_h):
#     """
#     Generate a mask that includes face (1..12) + hair (15,17).
#     Apply morphological cleanup, and return a final binary mask (255 = keep).
#     """
#     # Unify hair if mislabeled
#     corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)

#     # Classes to keep: face (1..12) + hair(15,17)
#     face_classes = list(range(1, 13))  # 1..12
#     hair_classes = [15, 17]
#     keep_classes = face_classes + hair_classes

#     # 512x512 binary
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask  # True/False

#     # Morphological ops at 512x512
#     opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN,
#                               np.ones((3,3), np.uint8), iterations=1)
#     closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE,
#                               np.ones((3,3), np.uint8), iterations=1)
#     largest = keep_largest_component(closed)
#     filled = fill_mask_holes(largest)

#     # Additional morphological close with bigger kernel
#     double_closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE,
#                                      np.ones((7,7), np.uint8), iterations=3)
#     final_512 = keep_largest_component(double_closed)

#     # Resize back to original resolution
#     mask_original = cv2.resize(final_512, (orig_w, orig_h),
#                                interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def get_hair_mask(parsing_map, orig_w, orig_h):
#     """
#     Returns a hair-only mask (255 for hair, 0 otherwise), sized to (orig_h, orig_w).
#     We'll do a small morphological close to unify the hair region.
#     """
#     # Hair is labeled 15 or 17
#     hair_mask_512 = ((parsing_map == 15) | (parsing_map == 17)).astype(np.uint8)*255
    
#     # small morphological close at 512x512
#     kernel = np.ones((5,5), np.uint8)
#     hair_closed_512 = cv2.morphologyEx(hair_mask_512, cv2.MORPH_CLOSE, kernel, iterations=1)
    
#     # resize to original
#     hair_mask_original = cv2.resize(hair_closed_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return hair_mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create an RGBA image (same size as original) where the face/hair region
#     is opaque and the rest is transparent.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def remove_small_alpha_holes(rgba, max_hole_area=800):
#     """
#     Find small alpha=0 connected components fully inside the face bounding box,
#     and force them to become alpha=255. This merges leftover 'holes' into the face region.
#     """
#     alpha = rgba[:, :, 3]
    
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)
    
#     alpha_inv = (alpha == 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_inv, connectivity=8)
    
#     for label_id in range(1, num_labels):
#         area = stats[label_id, cv2.CC_STAT_AREA]
#         left = stats[label_id, cv2.CC_STAT_LEFT]
#         top = stats[label_id, cv2.CC_STAT_TOP]
#         width = stats[label_id, cv2.CC_STAT_WIDTH]
#         height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
#         # Check if this hole is entirely inside the face bounding rect
#         inside_face_box = (
#             left >= x and top >= y and
#             (left + width) <= x + w and
#             (top + height) <= y + h
#         )
#         # If it's small enough, unify it
#         if inside_face_box and (area <= max_hole_area):
#             alpha[labels == label_id] = 255
    
#     rgba[:, :, 3] = alpha
#     return rgba

# def inpaint_alpha(rgba):
#     """
#     After we've merged small alpha holes, we may have no color info for those pixels.
#     We'll inpaint them from the surrounding face region so they blend in nicely.
#     """
#     alpha = rgba[:, :, 3]
#     bgr = rgba[:, :, :3].copy()
    
#     mask = np.zeros(alpha.shape, dtype=np.uint8)
    
#     # Mark inpaint region: alpha=255 but color is near black
#     black_thresh = 80  
#     dark_pixels = np.where(
#         (bgr[:, :, 0] < black_thresh) &
#         (bgr[:, :, 1] < black_thresh) &
#         (bgr[:, :, 2] < black_thresh) &
#         (alpha == 255)
#     )
#     mask[dark_pixels] = 255
    
#     inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
#     final_rgba = np.dstack((inpainted_bgr, alpha))
#     return final_rgba

# def feather_alpha(rgba, radius=2):
#     """
#     Feather (blur) the alpha channel to smooth the edge.
#     """
#     alpha = rgba[:, :, 3].astype(np.float32) / 255.0
#     alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
#     alpha = np.clip(alpha, 0, 1)
#     rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
#     return rgba

# def darken_hair_in_rgba(rgba, hair_mask, dark_factor=0.6):
#     """
#     Darken the pixels in `rgba` (H x W x 4) where hair_mask == 255
#     by multiplying their RGB values by `dark_factor`.
#     """
#     hair_pixels = (hair_mask == 255)
#     # Make sure hair_pixels has same shape as rgba
#     if hair_pixels.shape[:2] == rgba.shape[:2]:
#         rgba[hair_pixels, 0:3] = (rgba[hair_pixels, 0:3].astype(np.float32) * dark_factor).astype(np.uint8)
#     return rgba

# def main(image_path):
#     # 1) Load image
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # 2) Ensure exactly one face
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     if not detect_single_face(image_rgb):
#         print("No face or multiple faces detected. Exiting.")
#         sys.exit(0)
#     else:
#         print("Exactly one face detected.")
    
#     # 3) Load model
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     # 4) Parse image => get parsing_map
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
    
#     # 5) Build the final face+hair mask
#     print("Generating face+hair mask (morphological cleanup)...")
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)

#     # 6) Create a full-size RGBA image (no cropping!)
#     print("Creating RGBA image (full size, transparent background)...")
#     rgba_image = create_rgba_image(image_bgr, final_mask)

#     # 7) Remove small alpha holes => Inpaint => Feather
#     print("Merging small holes in alpha channel...")
#     rgba_image = remove_small_alpha_holes(rgba_image)
    
#     print("Inpainting newly filled areas...")
#     rgba_image = inpaint_alpha(rgba_image)
    
#     print("Feathering alpha edges...")
#     rgba_image = feather_alpha(rgba_image, radius=2)

#     # 8) Optionally darken the hair
#     hair_mask_full = get_hair_mask(parsing_map, orig_w, orig_h)
#     dark_factor = 0.6  # Adjust as needed (0.5 => darker, 0.8 => lighter)
#     rgba_image = darken_hair_in_rgba(rgba_image, hair_mask_full, dark_factor=dark_factor)

#     # 9) Save result
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, rgba_image)
#     print(f"Saved to {out_name}")
    
#     # Display if desired
#     cv2.imshow("Final Output (Darkened Hair)", rgba_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection1.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])
















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

# def keep_largest_component(mask_255):
#     """
#     Keep only the largest white region in the mask (0 or 255).
#     """
#     mask_bin = (mask_255 > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels <= 1:
#         return mask_255  # no white region or just background

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
#     Fill any holes fully enclosed by the mask. This removes
#     black spots inside the face region IF they're fully enclosed.
#     """
#     inv_mask = cv2.bitwise_not(mask_255)  # invert => background becomes white
#     largest_bg = keep_largest_component(inv_mask)
#     filled_mask = cv2.bitwise_not(largest_bg)
#     return filled_mask

# def unify_hair_if_mislabeled(parsing_map):
#     """
#     Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
#     Also unify adjacent neck(13) or clothes(14) if they were mislabeled as hair.
#     """
#     hair_mask = ((parsing_map == 15) | (parsing_map == 17))
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Slight dilation to catch adjacent neck/clothes mislabeled as hair
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8),
#                               np.ones((3,3), np.uint8), iterations=1)
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def parse_image(image_bgr, model):
#     """
#     Forward pass through BiSeNet, returning:
#       - parsing_map (the segmentation classes)
#       - (orig_h, orig_w) for reference
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape

#     # 1) Resize to 512
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))

#     # 2) Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)

#     # 3) Forward pass
#     with torch.no_grad():
#         outputs = model(tensor)
#         if isinstance(outputs, (tuple, list)):
#             main_output = outputs[0]
#         else:
#             main_output = outputs
        
#         if main_output.ndim == 4:
#             main_output = main_output.squeeze(0)

#         # Argmax
#         if main_output.shape[0] == 19:  # channels-first
#             parsing_map = torch.argmax(main_output, dim=0).cpu().numpy()
#         elif main_output.shape[-1] == 19:  # channels-last
#             parsing_map = torch.argmax(main_output, dim=-1).cpu().numpy()
#         else:
#             print(f"[ERROR] Unexpected shape: {main_output.shape}")
#             sys.exit(1)

#     return parsing_map, orig_h, orig_w

# def face_hair_mask(parsing_map, orig_w, orig_h):
#     """
#     Generate a mask that includes face (1..12) + hair (15,17).
#     Apply morphological cleanup, and return a final binary mask (255 = keep).
#     """
#     # Unify hair if mislabeled
#     corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)

#     # Classes to keep: face (1..12) + hair(15,17)
#     face_classes = list(range(1, 13))  # 1..12
#     hair_classes = [15, 17]
#     keep_classes = face_classes + hair_classes

#     # 512x512 binary
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask  # True/False

#     # Morphological ops at 512x512
#     opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN,
#                               np.ones((3,3), np.uint8), iterations=1)
#     closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE,
#                               np.ones((3,3), np.uint8), iterations=1)
#     largest = keep_largest_component(closed)
#     filled = fill_mask_holes(largest)

#     # Additional morphological close with bigger kernel
#     double_closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE,
#                                      np.ones((7,7), np.uint8), iterations=3)
#     final_512 = keep_largest_component(double_closed)

#     # Resize back to original resolution
#     mask_original = cv2.resize(final_512, (orig_w, orig_h),
#                                interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def get_hair_mask(parsing_map, orig_w, orig_h):
#     """
#     Returns a hair-only mask (255 for hair, 0 otherwise), sized to (orig_h, orig_w).
#     We'll do a small morphological close to unify the hair region.
#     """
#     # Hair is labeled 15 or 17
#     hair_mask_512 = ((parsing_map == 15) | (parsing_map == 17)).astype(np.uint8)*255
    
#     # small morphological close at 512x512
#     kernel = np.ones((5,5), np.uint8)
#     hair_closed_512 = cv2.morphologyEx(hair_mask_512, cv2.MORPH_CLOSE, kernel, iterations=1)
    
#     # resize to original
#     hair_mask_original = cv2.resize(hair_closed_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return hair_mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create an RGBA image (same size as original) where the face/hair region
#     is opaque and the rest is transparent.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def remove_small_alpha_holes(rgba, max_hole_area=800):
#     """
#     Find small alpha=0 connected components fully inside the face bounding box,
#     and force them to become alpha=255. This merges leftover 'holes' into the face region.
#     """
#     alpha = rgba[:, :, 3]
    
#     coords = cv2.findNonZero(alpha)
#     if coords is None:
#         return rgba
#     x, y, w, h = cv2.boundingRect(coords)
    
#     alpha_inv = (alpha == 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_inv, connectivity=8)
    
#     for label_id in range(1, num_labels):
#         area = stats[label_id, cv2.CC_STAT_AREA]
#         left = stats[label_id, cv2.CC_STAT_LEFT]
#         top = stats[label_id, cv2.CC_STAT_TOP]
#         width = stats[label_id, cv2.CC_STAT_WIDTH]
#         height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
#         # Check if this hole is entirely inside the face bounding rect
#         inside_face_box = (
#             left >= x and top >= y and
#             (left + width) <= x + w and
#             (top + height) <= y + h
#         )
#         # If it's small enough, unify it
#         if inside_face_box and (area <= max_hole_area):
#             alpha[labels == label_id] = 255
    
#     rgba[:, :, 3] = alpha
#     return rgba

# def inpaint_alpha(rgba):
#     """
#     After we've merged small alpha holes, we may have no color info for those pixels.
#     We'll inpaint them from the surrounding face region so they blend in nicely.
#     """
#     alpha = rgba[:, :, 3]
#     bgr = rgba[:, :, :3].copy()
    
#     mask = np.zeros(alpha.shape, dtype=np.uint8)
    
#     # Mark inpaint region: alpha=255 but color is near black
#     black_thresh = 80  
#     dark_pixels = np.where(
#         (bgr[:, :, 0] < black_thresh) &
#         (bgr[:, :, 1] < black_thresh) &
#         (bgr[:, :, 2] < black_thresh) &
#         (alpha == 255)
#     )
#     mask[dark_pixels] = 255
    
#     inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
#     final_rgba = np.dstack((inpainted_bgr, alpha))
#     return final_rgba

# def feather_alpha(rgba, radius=2):
#     """
#     Feather (blur) the alpha channel to smooth the edge.
#     """
#     alpha = rgba[:, :, 3].astype(np.float32) / 255.0
#     alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
#     alpha = np.clip(alpha, 0, 1)
#     rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
#     return rgba

# def darken_hair_in_rgba(rgba, hair_mask, dark_factor=0.6):
#     """
#     Darken the pixels in `rgba` (H x W x 4) where hair_mask == 255
#     by multiplying their RGB values by `dark_factor`.
#     """
#     hair_pixels = (hair_mask == 255)
#     # Make sure hair_pixels has same shape as rgba
#     if hair_pixels.shape[:2] == rgba.shape[:2]:
#         rgba[hair_pixels, 0:3] = (rgba[hair_pixels, 0:3].astype(np.float32) * dark_factor).astype(np.uint8)
#     return rgba

# def main(image_path):
#     # 1) Load image
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # 2) Ensure exactly one face
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     if not detect_single_face(image_rgb):
#         print("No face or multiple faces detected. Exiting.")
#         sys.exit(0)
#     else:
#         print("Exactly one face detected.")
    
#     # 3) Load model
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     # 4) Parse image => get parsing_map
#     parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
    
#     # 5) Build the final face+hair mask
#     print("Generating face+hair mask (morphological cleanup)...")
#     final_mask = face_hair_mask(parsing_map, orig_w, orig_h)

#     # 6) Create a full-size RGBA image (no cropping!)
#     print("Creating RGBA image (full size, transparent background)...")
#     rgba_image = create_rgba_image(image_bgr, final_mask)

#     # 7) Remove small alpha holes => Inpaint => Feather
#     print("Merging small holes in alpha channel...")
#     rgba_image = remove_small_alpha_holes(rgba_image)
    
#     print("Inpainting newly filled areas...")
#     rgba_image = inpaint_alpha(rgba_image)
    
#     print("Feathering alpha edges...")
#     rgba_image = feather_alpha(rgba_image, radius=2)

#     # 8) Optionally darken the hair
#     hair_mask_full = get_hair_mask(parsing_map, orig_w, orig_h)
#     dark_factor = 0.6  # Adjust as needed (0.5 => darker, 0.8 => lighter)
#     rgba_image = darken_hair_in_rgba(rgba_image, hair_mask_full, dark_factor=dark_factor)

#     # 9) Save result as PNG (no window display)
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, rgba_image)
#     print(f"Saved to {out_name}")

#     # Removed the cv2.imshow and related lines so no GUI window appears.

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection1.py <image_path>")
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

def keep_largest_component(mask_255):
    """
    Keep only the largest white region in the mask (0 or 255).
    """
    mask_bin = (mask_255 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_255  # no white region or just background

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
    Fill any holes fully enclosed by the mask. 
    """
    inv_mask = cv2.bitwise_not(mask_255)
    largest_bg = keep_largest_component(inv_mask)
    filled_mask = cv2.bitwise_not(largest_bg)
    return filled_mask

def unify_hair_if_mislabeled(parsing_map):
    """
    Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
    Also unify adjacent neck(13) or clothes(14) if mislabeled as hair.
    """
    hair_mask = ((parsing_map == 15) | (parsing_map == 17))
    neck_mask = (parsing_map == 13)
    clothes_mask = (parsing_map == 14)

    # Slight dilation to catch adjacent neck/clothes mislabeled as hair
    hair_dilated = cv2.dilate(hair_mask.astype(np.uint8),
                              np.ones((3,3), np.uint8), iterations=1)
    adjacent_neck = neck_mask & (hair_dilated > 0)
    adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
    corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
    return corrected_hair_mask

def parse_image(image_bgr, model):
    """
    Forward pass through BiSeNet, returning:
      - parsing_map (the segmentation classes)
      - (orig_h, orig_w) for reference
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image_rgb.shape

    # 1) Resize to 512
    input_size = 512
    image_resized = cv2.resize(image_rgb, (input_size, input_size))

    # 2) Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tensor = transform(image_resized).unsqueeze(0)

    # 3) Forward pass
    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, (tuple, list)):
            main_output = outputs[0]
        else:
            main_output = outputs
        
        if main_output.ndim == 4:
            main_output = main_output.squeeze(0)

        # Argmax
        if main_output.shape[0] == 19:  # channels-first
            parsing_map = torch.argmax(main_output, dim=0).cpu().numpy()
        elif main_output.shape[-1] == 19:  # channels-last
            parsing_map = torch.argmax(main_output, dim=-1).cpu().numpy()
        else:
            print(f"[ERROR] Unexpected shape: {main_output.shape}")
            sys.exit(1)

    return parsing_map, orig_h, orig_w

def face_hair_mask(parsing_map, orig_w, orig_h):
    """
    Generate a mask that includes face (1..12) + hair (15,17).
    Apply minimal morphological cleanup, and return a final binary mask (255 = keep).
    """
    # Unify hair if mislabeled
    corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)

    # Classes to keep: face (1..12) + hair(15,17)
    face_classes = list(range(1, 13))  # 1..12
    hair_classes = [15, 17]
    keep_classes = face_classes + hair_classes

    # 512x512 binary
    mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    face_mask = (mask_512 > 0)
    combined_mask = face_mask | corrected_hair_mask  # True/False

    # --- Light morphological ops (small open & close) ---
    opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((3,3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE,
                              np.ones((3,3), np.uint8), iterations=1)

    # Keep largest region, fill holes
    largest = keep_largest_component(closed)
    filled = fill_mask_holes(largest)

    # NOTE: We remove the "double close" with a (7,7) kernel here 
    # to avoid losing too much hair detail.

    final_512 = keep_largest_component(filled)

    # Resize back to original resolution
    mask_original = cv2.resize(final_512, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    return mask_original

def create_rgba_image(original_bgr, mask):
    """
    Create an RGBA image (same size as original) 
    where the face/hair region is opaque and the rest is transparent.
    """
    orig_h, orig_w, _ = original_bgr.shape
    rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    face_pixels = (mask == 255)
    rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
    rgba_image[face_pixels, 3] = 255
    return rgba_image

# ------------------------------------------------------------------------
# We will COMMENT OUT the following two steps or make them optional.
# They sometimes cause those unwanted “spots” or over-filling in hair.
# ------------------------------------------------------------------------

def remove_small_alpha_holes(rgba, max_hole_area=800):
    """
    Find small alpha=0 connected components fully inside the face bounding box,
    and force them to become alpha=255.
    """
    alpha = rgba[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return rgba

    x, y, w, h = cv2.boundingRect(coords)
    alpha_inv = (alpha == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_inv, connectivity=8)
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
        inside_face_box = (
            left >= x and top >= y and
            (left + width) <= x + w and
            (top + height) <= y + h
        )
        if inside_face_box and (area <= max_hole_area):
            alpha[labels == label_id] = 255
    
    rgba[:, :, 3] = alpha
    return rgba

def inpaint_alpha(rgba):
    """
    Inpaint areas that became alpha=255 but had no color info. 
    This can be risky if misapplied to hair, so let's skip for now.
    """
    alpha = rgba[:, :, 3]
    bgr = rgba[:, :, :3].copy()
    
    mask = np.zeros(alpha.shape, dtype=np.uint8)
    black_thresh = 80  
    dark_pixels = np.where(
        (bgr[:, :, 0] < black_thresh) &
        (bgr[:, :, 1] < black_thresh) &
        (bgr[:, :, 2] < black_thresh) &
        (alpha == 255)
    )
    mask[dark_pixels] = 255
    
    inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    final_rgba = np.dstack((inpainted_bgr, alpha))
    return final_rgba

def feather_alpha(rgba, radius=2):
    """
    Feather (blur) the alpha channel to smooth the edge.
    """
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), radius)
    alpha = np.clip(alpha, 0, 1)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return rgba

# ------------------------------------------------------------------------
# We REMOVE the hair darkening step (or set dark_factor=1.0)
# ------------------------------------------------------------------------
def darken_hair_in_rgba(rgba, hair_mask, dark_factor=1.0):
    """
    By default, do NOT darken the hair (dark_factor=1.0 => no change).
    """
    hair_pixels = (hair_mask == 255)
    if hair_pixels.shape[:2] == rgba.shape[:2]:
        rgba[hair_pixels, 0:3] = (rgba[hair_pixels, 0:3].astype(np.float32) * dark_factor).astype(np.uint8)
    return rgba

def main(image_path):
    # 1) Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Cannot read {image_path}")
        sys.exit(1)
    
    # 2) Ensure exactly one face
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not detect_single_face(image_rgb):
        print("No face or multiple faces detected. Exiting.")
        sys.exit(0)
    else:
        print("Exactly one face detected.")
    
    # 3) Load model
    print("Loading BiSeNet model...")
    model = load_bisenet_model('79999_iter.pth', 19)
    
    # 4) Parse image => get parsing_map
    parsing_map, orig_h, orig_w = parse_image(image_bgr, model)
    
    # 5) Build the final face+hair mask (light morphological cleanup)
    print("Generating face+hair mask (light morphological ops)...")
    final_mask = face_hair_mask(parsing_map, orig_w, orig_h)

    # 6) Create a full-size RGBA image (no cropping!)
    rgba_image = create_rgba_image(image_bgr, final_mask)

    # ---------------------------------------------------------------------
    # 7) Optionally remove small alpha holes & inpaint 
    #    (Comment out or reduce usage if it causes issues in hair region)
    # ---------------------------------------------------------------------
    # rgba_image = remove_small_alpha_holes(rgba_image, max_hole_area=300)  # smaller hole area if you want to be safer
    # rgba_image = inpaint_alpha(rgba_image)

    # 8) Feather alpha edges slightly
    rgba_image = feather_alpha(rgba_image, radius=2)

    # 9) (OPTIONAL) “Darken hair” => but here we do factor=1.0 => no darkening
    #    If you want to keep the original hair color, do factor=1.0
    #    If you do want slight darkening, set factor=0.9 or 0.8, etc.
    # hair_mask_full = get_hair_mask(parsing_map, orig_w, orig_h)
    # rgba_image = darken_hair_in_rgba(rgba_image, hair_mask_full, dark_factor=1.0)

    # 10) Save result as PNG
    out_name = "face_hair_segmented.png"
    cv2.imwrite(out_name, rgba_image)
    print(f"Saved to {out_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_detection1.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
