


# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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

# def fill_small_holes_and_keep_largest(mask_512):
#     """
#     1) Apply morphological closing with a larger kernel multiple times to fill holes.
#     2) Optionally apply dilation.
#     3) Keep only the largest connected white region (the main face), removing stray areas.
#     """
#     # 1. Morphological Closing (fills holes)
#     # Increase kernel size or iterations if small black spots remain
#     kernel = np.ones((7,7), np.uint8)  # bigger kernel => more aggressive fill
#     mask_close = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=2)
    
#     # 2. Optional Dilation to further fill holes
#     mask_dilate = cv2.dilate(mask_close, kernel, iterations=1)
    
#     # 3. Keep only the largest connected component
#     # Convert from [0,255] mask to [0,1] for connectedComponents
#     mask_binary = (mask_dilate > 0).astype(np.uint8)
    
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
#     if num_labels <= 1:
#         # Means there's either no white region or just background
#         return mask_dilate  # nothing to keep
    
#     # Find the largest white region (excluding label=0 which is background)
#     largest_label = 1
#     max_area = stats[1, cv2.CC_STAT_AREA]
#     for label_id in range(2, num_labels):
#         area = stats[label_id, cv2.CC_STAT_AREA]
#         if area > max_area:
#             max_area = area
#             largest_label = label_id
    
#     # Rebuild a mask with only the largest connected component
#     final_mask = np.zeros_like(mask_binary, dtype=np.uint8)
#     final_mask[labels == largest_label] = 255  # keep largest component as 255
    
#     # Convert back to [0,255]
#     return final_mask

# def face_hair_mask(image_bgr, model):
#     """
#     Generates a binary mask (255=face+hair+eyes etc., 0=background), then post-processes
#     to fill small black holes around lips, eyes, etc.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize to 512x512
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Convert to tensor & normalize
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
#     with torch.no_grad():
#         outputs = model(tensor)
#         if isinstance(outputs, (tuple, list)):
#             main_output = outputs[0]
#         else:
#             main_output = outputs
        
#         # Squeeze batch => shape (19,512,512) or (512,512,19)
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
    
#     # 4. Keep classes for face, eyes, brows, hair, etc.
#     keep_classes = [1,2,3,4,5,6,7,8,9,10,11,12,15]
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255
    
#     # 5. Fill small holes and keep largest connected region
#     mask_512_filled = fill_small_holes_and_keep_largest(mask_512)
    
#     # 6. Resize mask back to original image size
#     mask_original = cv2.resize(mask_512_filled, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # Ensure exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces detected. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating mask with post-processing...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Create RGBA
#     image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
#     image_rgba[..., 3] = mask
    
#     # Zero-out background color
#     bg_indices = (mask == 0)
#     image_rgba[bg_indices, 0] = 0
#     image_rgba[bg_indices, 1] = 0
#     image_rgba[bg_indices, 2] = 0
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, image_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation", image_rgba)
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

# # If needed, add your face-parsing.PyTorch folder:
# # sys.path.append("./face-parsing.PyTorch")

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

# def fill_small_holes(mask_512):
#     """
#     Apply morphological closing to fill small holes in the face region.
#     If you still see black spots, try increasing the kernel size or iterations.
#     """
#     kernel = np.ones((5,5), np.uint8)
#     mask_closed = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=2)
#     return mask_closed

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask (255=face/hair, 0=background) with morphological closing to fill holes.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize to 512x512
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)  # (1,3,512,512)
    
#     # 3. Forward pass
#     with torch.no_grad():
#         outputs = model(tensor)
#         if isinstance(outputs, (tuple, list)):
#             main_output = outputs[0]
#         else:
#             main_output = outputs
        
#         # Squeeze batch => shape (19,512,512) or (512,512,19)
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
    
#     # 4. Keep classes for face, eyes, brows, lips, hair, etc.
#     #    1=skin,2/3=brows,4/5=eyes,6=glasses,7/8=ears,9=earrings,10/11/12=mouth,15=hair
#     keep_classes = [1,2,3,4,5,6,7,8,9,10,11,12,15]
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8) * 255
    
#     # 5. Fill small holes
#     mask_512_closed = fill_small_holes(mask_512)
    
#     # 6. Resize back to original
#     mask_original = cv2.resize(mask_512_closed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create a fresh RGBA image with original colors only where mask=255, alpha=255.
#     Else alpha=0.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     # Initialize a blank RGBA canvas
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    
#     # Where mask=255, copy the BGR channels & set alpha=255
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
    
#     return rgba_image

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read image from {image_path}")
#         sys.exit(1)
    
#     # Check exactly one face
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     if not detect_single_face(image_rgb):
#         print("No face or multiple faces detected. Exiting.")
#         sys.exit(0)
#     else:
#         print("Exactly one face detected.")
    
#     # Load model
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     # Generate mask
#     print("Generating face+hair mask with morphological closing...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Create a fresh RGBA image (preserves original color)
#     image_rgba = create_rgba_image(image_bgr, mask)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, image_rgba)
#     print(f"Saved to {out_name}")
    
#     # Display
#     cv2.imshow("Face+Hair Segmentation", image_rgba)
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

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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

# def fill_small_holes_and_expand_hair(mask_512):
#     """
#     1) Morphological closing with a larger kernel to fill black holes around eyes/lips.
#     2) Morphological opening with a smaller kernel to remove small extrusions.
#     3) Optionally, expand hair region slightly if needed.
#     4) Keep only the largest connected component (the main face).
#     """
#     # 1. Morphological Closing
#     close_kernel = np.ones((7,7), np.uint8)  # Larger kernel => more hole filling
#     mask_close = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
#     # 2. Morphological Opening
#     open_kernel = np.ones((3,3), np.uint8)  # Smaller kernel => remove small specks
#     mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, open_kernel, iterations=1)
    
#     # 3. Expand hair region slightly (class=15).
#     #    We'll do this by separating hair from the rest, dilating it, then recombining.
#     hair_only = (mask_open == 255) & (g_hair_mask == 255)  # We'll define g_hair_mask globally
#     hair_dilated = cv2.dilate(hair_only.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
#     # Combine hair back
#     combined = ((mask_open > 0) | (hair_dilated > 0)).astype(np.uint8)*255
    
#     # 4. Keep only the largest connected component
#     final_mask = keep_largest_component(combined)
#     return final_mask

# def keep_largest_component(mask_255):
#     """
#     Keep only the largest white region in the mask.
#     """
#     mask_bin = (mask_255 > 0).astype(np.uint8)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels <= 1:
#         return mask_255  # nothing to keep

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

# # We'll define a global mask for hair to differentiate it from other classes
# g_hair_mask = None

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask (255=face+hair, 0=background) with advanced morphological post-processing.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize to 512x512
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
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
    
#     # 4. Keep classes for face, eyes, brows, lips, hair, etc.
#     keep_classes = [1,2,3,4,5,6,7,8,9,10,11,12,15]
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
#     # Also create a hair_only mask (class=15) for the hair expansion step
#     global g_hair_mask
#     hair_512 = (parsing_map == 15).astype(np.uint8)*255
#     g_hair_mask = hair_512  # We'll use it in fill_small_holes_and_expand_hair
    
#     # 5. Post-processing
#     mask_512_processed = fill_small_holes_and_expand_hair(mask_512)
    
#     # 6. Resize final mask back
#     mask_original = cv2.resize(mask_512_processed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create a fresh RGBA image from scratch. Copy original BGR where mask=255, alpha=255.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read image: {image_path}")
#         sys.exit(1)
    
#     # Check exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces detected. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating face+hair mask with advanced post-processing...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Create a fresh RGBA with original color
#     image_rgba = create_rgba_image(image_bgr, mask)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, image_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation", image_rgba)
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

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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
#     Keep only the largest white region in the mask.
#     """
#     mask_bin = (mask_255 > 0).astype(np.uint8)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
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

# def unify_hair_neck_clothes(parsing_map):
#     """
#     If hair is mislabeled as neck(13) or clothes(14), unify them with hair(15)
#     if they are adjacent. We'll produce a mask that includes hair + 
#     any neck/clothes pixels that touch hair.
#     """
#     # shape: (512, 512)
#     hair_mask = (parsing_map == 15)
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Combine them if they are adjacent to hair. We'll do a dilation on hair
#     # to pick up adjacent neck/clothes pixels, then unify them.
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
    
#     # Now, if neck or clothes are adjacent to the dilated hair, unify them.
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     # We'll unify those pixels with hair. This helps if hair is mislabeled as neck/clothes.
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask that includes face, eyes, lips, and hair. We also allow 
#     neck(13) or clothes(14) in case hair is mislabeled, then unify them with hair if adjacent.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
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
    
#     # 4. Keep classes for face + eyes + mouth + etc. + hair + neck + clothes
#     #    We keep 13 & 14 in case hair is mislabeled as neck/clothes.
#     keep_classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
#     # 5. Unify hair with adjacent neck/clothes if mislabeled
#     corrected_hair_mask = unify_hair_neck_clothes(parsing_map)  # boolean mask for hair region
#     # Combine the corrected hair mask with the rest of face region
#     # Because the rest of face includes skin(1), eyes(4,5), lips(10..12), etc.
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask
    
#     # 6. Morphological closing to remove black holes
#     kernel_close = np.ones((7,7), np.uint8)
#     combined_closed = cv2.morphologyEx(combined_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
#     # 7. Keep largest connected component (the main face + hair)
#     largest = keep_largest_component(combined_closed)
    
#     # 8. Resize back
#     mask_original = cv2.resize(largest, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create a fresh RGBA. Copy original color where mask=255, alpha=255, else alpha=0.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # Ensure exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating face+hair mask with hair recovery...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Create RGBA
#     image_rgba = create_rgba_image(image_bgr, mask)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, image_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation", image_rgba)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection1.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])











#WORKING CODE :-



# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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
#     Keep only the largest white region in the mask.
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

# def unify_hair_if_mislabeled(parsing_map):
#     """
#     Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
#     If hair is mislabeled as neck(13) or clothes(14), unify them if adjacent.
#     This DOES NOT keep neck/clothes in general—only the parts that are truly hair.
#     """
#     # Combine 15 & 17 as hair
#     hair_mask = ((parsing_map == 15) | (parsing_map == 17))
    
#     # Potentially mislabeled hair in neck/clothes
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Dilate hair to catch adjacent neck/clothes
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
    
#     # If neck or clothes is adjacent to hair, unify that region
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask that includes only face (1..12) + hair(15/17).
#     We do NOT keep 13(neck) or 14(clothes), except if mislabeled hair is adjacent.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
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
    
#     # 4. Keep face classes 1..12 plus hair classes (15,17). 
#     #    We intentionally EXCLUDE 13 (neck) and 14 (clothes).
#     face_classes = list(range(1, 13))  # 1..12
#     hair_classes = [15, 17]
#     keep_classes = face_classes + hair_classes
#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
#     # 5. Unify hair if mislabeled as neck(13) or clothes(14)
#     corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)
    
#     # Combine corrected hair with the face region
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask
    
#     # 6. Morphological DILATION (bigger kernel, more iterations) to expand hair
#     kernel_dilate = np.ones((7,7), np.uint8)
#     combined_dilated = cv2.dilate(combined_mask.astype(np.uint8), kernel_dilate, iterations=3)
    
#     # 7. Morphological closing to fill holes
#     kernel_close = np.ones((7,7), np.uint8)
#     combined_closed = cv2.morphologyEx(combined_dilated*255, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
#     # 8. Keep largest connected component (main face + hair)
#     largest = keep_largest_component(combined_closed)
    
#     # 9. Resize back
#     mask_original = cv2.resize(largest, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create an RGBA image where the face/hair region is opaque and the rest is transparent.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def crop_face_hair(image_bgr, mask):
#     """
#     Crop the image and mask to the bounding box of the non-zero region
#     so the output is a 'floating head' with minimal background.
#     """
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         return None, None

#     x, y, w, h = cv2.boundingRect(coords)

#     # Increase margin so we don't cut off hair
#     margin = 20
#     max_y, max_x = image_bgr.shape[:2]
#     x1 = max(x - margin, 0)
#     y1 = max(y - margin, 0)
#     x2 = min(x + w + margin, max_x)
#     y2 = min(y + h + margin, max_y)

#     cropped_bgr = image_bgr[y1:y2, x1:x2]
#     cropped_mask = mask[y1:y2, x1:x2]
#     return cropped_bgr, cropped_mask

# def create_cropped_rgba(cropped_bgr, cropped_mask):
#     """
#     Create an RGBA image from the cropped region, with transparency outside the face/hair.
#     """
#     if cropped_bgr is None or cropped_mask is None:
#         return None

#     h, w = cropped_mask.shape[:2]
#     rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
#     face_pixels = (cropped_mask == 255)
#     rgba_image[face_pixels, 0:3] = cropped_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # Ensure exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating face+hair mask (excluding neck/shirt)...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Crop to just the face/hair region
#     cropped_bgr, cropped_mask = crop_face_hair(image_bgr, mask)
#     if cropped_bgr is None:
#         print("No valid face/hair region found.")
#         sys.exit(0)
    
#     # Create RGBA from the cropped region
#     cropped_rgba = create_cropped_rgba(cropped_bgr, cropped_mask)
#     if cropped_rgba is None:
#         print("Cropping failed.")
#         sys.exit(0)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, cropped_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation (Cropped)", cropped_rgba)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection1.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])










# #WORKING CODE WITH BALCK SPOTS

# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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
#     Keep only the largest white region in the mask.
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

# def unify_hair_if_mislabeled(parsing_map):
#     """
#     Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
#     If hair is mislabeled as neck(13) or clothes(14), unify them if adjacent.
#     This does NOT keep neck/clothes, except where they're actually hair.
#     """
#     # Combine 15 & 17 as hair
#     hair_mask = ((parsing_map == 15) | (parsing_map == 17))
    
#     # Potentially mislabeled hair in neck/clothes
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Dilate hair slightly to catch adjacent neck/clothes
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
    
#     # If neck or clothes is adjacent to hair, unify that region
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask that includes only the face region (classes 1..12) + hair (15,17).
#     We exclude neck(13) and clothes(14), except if they are actually mislabeled hair.
#     Then we do smaller morphological operations for a tighter, smoother boundary.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
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
    
#     # 4. Keep face classes 1..12 plus hair classes (15,17)
#     face_classes = list(range(1, 13))  # 1..12
#     hair_classes = [15, 17]
#     keep_classes = face_classes + hair_classes

#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
#     # 5. Unify hair if mislabeled as neck(13) or clothes(14)
#     corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)
    
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask
    
#     # 6. Morphological OPEN to remove small specks & smooth edges
#     kernel_open = np.ones((3,3), np.uint8)
#     opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=1)
    
#     # 7. Morphological CLOSE with a small kernel for a tighter boundary
#     kernel_close = np.ones((3,3), np.uint8)
#     closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
#     # 8. Keep largest connected component (main face + hair)
#     largest = keep_largest_component(closed)
    
#     # 9. Resize back
#     mask_original = cv2.resize(largest, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create an RGBA image where the face/hair region is opaque and the rest is transparent.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def crop_face_hair(image_bgr, mask):
#     """
#     Crop the image and mask to the bounding box of the non-zero region
#     so the output is a 'floating head' with minimal background.
#     """
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         return None, None

#     x, y, w, h = cv2.boundingRect(coords)

#     # A smaller margin than before, for a tighter crop
#     margin = 10
#     max_y, max_x = image_bgr.shape[:2]
#     x1 = max(x - margin, 0)
#     y1 = max(y - margin, 0)
#     x2 = min(x + w + margin, max_x)
#     y2 = min(y + h + margin, max_y)

#     cropped_bgr = image_bgr[y1:y2, x1:x2]
#     cropped_mask = mask[y1:y2, x1:x2]
#     return cropped_bgr, cropped_mask

# def create_cropped_rgba(cropped_bgr, cropped_mask):
#     """
#     Create an RGBA image from the cropped region, with transparency outside the face/hair.
#     """
#     if cropped_bgr is None or cropped_mask is None:
#         return None

#     h, w = cropped_mask.shape[:2]
#     rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
#     face_pixels = (cropped_mask == 255)
#     rgba_image[face_pixels, 0:3] = cropped_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # Ensure exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating face+hair mask (excluding neck/shirt)...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Crop to just the face/hair region
#     cropped_bgr, cropped_mask = crop_face_hair(image_bgr, mask)
#     if cropped_bgr is None:
#         print("No valid face/hair region found.")
#         sys.exit(0)
    
#     # Create RGBA from the cropped region
#     cropped_rgba = create_cropped_rgba(cropped_bgr, cropped_mask)
#     if cropped_rgba is None:
#         print("Cropping failed.")
#         sys.exit(0)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, cropped_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation (Cropped)", cropped_rgba)
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

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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
#     black spots inside the face region (e.g., inside the lips or eyes).
#     """
#     # mask_255 is 0 or 255. We'll invert it, keep the largest component
#     # (which corresponds to the outside background), then invert again.
#     inv_mask = cv2.bitwise_not(mask_255)  # 0->255, 255->0
#     # Keep largest connected component in the inverted image => outside background
#     largest_bg = keep_largest_component(inv_mask)
#     # Invert back => holes inside the face are now filled
#     filled_mask = cv2.bitwise_not(largest_bg)
#     return filled_mask

# def unify_hair_if_mislabeled(parsing_map):
#     """
#     Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
#     If hair is mislabeled as neck(13) or clothes(14), unify them if adjacent.
#     """
#     hair_mask = ((parsing_map == 15) | (parsing_map == 17))
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Slight dilation to catch adjacent neck/clothes mislabeled as hair
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask that includes face (1..12) + hair (15,17).
#     We exclude neck(13)/clothes(14), except if mislabeled hair is adjacent.
#     Then we fill small holes so the face is solid (no black spots in mouth/eye).
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
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
    
#     # 4. Keep face classes 1..12 plus hair classes (15,17)
#     face_classes = list(range(1, 13))  # 1..12
#     hair_classes = [15, 17]
#     keep_classes = face_classes + hair_classes

#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
#     # 5. Unify hair if mislabeled as neck/clothes
#     corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)
    
#     # Combine face region with corrected hair region
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask
    
#     # 6. Morphological opening to remove small specks
#     kernel_open = np.ones((3,3), np.uint8)
#     opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=1)
    
#     # 7. Morphological closing to smooth edges
#     kernel_close = np.ones((3,3), np.uint8)
#     closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
#     # 8. Keep largest connected component
#     largest = keep_largest_component(closed)
    
#     # 9. Fill any holes inside the face region
#     filled = fill_mask_holes(largest)
    
#     # 10. Resize back
#     mask_original = cv2.resize(filled, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create an RGBA image where the face/hair region is opaque and the rest is transparent.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def crop_face_hair(image_bgr, mask):
#     """
#     Crop the image and mask to the bounding box of the non-zero region
#     so the output is a 'floating head' with minimal background.
#     """
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         return None, None

#     x, y, w, h = cv2.boundingRect(coords)

#     # A small margin so we don't chop off hair
#     margin = 10
#     max_y, max_x = image_bgr.shape[:2]
#     x1 = max(x - margin, 0)
#     y1 = max(y - margin, 0)
#     x2 = min(x + w + margin, max_x)
#     y2 = min(y + h + margin, max_y)

#     cropped_bgr = image_bgr[y1:y2, x1:x2]
#     cropped_mask = mask[y1:y2, x1:x2]
#     return cropped_bgr, cropped_mask

# def create_cropped_rgba(cropped_bgr, cropped_mask):
#     """
#     Create an RGBA image from the cropped region, with transparency outside the face/hair.
#     """
#     if cropped_bgr is None or cropped_mask is None:
#         return None

#     h, w = cropped_mask.shape[:2]
#     rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
#     face_pixels = (cropped_mask == 255)
#     rgba_image[face_pixels, 0:3] = cropped_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # Ensure exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating face+hair mask (removing black spots in lips/eyes)...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Crop to just the face/hair region
#     cropped_bgr, cropped_mask = crop_face_hair(image_bgr, mask)
#     if cropped_bgr is None:
#         print("No valid face/hair region found.")
#         sys.exit(0)
    
#     # Create RGBA from the cropped region
#     cropped_rgba = create_cropped_rgba(cropped_bgr, cropped_mask)
#     if cropped_rgba is None:
#         print("Cropping failed.")
#         sys.exit(0)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, cropped_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation (Cropped)", cropped_rgba)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection1.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])






#WORKING BUT DARK SPOT ON THE LEFT EYE

# import sys
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import mediapipe as mp

# # If needed:
# # sys.path.append("./face-parsing.PyTorch")
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
#     If hair is mislabeled as neck(13) or clothes(14), unify them if adjacent.
#     """
#     hair_mask = ((parsing_map == 15) | (parsing_map == 17))
#     neck_mask = (parsing_map == 13)
#     clothes_mask = (parsing_map == 14)

#     # Slight dilation to catch adjacent neck/clothes mislabeled as hair
#     hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
#     adjacent_neck = neck_mask & (hair_dilated > 0)
#     adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
#     corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
#     return corrected_hair_mask

# def face_hair_mask(image_bgr, model):
#     """
#     Generate a mask that includes face (1..12) + hair (15,17).
#     Exclude neck(13)/clothes(14), except if mislabeled hair is adjacent.
#     Then fill small holes, and do morphological ops for a clean boundary.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     orig_h, orig_w, _ = image_rgb.shape
    
#     # 1. Resize
#     input_size = 512
#     image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
#     # 2. Preprocess
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     tensor = transform(image_resized).unsqueeze(0)
    
#     # 3. Forward pass
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
    
#     # 4. Keep face classes 1..12 plus hair classes (15,17)
#     face_classes = list(range(1, 13))  # 1..12
#     hair_classes = [15, 17]
#     keep_classes = face_classes + hair_classes

#     mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
#     # 5. Unify hair if mislabeled as neck/clothes
#     corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)
    
#     # Combine face region with corrected hair region
#     face_mask = (mask_512 > 0)
#     combined_mask = face_mask | corrected_hair_mask
    
#     # 6. Morphological opening to remove small specks
#     kernel_open = np.ones((3,3), np.uint8)
#     opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=1)
    
#     # 7. Morphological closing to smooth edges
#     kernel_close = np.ones((3,3), np.uint8)
#     closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
#     # 8. Keep largest connected component
#     largest = keep_largest_component(closed)
    
#     # 9. Fill any holes inside the face region
#     filled = fill_mask_holes(largest)
    
#     # 10. Additional morphological close with bigger kernel to remove leftover spots
#     kernel_close2 = np.ones((5,5), np.uint8)
#     double_closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close2, iterations=2)
#     final_mask = keep_largest_component(double_closed)
    
#     # 11. Resize back
#     mask_original = cv2.resize(final_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#     return mask_original

# def create_rgba_image(original_bgr, mask):
#     """
#     Create an RGBA image where the face/hair region is opaque and the rest is transparent.
#     """
#     orig_h, orig_w, _ = original_bgr.shape
#     rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
#     face_pixels = (mask == 255)
#     rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def crop_face_hair(image_bgr, mask):
#     """
#     Crop the image and mask to the bounding box of the non-zero region
#     so the output is a 'floating head' with minimal background.
#     """
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         return None, None

#     x, y, w, h = cv2.boundingRect(coords)

#     # A small margin so we don't chop off hair
#     margin = 10
#     max_y, max_x = image_bgr.shape[:2]
#     x1 = max(x - margin, 0)
#     y1 = max(y - margin, 0)
#     x2 = min(x + w + margin, max_x)
#     y2 = min(y + h + margin, max_y)

#     cropped_bgr = image_bgr[y1:y2, x1:x2]
#     cropped_mask = mask[y1:y2, x1:x2]
#     return cropped_bgr, cropped_mask

# def create_cropped_rgba(cropped_bgr, cropped_mask):
#     """
#     Create an RGBA image from the cropped region, with transparency outside the face/hair.
#     """
#     if cropped_bgr is None or cropped_mask is None:
#         return None

#     h, w = cropped_mask.shape[:2]
#     rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
#     face_pixels = (cropped_mask == 255)
#     rgba_image[face_pixels, 0:3] = cropped_bgr[face_pixels, :]
#     rgba_image[face_pixels, 3] = 255
#     return rgba_image

# def remove_black_spots_inpaint(rgba_img, black_thresh=30):
#     """
#     As a last resort, forcibly remove very dark pixels (R,G,B < black_thresh)
#     within the opaque region (alpha=255) by inpainting them with surrounding color.
#     This helps if there's still a stubborn black spot in the eye or mouth area
#     that morphological steps didn't remove.
#     """
#     # Separate color & alpha
#     bgr = rgba_img[:, :, 0:3].copy()
#     alpha = rgba_img[:, :, 3]
    
#     # Create an inpaint mask for very dark pixels in the opaque region
#     mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
#     # Find pixels that are all < black_thresh in B,G,R and alpha=255
#     dark_pixels = np.where(
#         (bgr[:, :, 0] < black_thresh) &
#         (bgr[:, :, 1] < black_thresh) &
#         (bgr[:, :, 2] < black_thresh) &
#         (alpha == 255)
#     )
#     mask[dark_pixels] = 255

#     # Inpaint those pixels
#     # cv2.INPAINT_TELEA usually produces smoother results for small areas
#     inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)

#     # Combine back with alpha
#     final_rgba = np.dstack((inpainted_bgr, alpha))
#     return final_rgba

# def main(image_path):
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Error: Cannot read {image_path}")
#         sys.exit(1)
    
#     # Ensure exactly one face
#     mp_face = mp.solutions.face_detection
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
#         results = fd.process(image_rgb)
#         if not results.detections or len(results.detections) != 1:
#             print("No face or multiple faces. Exiting.")
#             sys.exit(0)
#         else:
#             print("Exactly one face detected.")
    
#     print("Loading BiSeNet model...")
#     model = load_bisenet_model('79999_iter.pth', 19)
    
#     print("Generating face+hair mask with morphological cleanup...")
#     mask = face_hair_mask(image_bgr, model)
    
#     # Crop to just the face/hair region
#     cropped_bgr, cropped_mask = crop_face_hair(image_bgr, mask)
#     if cropped_bgr is None:
#         print("No valid face/hair region found.")
#         sys.exit(0)
    
#     # Create RGBA from the cropped region
#     cropped_rgba = create_cropped_rgba(cropped_bgr, cropped_mask)
#     if cropped_rgba is None:
#         print("Cropping failed.")
#         sys.exit(0)
    
#     # FINAL STEP: Inpaint any remaining black spots in the face region
#     # Increase black_thresh if the spot is not removed (but be careful with black hair).
#     print("Inpainting any leftover dark pixels inside the face region...")
#     final_rgba = remove_black_spots_inpaint(cropped_rgba, black_thresh=30)
    
#     out_name = "face_hair_segmented.png"
#     cv2.imwrite(out_name, final_rgba)
#     print(f"Saved to {out_name}")
    
#     cv2.imshow("Face+Hair Segmentation (Cropped + Inpaint)", final_rgba)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_detection1.py <image_path>")
#         sys.exit(1)
#     main(sys.argv[1])
