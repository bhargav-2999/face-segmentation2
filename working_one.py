# WORKING CODE EXCEPT BLACK SPOT ON THE LEFT EYE


import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import mediapipe as mp

# If needed:
# sys.path.append("./face-parsing.PyTorch")
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
    Fill any holes fully enclosed by the mask. This removes
    black spots inside the face region IF they're fully enclosed.
    """
    inv_mask = cv2.bitwise_not(mask_255)  # invert => background becomes white
    largest_bg = keep_largest_component(inv_mask)
    filled_mask = cv2.bitwise_not(largest_bg)
    return filled_mask

def unify_hair_if_mislabeled(parsing_map):
    """
    Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
    If hair is mislabeled as neck(13) or clothes(14), unify them if adjacent.
    """
    hair_mask = ((parsing_map == 15) | (parsing_map == 17))
    neck_mask = (parsing_map == 13)
    clothes_mask = (parsing_map == 14)

    # Slight dilation to catch adjacent neck/clothes mislabeled as hair
    hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
    adjacent_neck = neck_mask & (hair_dilated > 0)
    adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
    corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
    return corrected_hair_mask

def face_hair_mask(image_bgr, model):
    """
    Generate a mask that includes face (1..12) + hair (15,17).
    Exclude neck(13)/clothes(14), except if mislabeled hair is adjacent.
    Then fill small holes, and do morphological ops for a clean boundary.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image_rgb.shape
    
    # 1. Resize
    input_size = 512
    image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
    # 2. Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(image_resized).unsqueeze(0)
    
    # 3. Forward pass
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
    
    # 4. Keep face classes 1..12 plus hair classes (15,17)
    face_classes = list(range(1, 13))  # 1..12
    hair_classes = [15, 17]
    keep_classes = face_classes + hair_classes

    mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
    # 5. Unify hair if mislabeled as neck/clothes
    corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)
    
    # Combine face region with corrected hair region
    face_mask = (mask_512 > 0)
    combined_mask = face_mask | corrected_hair_mask
    
    # 6. Morphological opening to remove small specks
    kernel_open = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 7. Morphological closing to smooth edges
    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 8. Keep largest connected component
    largest = keep_largest_component(closed)
    
    # 9. Fill any holes inside the face region
    filled = fill_mask_holes(largest)
    
    # 10. Additional morphological close with bigger kernel to remove leftover spots
    kernel_close2 = np.ones((5,5), np.uint8)
    double_closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close2, iterations=2)
    final_mask = keep_largest_component(double_closed)
    
    # 11. Resize back
    mask_original = cv2.resize(final_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_original

def create_rgba_image(original_bgr, mask):
    """
    Create an RGBA image where the face/hair region is opaque and the rest is transparent.
    """
    orig_h, orig_w, _ = original_bgr.shape
    rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    face_pixels = (mask == 255)
    rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
    rgba_image[face_pixels, 3] = 255
    return rgba_image

def crop_face_hair(image_bgr, mask):
    """
    Crop the image and mask to the bounding box of the non-zero region
    so the output is a 'floating head' with minimal background.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)

    # A small margin so we don't chop off hair
    margin = 10
    max_y, max_x = image_bgr.shape[:2]
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, max_x)
    y2 = min(y + h + margin, max_y)

    cropped_bgr = image_bgr[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    return cropped_bgr, cropped_mask

def create_cropped_rgba(cropped_bgr, cropped_mask):
    """
    Create an RGBA image from the cropped region, with transparency outside the face/hair.
    """
    if cropped_bgr is None or cropped_mask is None:
        return None

    h, w = cropped_mask.shape[:2]
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    face_pixels = (cropped_mask == 255)
    rgba_image[face_pixels, 0:3] = cropped_bgr[face_pixels, :]
    rgba_image[face_pixels, 3] = 255
    return rgba_image

def remove_small_alpha_holes(rgba, max_hole_area=500):
    """
    Find small alpha=0 connected components fully inside the face bounding box,
    and force them to become alpha=255. This merges leftover 'holes' (like the
    dark spot near the left eye) into the face region.
    """
    alpha = rgba[:, :, 3]
    
    # 1) Get bounding rect of the main face (alpha=255)
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return rgba
    x, y, w, h = cv2.boundingRect(coords)
    
    # 2) Identify connected components in alpha=0
    alpha_inv = (alpha == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_inv, connectivity=8)
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
        # Check if this hole is entirely inside the face bounding rect
        inside_face_box = (left >= x) and (top >= y) and ((left + width) <= x + w) and ((top + height) <= y + h)
        
        # If it's small enough, unify it with alpha=255
        if inside_face_box and (area <= max_hole_area):
            alpha[labels == label_id] = 255
    
    rgba[:, :, 3] = alpha
    return rgba

def inpaint_alpha(rgba):
    """
    After we've merged small alpha holes, we may have no color info for those pixels.
    We'll inpaint them from the surrounding face region so they blend in nicely.
    """
    alpha = rgba[:, :, 3]
    bgr = rgba[:, :, :3].copy()
    
    # Our inpaint mask: newly changed alpha=255 pixels that have no color info
    # We'll say any pixel that was alpha=0 but is now alpha=255 is black in bgr
    # or we can detect bgr=(0,0,0).
    mask = np.zeros(alpha.shape, dtype=np.uint8)
    
    # Mark inpaint region: alpha=255 but color is near black
    black_thresh = 20
    dark_pixels = np.where(
        (bgr[:, :, 0] < black_thresh) &
        (bgr[:, :, 1] < black_thresh) &
        (bgr[:, :, 2] < black_thresh) &
        (alpha == 255)
    )
    mask[dark_pixels] = 255
    
    # Inpaint
    inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    final_rgba = np.dstack((inpainted_bgr, alpha))
    return final_rgba

def main(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Cannot read {image_path}")
        sys.exit(1)
    
    # Ensure exactly one face
    mp_face = mp.solutions.face_detection
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(image_rgb)
        if not results.detections or len(results.detections) != 1:
            print("No face or multiple faces. Exiting.")
            sys.exit(0)
        else:
            print("Exactly one face detected.")
    
    print("Loading BiSeNet model...")
    model = load_bisenet_model('79999_iter.pth', 19)
    
    print("Generating face+hair mask with morphological cleanup...")
    mask = face_hair_mask(image_bgr, model)
    
    # Crop to just the face/hair region
    cropped_bgr, cropped_mask = crop_face_hair(image_bgr, mask)
    if cropped_bgr is None:
        print("No valid face/hair region found.")
        sys.exit(0)
    
    # Create RGBA from the cropped region
    cropped_rgba = create_cropped_rgba(cropped_bgr, cropped_mask)
    if cropped_rgba is None:
        print("Cropping failed.")
        sys.exit(0)
    
    # 1) Merge small alpha holes into the face
    print("Merging small holes in alpha channel (removing leftover dark spots)...")
    alpha_fixed = remove_small_alpha_holes(cropped_rgba, max_hole_area=500)
    
    # 2) Inpaint the newly filled areas so they blend with surrounding color
    print("Inpainting newly filled areas...")
    final_rgba = inpaint_alpha(alpha_fixed)
    
    out_name = "face_hair_segmented.png"
    cv2.imwrite(out_name, final_rgba)
    print(f"Saved to {out_name}")
    
    cv2.imshow("Face+Hair Segmentation (Cropped, Holes Removed)", final_rgba)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_detection1.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])





#WORKING ONE BUT HAIR IS NOT PROPER


import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import mediapipe as mp

# If needed:
# sys.path.append("./face-parsing.PyTorch")
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
    Fill any holes fully enclosed by the mask. This removes
    black spots inside the face region IF they're fully enclosed.
    """
    inv_mask = cv2.bitwise_not(mask_255)  # invert => background becomes white
    largest_bg = keep_largest_component(inv_mask)
    filled_mask = cv2.bitwise_not(largest_bg)
    return filled_mask

def unify_hair_if_mislabeled(parsing_map):
    """
    Some models label hair as 15, others as 17. We'll unify them both as 'hair'.
    If hair is mislabeled as neck(13) or clothes(14), unify them if adjacent.
    """
    hair_mask = ((parsing_map == 15) | (parsing_map == 17))
    neck_mask = (parsing_map == 13)
    clothes_mask = (parsing_map == 14)

    # Slight dilation to catch adjacent neck/clothes mislabeled as hair
    hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
    adjacent_neck = neck_mask & (hair_dilated > 0)
    adjacent_clothes = clothes_mask & (hair_dilated > 0)
    
    corrected_hair_mask = (hair_mask | adjacent_neck | adjacent_clothes)
    return corrected_hair_mask

def face_hair_mask(image_bgr, model):
    """
    Generate a mask that includes face (1..12) + hair (15,17).
    Exclude neck(13)/clothes(14), except if mislabeled hair is adjacent.
    Then fill small holes, and do morphological ops for a clean boundary.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image_rgb.shape
    
    # 1. Resize
    input_size = 512
    image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
    # 2. Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(image_resized).unsqueeze(0)
    
    # 3. Forward pass
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
    
    # 4. Keep face classes 1..12 plus hair classes (15,17)
    face_classes = list(range(1, 13))  # 1..12
    hair_classes = [15, 17]
    keep_classes = face_classes + hair_classes

    mask_512 = np.isin(parsing_map, keep_classes).astype(np.uint8)*255
    
    # 5. Unify hair if mislabeled as neck/clothes
    corrected_hair_mask = unify_hair_if_mislabeled(parsing_map)
    
    # Combine face region with corrected hair region
    face_mask = (mask_512 > 0)
    combined_mask = face_mask | corrected_hair_mask
    
    # 6. Morphological opening to remove small specks
    kernel_open = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 7. Morphological closing to smooth edges
    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(opened*255, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 8. Keep largest connected component
    largest = keep_largest_component(closed)
    
    # 9. Fill any holes inside the face region
    filled = fill_mask_holes(largest)
    
    # 10. Additional morphological closing with a bigger kernel to remove leftover spots
    # Updated: using a larger kernel and more iterations.
    kernel_close2 = np.ones((7,7), np.uint8)
    double_closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close2, iterations=3)
    final_mask = keep_largest_component(double_closed)
    
    # 11. Resize back
    mask_original = cv2.resize(final_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_original

def create_rgba_image(original_bgr, mask):
    """
    Create an RGBA image where the face/hair region is opaque and the rest is transparent.
    """
    orig_h, orig_w, _ = original_bgr.shape
    rgba_image = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    face_pixels = (mask == 255)
    rgba_image[face_pixels, 0:3] = original_bgr[face_pixels, :]
    rgba_image[face_pixels, 3] = 255
    return rgba_image

def crop_face_hair(image_bgr, mask):
    """
    Crop the image and mask to the bounding box of the non-zero region
    so the output is a 'floating head' with minimal background.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)

    # A small margin so we don't chop off hair
    margin = 10
    max_y, max_x = image_bgr.shape[:2]
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, max_x)
    y2 = min(y + h + margin, max_y)

    cropped_bgr = image_bgr[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    return cropped_bgr, cropped_mask

def create_cropped_rgba(cropped_bgr, cropped_mask):
    """
    Create an RGBA image from the cropped region, with transparency outside the face/hair.
    """
    if cropped_bgr is None or cropped_mask is None:
        return None

    h, w = cropped_mask.shape[:2]
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    face_pixels = (cropped_mask == 255)
    rgba_image[face_pixels, 0:3] = cropped_bgr[face_pixels, :]
    rgba_image[face_pixels, 3] = 255
    return rgba_image

def remove_small_alpha_holes(rgba, max_hole_area=800):
    """
    Find small alpha=0 connected components fully inside the face bounding box,
    and force them to become alpha=255. This merges leftover 'holes' (like the
    dark spot near the left eye) into the face region.
    """
    alpha = rgba[:, :, 3]
    
    # 1) Get bounding rect of the main face (alpha=255)
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return rgba
    x, y, w, h = cv2.boundingRect(coords)
    
    # 2) Identify connected components in alpha=0
    alpha_inv = (alpha == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_inv, connectivity=8)
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
        # Check if this hole is entirely inside the face bounding rect
        inside_face_box = (left >= x) and (top >= y) and ((left + width) <= x + w) and ((top + height) <= y + h)
        
        # If it's small enough, unify it with alpha=255
        if inside_face_box and (area <= max_hole_area):
            alpha[labels == label_id] = 255
    
    rgba[:, :, 3] = alpha
    return rgba

def inpaint_alpha(rgba):
    """
    After we've merged small alpha holes, we may have no color info for those pixels.
    We'll inpaint them from the surrounding face region so they blend in nicely.
    """
    alpha = rgba[:, :, 3]
    bgr = rgba[:, :, :3].copy()
    
    # Our inpaint mask: newly changed alpha=255 pixels that have no color info.
    mask = np.zeros(alpha.shape, dtype=np.uint8)
    
    # Mark inpaint region: alpha=255 but color is near black.
    # Increased black_thresh from 20 to 80 to catch dark gray values.
    black_thresh = 80  
    dark_pixels = np.where(
        (bgr[:, :, 0] < black_thresh) &
        (bgr[:, :, 1] < black_thresh) &
        (bgr[:, :, 2] < black_thresh) &
        (alpha == 255)
    )
    mask[dark_pixels] = 255
    
    # Inpaint
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

def main(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Cannot read {image_path}")
        sys.exit(1)
    
    # Ensure exactly one face
    mp_face = mp.solutions.face_detection
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(image_rgb)
        if not results.detections or len(results.detections) != 1:
            print("No face or multiple faces. Exiting.")
            sys.exit(0)
        else:
            print("Exactly one face detected.")
    
    print("Loading BiSeNet model...")
    model = load_bisenet_model('79999_iter.pth', 19)
    
    print("Generating face+hair mask with morphological cleanup...")
    mask = face_hair_mask(image_bgr, model)
    
    # Crop to just the face/hair region
    cropped_bgr, cropped_mask = crop_face_hair(image_bgr, mask)
    if cropped_bgr is None:
        print("No valid face/hair region found.")
        sys.exit(0)
    
    # Create RGBA from the cropped region
    cropped_rgba = create_cropped_rgba(cropped_bgr, cropped_mask)
    if cropped_rgba is None:
        print("Cropping failed.")
        sys.exit(0)
    
    # 1) Merge small alpha holes into the face
    print("Merging small holes in alpha channel (removing leftover dark spots)...")
    alpha_fixed = remove_small_alpha_holes(cropped_rgba)  # default max_hole_area=800
    
    # 2) Inpaint the newly filled areas so they blend with surrounding color
    print("Inpainting newly filled areas...")
    inpainted = inpaint_alpha(alpha_fixed)
    
    # 3) Feather the alpha channel to smooth the edge
    final_rgba = feather_alpha(inpainted, radius=2)
    
    out_name = "face_hair_segmented.png"
    cv2.imwrite(out_name, final_rgba)
    print(f"Saved to {out_name}")
    
    cv2.imshow("Face+Hair Segmentation (Cropped, Holes Removed)", final_rgba)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_detection1.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])