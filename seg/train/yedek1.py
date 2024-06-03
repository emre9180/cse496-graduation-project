import json
import re
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import yaml
from scipy.signal import correlate2d

# Load calibration files
def load_calibration_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

pose_calibration = load_calibration_file('pose_calibration.yaml')
rgb_calibration = load_calibration_file('rgb_calibration.yaml')
depth_calibration = load_calibration_file('depth_calibration.yaml')

# Extract calibration parameters
K_rgb = np.array(rgb_calibration['camera_matrix']['data']).reshape((3, 3))
D_rgb = np.array(rgb_calibration['distortion_coefficients']['data'])
K_depth_original = np.array(depth_calibration['camera_matrix']['data']).reshape((3, 3))
D_depth = np.array(depth_calibration['distortion_coefficients']['data'])

# Check keys in pose_calibration
if 'rotation_matrix' in pose_calibration and 'translation_matrix' in pose_calibration:
    R = np.array(pose_calibration['rotation_matrix']['data']).reshape((3, 3))
    T = np.array(pose_calibration['translation_matrix']['data']).reshape((3, 1))
else:
    raise KeyError("Keys 'rotation_matrix' and 'translation_matrix' not found in pose_calibration.yaml")

# Load COCO annotations
annotations_path = '_annotations.coco.json'
with open(annotations_path) as f:
    coco_data = json.load(f)

coco = COCO(annotations_path)

# Constants for depth frame dimensions
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424

# Constants for original RGB frame dimensions
COLOR_WIDTH_ORIG = 1920
COLOR_HEIGHT_ORIG = 1080

# Create directories to save the annotated images, segmented depth images, and segmented images with IDs
output_dir = 'annotated_images'
segmented_depth_dir = 'segmented_depth'
segmented_images_dir = 'segmented_images'
resized_segmented_images_dir = 'resized_segmented_images'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(segmented_depth_dir, exist_ok=True)
os.makedirs(segmented_images_dir, exist_ok=True)
os.makedirs(resized_segmented_images_dir, exist_ok=True)

def extract_number_after_mayis(file_name):
    match = re.search(r'mayis(\d+)', file_name)
    if match:
        return match.group(1)
    return None

# Function to fit an ellipse and calculate major and minor axis lengths
def fit_ellipse_and_get_axes(segmentation):
    points = np.array(segmentation).reshape((-1, 2)).astype(np.float32)
    if len(points) < 5:
        return None, None, None
    ellipse = cv2.fitEllipse(points)
    (center, axes, angle) = ellipse
    major_axis_length = max(axes)
    minor_axis_length = min(axes)
    return major_axis_length, minor_axis_length, ellipse

# Function to resize coordinates from one resolution to another
def resize_coordinates(segmentation, src_width, src_height, dst_width, dst_height):
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    resized_segmentation = [(int(x * scale_x), int(y * scale_y)) for x, y in segmentation]
    return resized_segmentation

# Function to compute the average depth in a segmented region, excluding outliers
def compute_average_depth(segmentation, depth_data):
    points = np.array(segmentation, dtype=np.int32)
    mask = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    depth_values = depth_data[mask == 1]
    if len(depth_values) == 0:
        return None

    # Remove outliers using IQR
    q1, q3 = np.percentile(depth_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_depth_values = depth_values[(depth_values >= lower_bound) & (depth_values <= upper_bound)]

    if len(filtered_depth_values) == 0:
        return None

    average_depth = np.mean(filtered_depth_values)
    return average_depth

# Function to find the best matching region in the depth data based on segmentation pattern
def find_best_matching_region(segmentation_depth_values, depth_data):
    # Normalize the segmentation depth values
    segmentation_depth_values = (segmentation_depth_values - np.mean(segmentation_depth_values)) / np.std(segmentation_depth_values)
    
    # Compute correlation between the segmentation depth pattern and the depth data
    correlation = correlate2d(depth_data, segmentation_depth_values, mode='valid')
    
    # Find the location with the highest correlation
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    return (x, y), correlation

# Process each image and its annotations
results = []
for image_info in coco_data['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']
    image_path = os.path.join('images', file_name)
    
    print(f"Processing image: {image_path}")  # Debug print
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        continue
    
    # Load image for annotation
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        continue
    
    # Resize the original image to match the depth image size
    resized_image = cv2.resize(image, (DEPTH_WIDTH, DEPTH_HEIGHT))

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    # Extract the number after "mayis" in the filename
    mayis_number = extract_number_after_mayis(file_name)
    if mayis_number is None:
        print(f"Warning: Unable to extract number from filename {file_name}. Skipping.")
        continue
    
    depth_file_name = f"depth_data_17mayis{mayis_number}.json"
    depth_file_path = os.path.join('depth_data', depth_file_name)
    if not os.path.exists(depth_file_path):
        print(f"Warning: Depth file {depth_file_path} does not exist. Skipping.")
        continue
    
    with open(depth_file_path) as depth_file:
        depth_data = np.array(json.load(depth_file)).astype(np.float32)

    depth_visual = (depth_data / np.max(depth_data) * 255).astype(np.uint8)
    depth_visual_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

    for ann in anns:
        segmentation = ann['segmentation'][0]
        major_axis_length, minor_axis_length, ellipse = fit_ellipse_and_get_axes(segmentation)
        
        if major_axis_length is not None and minor_axis_length is not None:
            # Resize segmentation coordinates to match depth data dimensions
            resized_segmentation = resize_coordinates(np.array(segmentation).reshape((-1, 2)), COLOR_WIDTH_ORIG, COLOR_HEIGHT_ORIG, DEPTH_WIDTH, DEPTH_HEIGHT)
            
            # Extract the depth values for the segmentation
            mask = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(resized_segmentation, dtype=np.int32)], 255)
            segmentation_depth_values = depth_data[mask == 255]

            if len(segmentation_depth_values) == 0:
                continue

            # Find the best matching region in the depth data
            top_left, correlation = find_best_matching_region(segmentation_depth_values, depth_data)
            bottom_right = (top_left[0] + mask.shape[1], top_left[1] + mask.shape[0])

            # Compute average depth for the best matching region
            best_match_segmentation = [(top_left[0], top_left[1]), (bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1]), (top_left[0], bottom_right[1])]
            average_depth = compute_average_depth(best_match_segmentation, depth_data)
            
            result = {
                'id': ann['id'],
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor_axis_length,
                'average_depth': average_depth
            }
            results.append(result)
            
            # Draw the ellipse and ID on the resized image
            cv2.ellipse(resized_image, ellipse, (0, 255, 0), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.putText(resized_image, str(ann['id']), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw the best matching region on the depth visual image
            cv2.rectangle(depth_visual_colored, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(depth_visual_colored, str(ann['id']), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw the resized segmentation on the resized original image
            cv2.polylines(resized_image, [np.array(resized_segmentation, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(resized_image, str(ann['id']), tuple(resized_segmentation[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw the resized segmentation on the depth visual image
            cv2.polylines(depth_visual_colored, [np.array(resized_segmentation, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(depth_visual_colored, str(ann['id']), tuple(resized_segmentation[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the resized annotated image with IDs
    output_path = os.path.join(segmented_images_dir, file_name)
    cv2.imwrite(output_path, resized_image)
    
    # Save and visualize the depth visual image with segmentation
    depth_visual_path = os.path.join(segmented_depth_dir, f"depth_{file_name}")
    cv2.imwrite(depth_visual_path, depth_visual_colored)
    cv2.imshow("Depth Image", depth_visual_colored)
    cv2.waitKey(30)

    # Save the resized image with the resized segmentation
    resized_segmented_image_path = os.path.join(resized_segmented_images_dir, f"resized_{file_name}")
    cv2.imwrite(resized_segmented_image_path, resized_image)

# Save the results to a new JSON file
with open('ellipse_results_with_depth.json', 'w') as f:
    json.dump(results, f, indent=4)

# Visualize an example annotated image
if len(coco_data['images']) > 0:
    example_image_path = os.path.join(segmented_images_dir, coco_data['images'][0]['file_name'])
    example_image = cv2.imread(example_image_path)
    if example_image is not None:
        plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print(f"Warning: Unable to read example image {example_image_path}.")

cv2.destroyAllWindows()
