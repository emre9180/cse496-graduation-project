import json
import re
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

def enhance_depth_visualization(depth_data, clip_min=None, clip_max=None):
    if clip_min is None:
        clip_min = np.min(depth_data)
    if clip_max is None:
        clip_max = np.max(depth_data)
    
    # Clip the depth data to the specified range
    depth_data_clipped = np.clip(depth_data, clip_min, clip_max)
    
    # Normalize the depth data to [0, 255]
    depth_visual = ((depth_data_clipped - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)
    depth_visual_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    
    return depth_visual_colored

def extract_number_after_mayis(file_name):
    match = re.search(r'nisan(\d+)', file_name)
    if match:
        return match.group(1)
    return None

# Load COCO annotations
annotations_path = '_annotations.coco.json'
with open(annotations_path) as f:
    coco_data = json.load(f)

coco = COCO(annotations_path)

depth_annotations_path = '_depth_annotations.coco.json'
with open(depth_annotations_path) as f:
    depth_coco_data = json.load(f)

depth_coco = COCO(depth_annotations_path)

# Create directories to save the annotated images
output_dir = 'annotated_images'
segmented_depth_dir = 'segmented_depth'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(segmented_depth_dir, exist_ok=True)

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

# Function to compute the mean depth value in a segmented region
def compute_mean_depth(segmentation, depth_data):
    mask = np.zeros(depth_data.shape, dtype=np.uint8)
    points = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    depth_values = depth_data[mask == 1]
    if len(depth_values) == 0:
        return None
    mean_depth = np.mean(depth_values)
    return mean_depth

# Process each image and its annotations
results = []
for image_info in coco_data['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']
    image_path = os.path.join('images', file_name)
    
    print(f"Processing image: {image_path}")  # Debug print
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        continue
    
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    # Extract the number after "mayis" in the filename
    mayis_number = extract_number_after_mayis(file_name)
    print(file_name)
    print(mayis_number)
    if mayis_number is None:
        print(f"Warning: Unable to extract number from filename {file_name}. Skipping.")
        continue
    
    depth_file_name = f"depth_data_30nisan{mayis_number}.json"
    depth_file_path = os.path.join('depth_data', depth_file_name)
    if not os.path.exists(depth_file_path):
        print(f"Warnasding: Depth file {depth_file_path} does not exist. Skipping.")
        continue
    
    with open(depth_file_path) as depth_file:
        depth_data = np.array(json.load(depth_file)).astype(np.float32)

    depth_visual = (depth_data / np.max(depth_data) * 255).astype(np.uint8)
    depth_visual_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

    for ann in anns:
        segmentation = ann['segmentation'][0]
        major_axis_length, minor_axis_length, ellipse = fit_ellipse_and_get_axes(segmentation)
        
        if major_axis_length is not None and minor_axis_length is not None:
            mean_depth = compute_mean_depth(segmentation, depth_data)
            result = {
                'id': ann['id'],
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor_axis_length,
                'mean_depth': 0,
                'real_major_axis_length': 0,
                'real_minor_axis_length': 0
            }
            results.append(result)
            
            # Draw the ellipse and ID on the image
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.putText(image, str(ann['id']), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the annotated image
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, image)

# Process depth annotations and visualize
depth_results = []
for image_info in coco_data['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']
    
    mayis_id = extract_number_after_mayis(file_name)
    print(file_name)
    depth_file_name = f"depth_data_30nisan{mayis_id}.json"
    depth_file_path = os.path.join('depth_data', depth_file_name)
    if not os.path.exists(depth_file_path):
        print(f"Warning: Depth file {depth_file_path} does not exist. Skipping.")
        continue
    
    with open(depth_file_path) as depth_file:
        depth_data = np.array(json.load(depth_file)).astype(np.float32)
    
    depth_visual_colored = enhance_depth_visualization(depth_data, clip_min=np.percentile(depth_data, 5), clip_max=np.percentile(depth_data, 95))


    ann_ids = depth_coco.getAnnIds(imgIds=image_id)
    anns = depth_coco.loadAnns(ann_ids)

    for ann in anns:
        segmentation = ann['segmentation'][0]
        # major_axis_length, minor_axis_length, ellipse = fit_ellipse_and_get_axes(segmentation)
        
        if major_axis_length is not None and minor_axis_length is not None:
            mean_depth = compute_mean_depth(segmentation, depth_data)
            result = {
                'id': ann['id'],
                # 'major_axis_length': major_axis_length,
                # 'minor_axis_length': minor_axis_length,
                'mean_depth': float(mean_depth) if mean_depth is not None else None  # Convert to standard Python float
            }
            depth_results.append(result)

            # Draw the segmentation on the depth visual image
            points = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
            # cv2.polylines(depth_visual_colored, [points], isClosed=True, color=(0, 255, 0), thickness=1)
            # cv2.putText(depth_visual_colored, str(ann['id']), tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save the depth visual image with segmentation
    depth_visual_path = os.path.join(segmented_depth_dir, f"depth_{file_name}")
    cv2.imwrite(depth_visual_path, depth_visual_colored)

# Save the depth results to a new JSON file
with open('ellipse_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save the depth results to a new JSON file
with open('depth_results.json', 'w') as f:
    json.dump(depth_results, f, indent=4)

# Visualize an example depth annotated image
if len(depth_coco_data['images']) > 0:
    example_depth_image_path = os.path.join(segmented_depth_dir, f"depth_{depth_coco_data['images'][0]['file_name']}")
    example_depth_image = cv2.imread(example_depth_image_path)
    if example_depth_image is not None:
        plt.imshow(cv2.cvtColor(example_depth_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print(f"Warning: Unable to read example depth image {example_depth_image_path}.")
