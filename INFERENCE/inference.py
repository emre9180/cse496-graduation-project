import json
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

# Function to enhance depth visualization
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

# Function to predict the real major axis length
def predict_real_major_length(major_axis_length, minor_axis_length, mean_depth):
    input_data = pd.DataFrame([[major_axis_length, minor_axis_length, mean_depth]], columns=['major_axis_length', 'minor_axis_length', 'mean_depth'])
    real_major_axis_length = model_major.predict(input_data)[0]
    return real_major_axis_length

# Function to predict the real minor axis length
def predict_real_minor_length(major_axis_length, minor_axis_length, mean_depth):
    input_data = pd.DataFrame([[major_axis_length, minor_axis_length, mean_depth]], columns=['major_axis_length', 'minor_axis_length', 'mean_depth'])
    real_minor_axis_length = model_minor.predict(input_data)[0]
    return real_minor_axis_length

# Load the regression models (assuming they are already trained as shown in your provided code)
model_major = LinearRegression()
model_major.coef_ = np.array([0.0245429,  0.01664848, 0.00261312])
model_major.intercept_ = -0.5840569503581086

model_minor = LinearRegression()
model_minor.coef_ = np.array([-0.00239179,  0.02397787,  0.0011271 ])
model_minor.intercept_ = 1.104498944597657

# Load the YOLO model
model = YOLO('best.pt')  # replace with the path to your model

# Load the image
image_path = 'color_image_17haz.jpg'
image = cv2.imread(image_path)

# Run the model on the image
confidence_threshold = 0.5
results = model.predict(image, conf=confidence_threshold)

# Load depth data from JSON file
with open('depth_data_17haz.json', 'r') as f:
    depth_data = np.array(json.load(f))

# Enhance depth visualization
depth_visual_colored = enhance_depth_visualization(depth_data)

# Check if results contain masks
if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()  # Get masks data and convert to numpy array on CPU

    # Create a colored mask for visualization
    color_mask = np.zeros_like(image)
    
    # Apply the masks to the color mask
    for mask in masks:
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask[mask_resized == 1] = (0, 255, 0)  # Set mask area to green

        # Find contours and fit ellipses
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 5:  # Minimum number of points required by fitEllipse
                ellipse = cv2.fitEllipse(contour)

                # Draw the original ellipse for reference
                cv2.ellipse(image, ellipse, (0, 0, 255), 2)  # Draw the ellipse in red on the original image

                # Calculate the major and minor axes in pixels
                major_axis_length = max(ellipse[1])
                minor_axis_length = min(ellipse[1])

                # Calculate the mean depth in the segmentation area
                mean_depth = compute_mean_depth(contour, depth_data)
                
                # Predict the real major and minor axis lengths
                real_major_axis_length = predict_real_major_length(major_axis_length, minor_axis_length, mean_depth)
                real_minor_axis_length = predict_real_minor_length(major_axis_length, minor_axis_length, mean_depth)
                
                print(f"Ellipse: Major axis length (pixels): {major_axis_length}, Minor axis length (pixels): {minor_axis_length}")
                print(f"Mean depth in segmentation area: {mean_depth:.2f}")
                print(f"Predicted real major axis length: {real_major_axis_length}")
                print(f"Predicted real minor axis length: {real_minor_axis_length}")

                # Draw the real major and minor axis lengths on the image
                text = f"Maj: {real_major_axis_length:.2f}, Min: {real_minor_axis_length:.2f}"
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                cv2.putText(image, text, (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Shrink the ellipse by 50% without changing its center
                new_axes = (ellipse[1][0] * 0.5, ellipse[1][1] * 0.5)  # Reduce axes by 50%
                small_ellipse = (ellipse[0], new_axes, ellipse[2])
                cv2.ellipse(image, small_ellipse, (255, 0, 0), 2)  # Draw the smaller ellipse in blue

    # Blend the original image with the color mask
    alpha = 0.5  # Transparency factor
    blended_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # Save the segmentation result
    cv2.imwrite('result.png', blended_image)
    cv2.imshow('result', blended_image)

    # Wait until ESC is pressed
    while True:
        if cv2.waitKey(1) == 27:  # 27 is the ESC key
            break
    cv2.destroyAllWindows()
    
    # Overlay segmentation on depth visualization
    overlay_depth = cv2.addWeighted(depth_visual_colored, 1 - alpha, color_mask, alpha, 0)

    # Save the depth overlay result
    cv2.imwrite('depth_result.png', overlay_depth)

    print("Segmentation completed. The results are saved as 'result.png' and 'depth_result.png'.")
else:
    print("No masks found in the results.")
