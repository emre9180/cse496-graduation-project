import json
import cv2
import numpy as np
import os
import yaml

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
K_depth = np.array(depth_calibration['camera_matrix']['data']).reshape((3, 3))
D_depth = np.array(depth_calibration['distortion_coefficients']['data'])
R = np.array(pose_calibration['rotation_matrix']['data']).reshape((3, 3))
T = np.array(pose_calibration['translation_vector']['data']).reshape((3, 1))

# Constants for frame dimensions
COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424

def load_depth_data(filepath):
    with open(filepath, 'r') as file:
        depth_data = json.load(file)
    depth_array = np.array(depth_data, dtype=np.uint16)
    return depth_array

def undistort_image(image, K, D):
    return cv2.undistort(image, K, D)

def project_depth_to_color(depth_data, R, T, K_depth, K_rgb):
    points_3d = []
    for i in range(DEPTH_HEIGHT):
        for j in range(DEPTH_WIDTH):
            depth = depth_data[i, j] / 1000.0  # Convert from mm to meters
            if depth > 0:
                z = depth
                x = (j - K_depth[0, 2]) * z / K_depth[0, 0]
                y = (i - K_depth[1, 2]) * z / K_depth[1, 1]
                points_3d.append([x, y, z])

    points_3d = np.array(points_3d)
    points_3d_transformed = (R @ points_3d.T + T).T
    points_2d = K_rgb @ points_3d_transformed.T
    points_2d = points_2d[:2] / points_2d[2]

    return points_2d.T.reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 2)

def align_depth_to_color(color_image, depth_data, projected_points):
    aligned_depth_image = np.zeros((COLOR_HEIGHT, COLOR_WIDTH), dtype=np.uint16)
    for i in range(DEPTH_HEIGHT):
        for j in range(DEPTH_WIDTH):
            x, y = projected_points[i, j]
            if 0 <= int(x) < COLOR_WIDTH and 0 <= int(y) < COLOR_HEIGHT:
                aligned_depth_image[int(y), int(x)] = depth_data[i, j]
    return aligned_depth_image

def main():
    color_dir = 'images/'
    depth_dir = 'depth/'
    aligned_dir = 'aligned_data/'
    os.makedirs(aligned_dir, exist_ok=True)

    frame_count = 0
    while True:
        color_path = os.path.join(color_dir, f'color_image_17mayis{frame_count}.jpg')
        depth_path = os.path.join(depth_dir, f'depth_data_17mayis{frame_count}.json')

        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"Completed processing {frame_count} frames.")
            break

        # Load color image
        color_image = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        if color_image is None:
            print(f"Failed to load color image: {color_path}")
            break

        # Load depth data
        depth_data = load_depth_data(depth_path)

        # Undistort color and depth images
        undistorted_color = undistort_image(color_image, K_rgb, D_rgb)
        undistorted_depth = undistort_image(depth_data, K_depth, D_depth)

        # Project depth to color space
        projected_points = project_depth_to_color(undistorted_depth, R, T, K_depth, K_rgb)

        # Align depth to color image
        aligned_depth_image = align_depth_to_color(undistorted_color, undistorted_depth, projected_points)

        # Save aligned images
        color_save_path = os.path.join(aligned_dir, f'color_image_17mayis{frame_count}.png')
        depth_save_path = os.path.join(aligned_dir, f'aligned_depth_image_17mayis{frame_count}.png')
        cv2.imwrite(color_save_path, undistorted_color)
        cv2.imwrite(depth_save_path, aligned_depth_image)

        frame_count += 1

if __name__ == "__main__":
    main()
