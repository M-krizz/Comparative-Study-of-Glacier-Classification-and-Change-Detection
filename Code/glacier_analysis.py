
import cv2
import numpy as np
import os
from PIL import Image

def extract_frames_from_gif(gif_path, output_dir):
    """Extract frames from a GIF file and save them as individual images."""
    gif = Image.open(gif_path)
    frame_number = 0
    while True:
        frame_path = os.path.join(output_dir, f"frame_{frame_number:03d}.png")
        gif.save(frame_path)
        try:
            gif.seek(gif.tell() + 1)
            frame_number += 1
        except EOFError:
            break
    print(f"Extracted {frame_number} frames.")
    return frame_number

def analyze_glacier_changes(frame_dir):
    """Analyze changes between consecutive frames."""
    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])
    differences = []

    for i in range(1, len(frames)):
        # Load consecutive frames
        img1 = cv2.imread(frames[i - 1], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)

        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)

        # Threshold to highlight significant changes
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate change area
        change_area = np.sum(diff_thresh > 0)
        differences.append({
            "frame_pair": (frames[i - 1], frames[i]),
            "change_area": change_area,
            "difference_image": diff_thresh
        })

        # Save or display the difference image
        diff_image_path = os.path.join(frame_dir, f"diff_{i - 1}_{i}.png")
        cv2.imwrite(diff_image_path, diff_thresh)

    return differences

# Example usage
if __name__ == "__main__":
    gif_path = "d:\GLOF\Satelight image\Imja_Tsho_lake_volume_change.gif"  # Path to the GIF file
    output_dir = "frames"     # Directory to save frames
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames
    extract_frames_from_gif(gif_path, output_dir)

    # Analyze glacier changes
    differences = analyze_glacier_changes(output_dir)

    # Display results
    for diff in differences:
        print(f"Frames: {diff['frame_pair']}, Change Area: {diff['change_area']}")
