import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

# Directories
image_path = "im/red-fabric-cloth-polyester-texture-background_35977-2520 (1).jpg"  # Path to the single fabric image
output_video_path = "fabric_rolling.avi"
final_output_path = "fabric_rolling_final.mp4"

# Video settings
frame_rate = 30  # frames per second
total_video_duration = 30  # seconds (adjust as needed)

# Desired resolution (window size)
target_width = 1920
target_height = 1080

# Load and preprocess the single image
def resize_image(image, target_width):
    """
    Resize the image to match the target width while maintaining aspect ratio.
    """
    # Resize the image to match the target width
    original_height, original_width = image.shape[:2]
    scale = target_width / original_width
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def create_fabric_rolling_video(image_path, output_path):
    """
    Create a video with a rolling fabric effect using a single image.
    """
    # Load and resize the single image to match the target width
    image = resize_image(cv2.imread(image_path), target_width)
    image_height, image_width, _ = image.shape

    # Calculate the step size based on height/10
    step_size = target_height // 10

    # Calculate total frames based on video duration
    total_frames = int(frame_rate * total_video_duration)

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (target_width, target_height))

    # Simulate the rolling effect
    for frame in range(total_frames):
        # Calculate the vertical position of the window
        y_offset = (frame * step_size) % image_height

        # Extract the window from the image
        if y_offset + target_height <= image_height:
            window = image[y_offset:y_offset + target_height, :]
        else:
            # Handle wrapping around to the top of the image
            remaining_height = image_height - y_offset
            window_top = image[y_offset:, :]
            window_bottom = image[:target_height - remaining_height, :]
            window = np.vstack((window_top, window_bottom))

        # Write the frame to the video
        out.write(window)

    # Release the VideoWriter
    out.release()
    print(f"Fabric rolling video saved as {output_path}")

# Create the fabric rolling video
create_fabric_rolling_video(image_path, output_video_path)

# Optionally, add audio or convert to final format
video_clip = VideoFileClip(output_video_path)
final_video = video_clip.set_audio(None)  # Add audio here if needed
final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

print(f"Final video saved as {final_output_path}")
