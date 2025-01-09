import cv2
import numpy as np
import os   
from moviepy.editor import VideoFileClip

# Directories
image_dir = "im"  # Folder containing your fabric images
output_video_path = "fabric_rolling.avi"
final_output_path = "fabric_rolling_final.mp4"

# Video settings
frame_rate = 30  # frames per second
rolling_speed = 100  # pixels per second (adjust for speed)
total_video_duration = 10  # seconds (adjust as needed)

# Desired resolution (window size)
target_width = 1920
target_height = 1080

# Collect image paths (ensure they are in the correct order)
image_paths = [
    os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir)) 
    if img.endswith(('.png', '.jpg'))
]

# Ensure there are exactly 3 images
if len(image_paths) != 3:
    raise ValueError("Please provide exactly 3 images for the fabric rolling effect.")

def resize_image(image, target_width):
    """
    Resize the image to match the target width while maintaining aspect ratio.
    Rotate the image by 90 degrees clockwise before resizing.
    """
    # Rotate the image by 90 degrees clockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Resize the rotated image to match the target width
    original_height, original_width = rotated_image.shape[:2]
    scale = target_width / original_width
    new_height = int(original_height * scale)
    resized_image = cv2.resize(rotated_image, (target_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def stack_images_vertically(images):
    """
    Stack images vertically into one tall image.
    """
    return np.vstack(images)

def create_fabric_rolling_video(image_paths, output_path):
    """
    Create a video with a rolling fabric effect using the provided images.
    """
    # Load and resize images to match the target width
    images = [resize_image(cv2.imread(path), target_width) for path in image_paths]

    # Stack images vertically
    stacked_image = stack_images_vertically(images)
    stacked_height, stacked_width, _ = stacked_image.shape

    # Calculate total frames based on video duration
    total_frames = int(frame_rate * total_video_duration)

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (target_width, target_height))

    # Simulate the rolling effect
    for frame in range(total_frames):
        # Calculate the vertical position of the window
        y_offset = int((frame * rolling_speed) % stacked_height)

        # Extract the window from the stacked image
        if y_offset + target_height <= stacked_height:
            window = stacked_image[y_offset:y_offset + target_height, :]
        else:
            # Handle wrapping around to the top of the stacked image
            remaining_height = stacked_height - y_offset
            window_top = stacked_image[y_offset:, :]
            window_bottom = stacked_image[:target_height - remaining_height, :]
            window = np.vstack((window_top, window_bottom))

        # Write the frame to the video
        out.write(window)

    # Release the VideoWriter
    out.release()
    print(f"Fabric rolling video saved as {output_path}")

# Create the fabric rolling video
create_fabric_rolling_video(image_paths, output_video_path)

# Optionally, add audio or convert to final format
video_clip = VideoFileClip(output_video_path)
final_video = video_clip.set_audio(None)  # Add audio here if needed
final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

print(f"Final video saved as {final_output_path}")