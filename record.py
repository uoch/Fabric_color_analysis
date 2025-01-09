import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class FabricRollingVideo:
    def __init__(self, image_path, output_video_path, final_output_path, frame_rate=30, total_duration=50, target_width=1920, target_height=1080):
        """
        Initialize the FabricRollingVideo class with paths and settings.
        """
        self.image_path = image_path
        self.output_video_path = output_video_path
        self.final_output_path = final_output_path
        self.frame_rate = frame_rate
        self.total_duration = total_duration
        self.target_width = target_width
        self.target_height = target_height

    def resize_image(self, image):
        """
        Resize the image to match the target width and ensure the height is a multiple of target_height.
        """
        original_height, original_width = image.shape[:2]
        scale = self.target_width / original_width
        new_height = int(original_height * scale)
        resized_image = cv2.resize(image, (self.target_width, new_height), interpolation=cv2.INTER_AREA)
        multiple_height = ((resized_image.shape[0] // self.target_height) + 1) * self.target_height
        padded_image = cv2.copyMakeBorder(resized_image, 0, multiple_height - resized_image.shape[0], 0, 0, cv2.BORDER_WRAP)
        return padded_image

    def create_video(self):
        """
        Create a video with a rolling fabric effect using a single image.
        """
        # Load and resize the single image
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        image = self.resize_image(image)
        image_height, image_width, _ = image.shape

        # Calculate the step size based on total frames and image height
        total_frames = int(self.frame_rate * self.total_duration)
        step_size = image_height / total_frames  # Smooth scrolling step size

        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.frame_rate, (self.target_width, self.target_height))

        if not out.isOpened():
            raise IOError(f"Could not open video writer for {self.output_video_path}")

        # Simulate the rolling effect
        for frame in range(total_frames):
            # Calculate the vertical position of the window
            y_offset = int((frame * step_size) % image_height)

            # Extract the window from the image
            window = image[y_offset:y_offset + self.target_height, :]

            # Write the frame to the video
            out.write(window)

        # Release the VideoWriter
        out.release()
        print(f"Fabric rolling video saved as {self.output_video_path}")

    def analyze_and_display_video(self):
        """
        Display the video at the top and the plots below it in real-time.
        """
        # Open the video
        cap = cv2.VideoCapture(self.output_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video at {self.output_video_path}")

        # Initialize matplotlib figures
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two rows for plots
        fig.suptitle("Color Uniformity Analysis")

        # Initialize data for plotting
        frame_numbers = []  # X-axis: Frame numbers
        left_sums = []  # Y-axis: Sum of pixel values for the left region
        middle_sums = []  # Y-axis: Sum of pixel values for the middle region
        right_sums = []  # Y-axis: Sum of pixel values for the right region
        global_red_intensities = []  # Y-axis: Global red channel intensity
        global_green_intensities = []  # Y-axis: Global green channel intensity
        global_blue_intensities = []  # Y-axis: Global blue channel intensity

        def calculate_sum_of_pixels(region):
            """Calculate the sum of pixel values for a region."""
            return np.sum(region, axis=(0, 1))

        frame_count = 0  # Track frame number for x-axis

        # Create a window to display video and plots
        cv2.namedWindow("Video and Plots", cv2.WINDOW_NORMAL)

        while True:
            # Read the current frame
            ret, frame = cap.read()
            if not ret:
                break

            # Split the frame into three regions (left, middle, right)
            region_width = frame.shape[1] // 3
            regions = {
                "left": frame[:, :region_width],
                "middle": frame[:, region_width:2 * region_width],
                "right": frame[:, 2 * region_width:]
            }

            # Calculate the sum of pixel values for each region
            sums = {region_name: calculate_sum_of_pixels(region) for region_name, region in regions.items()}

            # Store the sum of pixel values for each region
            left_sums.append(np.sum(sums["left"]))  # Sum of pixel values for the left region
            middle_sums.append(np.sum(sums["middle"]))  # Sum of pixel values for the middle region
            right_sums.append(np.sum(sums["right"]))  # Sum of pixel values for the right region
            frame_numbers.append(frame_count)  # X-axis: Frame number

            # Update the Regional Color Gradation plot
            ax1.clear()
            ax1.plot(frame_numbers, left_sums, color="cyan", label="Left Region")  # Cyan for Left
            ax1.plot(frame_numbers, middle_sums, color="magenta", label="Middle Region")  # Magenta for Middle
            ax1.plot(frame_numbers, right_sums, color="yellow", label="Right Region")  # Yellow for Right
            ax1.set_title("Regional Color Gradation Over Time (Sum of Pixels)")
            ax1.set_xlabel("Frame Number")
            ax1.set_ylabel("Sum of Pixel Values")
            ax1.legend()

            # Center the y-axis around the mean of the data
            all_sums = left_sums + middle_sums + right_sums
            if all_sums:  # Ensure there is data to calculate mean and range
                mean_sum = np.mean(all_sums)
                max_deviation = max(abs(np.max(all_sums) - mean_sum), abs(np.min(all_sums) - mean_sum))
                ax1.set_ylim(mean_sum - max_deviation * 1.1, mean_sum + max_deviation * 1.1)  # Add 10% padding

            # Temporal Color Analysis: Full fabric
            average_color = np.mean(frame, axis=(0, 1))  # Average color of the entire frame
            global_red_intensities.append(average_color[2])  # Red channel
            global_green_intensities.append(average_color[1])  # Green channel
            global_blue_intensities.append(average_color[0])  # Blue channel

            # Update the Full Fabric Color Uniformity plot
            ax2.clear()
            ax2.plot(frame_numbers, global_red_intensities, color="red", label="Red")
            ax2.plot(frame_numbers, global_green_intensities, color="green", label="Green")
            ax2.plot(frame_numbers, global_blue_intensities, color="blue", label="Blue")
            ax2.set_title("Full Fabric Color Uniformity Over Time")
            ax2.set_xlabel("Frame Number")
            ax2.set_ylabel("Color Intensity")
            ax2.legend()
            ax2.set_ylim(0, 255)  # Color intensity range (0-255)

            # Convert the matplotlib figure to an image
            fig.canvas.draw()
            plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Resize the plot image to match the video frame width
            plot_image = cv2.resize(plot_image, (frame.shape[1], plot_image.shape[0]))

            # Combine the video frame and plot image vertically
            combined_frame = np.vstack((frame, plot_image))

            # Display the combined frame
            cv2.imshow("Video and Plots", combined_frame)

            # Increment frame count
            frame_count += 1

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()

    def convert_video(self):
        """
        Convert the generated video to a final format (e.g., MP4).
        """
        try:
            video_clip = VideoFileClip(self.output_video_path)
            final_video = video_clip.set_audio(None)  # Add audio here if needed
            final_video.write_videofile(self.final_output_path, codec="libx264", audio_codec="aac")
            print(f"Final video saved as {self.final_output_path}")
        except Exception as e:
            print(f"Error processing final video: {e}")


def main():
    # Initialize the FabricRollingVideo class
    fabric_video = FabricRollingVideo(
        image_path="im/perfectly_long_nuancecenteraldefect01-1.jpg",
        output_video_path="erfectly_long_nuancecenteraldefect01.avi",
        final_output_path="erfectly_long_nuancecenteraldefect01.mp4",
        frame_rate=30,
        total_duration=30,
        target_width=1920,
        target_height=1080
    )

    # Generate the fabric rolling video
    try:
        fabric_video.create_video()
    except Exception as e:
        print(f"Error creating fabric rolling video: {e}")
        return

    # Analyze and display the video with plots
    try:
        fabric_video.analyze_and_display_video()
    except Exception as e:
        print(f"Error analyzing and displaying video: {e}")

    # Convert to final format
    try:
        fabric_video.convert_video()
    except Exception as e:
        print(f"Error converting video: {e}")


if __name__ == "__main__":
    main()