# Fabric Rolling Video Analysis

A Python-based tool for creating and analyzing rolling fabric videos with real-time color uniformity analysis. This project helps in visualizing and analyzing fabric patterns, defects, and color consistency through automated video generation and analysis.

## Features

- Creates smooth rolling video effects from static fabric images
- Real-time analysis of color uniformity across the fabric
- Split-view analysis of left, middle, and right regions
- Temporal color analysis showing RGB channel variations
- Interactive visualization with synchronized video and analysis plots
- Support for high-resolution video output (1920x1080)

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install moviepy
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd fabric-rolling-video
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your fabric image in the `im` directory
2. Modify the parameters in `main()` function as needed:

```python
fabric_video = FabricRollingVideo(
    image_path="im/your_fabric_image.jpg",
    output_video_path="output.avi",
    final_output_path="final_output.mp4",
    frame_rate=30,
    total_duration=30,
    target_width=1920,
    target_height=1080
)
```

3. Run the script:
```bash
python record.py
```

## Analysis Features

### Real-time Visualization
- Video playback with synchronized analysis plots
- Interactive display showing:
  - Original rolling fabric video
  - Regional color gradation analysis
  - Full fabric color uniformity analysis

### Color Analysis
1. **Regional Color Gradation**
   - Splits fabric into three vertical regions
   - Tracks pixel value sums for each region
   - Visualizes regional variations over time

2. **Full Fabric Color Uniformity**
   - Monitors RGB channel intensities
   - Tracks global color variations
   - Helps identify color inconsistencies

## Output

The program generates three main outputs:
1. An initial AVI video file (raw output)
2. Real-time analysis display with plots
3. Final MP4 video file (compressed version)

## Class Structure

### FabricRollingVideo
- `__init__`: Initializes video parameters
- `resize_image`: Handles image resizing and padding
- `create_video`: Generates the rolling effect video
- `analyze_and_display_video`: Performs real-time analysis
- `convert_video`: Converts to final video format

## Controls

- Press 'q' to exit the analysis window
- Close the window to end the session

## Error Handling

The script includes comprehensive error handling for:
- Missing input files
- Video writer initialization
- Video processing errors
- Format conversion issues

## Contributing

Chaouch Thameur
