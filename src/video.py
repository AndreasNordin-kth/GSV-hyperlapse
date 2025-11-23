import os
import subprocess

def create_high_quality_video(
    input_directory,
    output_file="output.mp4",
    fps=24,
    crf=20,
    preset="medium",
    bitrate="50M",
    maxrate="100M",
    bufsize="100M"
):
    """
    Combines images into a high-quality video using NVENC with enhanced bitrate control.

    Args:
        input_directory (str): Path to the directory containing images.
        output_file (str): Name of the output video file.
        fps (int): Frames per second for the output video (default 24).
        crf (int): Constant Rate Factor for quality (lower is better, default 20).
        preset (str):  Preset for quality ('medium' or 'slow').
        bitrate (str): Minimum bitrate for the output video (e.g., '40M').
        maxrate (str): Maximum bitrate for variable bitrate (e.g., '50M').
        bufsize (str): Buffer size for bitrate control (e.g., '100M').
    """
    # Ensure input directory exists
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Input directory '{input_directory}' does not exist.")
    
    # Get a sorted list of image files based on index
    image_files = sorted(
        [f for f in os.listdir(input_directory) if f.endswith('.jpg')],
        key=lambda x: int(x.split('_')[0])
    )
    
    if not image_files:
        raise ValueError("No .jpg files found in the input directory.")
    
    # Create a temporary text file listing all image files
    with open("image_list.txt", "w") as file_list:
        for image_file in image_files:
            file_list.write(f"file '{os.path.join(input_directory, image_file)}'\n")
    
    # FFMPEG command to create video
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),  # Frames per second
        "-i", "image_list.txt",
        "-vcodec", "libx264",  # libx264
        "-preset", preset,  # Quality preset
        "-crf", str(crf),  # Constant Quality for NVENC
        "-b:v", bitrate,  # Minimum bitrate
        "-maxrate", maxrate,  # Maximum bitrate
        "-bufsize", bufsize,  # Buffer size for smoothing bitrate
        "-pix_fmt", "yuv420p",  # Ensures compatibility
        output_file
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"High-quality video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFMPEG failed with error: {e}")
    finally:
        # Remove the temporary index file
        if os.path.exists("image_list.txt"):
            os.remove("image_list.txt")
