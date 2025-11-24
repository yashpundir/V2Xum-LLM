import os
import json
import subprocess
from datetime import datetime
import sys

# --- Configuration ---

# Path to the directory containing the videos
ACTIVITY_VIDEOS_PATH = "ActivityNet/Activity_Videos"

# Path to the cloned V2Xum repository
V2XUM_PROJECT_PATH = "V2Xum-LLM"

# Name of the inference script in the V2Xum project
V2XUM_INFERENCE_SCRIPT = "v2xumllm/inference.py" #<-- IMPORTANT: Change this if the script has a different name

# Name of the output file for the summaries
OUTPUT_FILE = "video_summaries.json"

# Supported video file extensions
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}

# --- End of Configuration ---


def find_videos(directory_path):
    """Finds all video files in a directory with supported extensions."""
    video_files = []
    print(f"Searching for videos in: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    print(f"Found {len(video_files)} videos.")
    return video_files


def run_v2xum_inference(video_path, v2xum_project_path):
    """
    Runs the V2Xum inference script for a single video and captures the output.
    """
    command = [
        sys.executable, # Use the current python interpreter
        "-m", "v2xumllm.inference", # Run as a module
        "--video_path",
        video_path
    ]

    print(f"  Running inference for: {os.path.basename(video_path)}")
 
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=v2xum_project_path # Run from the root of the V2Xum-LLM project
        )
        # Assuming the summary is the main output, strip any extra whitespace
        return process.stdout.strip()
    except FileNotFoundError:
        print(f"  [ERROR] Inference script not found at: {v2xum_script_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Inference failed for {os.path.basename(video_path)}.")
        print(f"  Return code: {e.returncode}")
        print(f"  Stderr: {e.stderr.strip()}")
        return None
    except Exception as e:
        print(f"  [ERROR] An unexpected error occurred for {os.path.basename(video_path)}: {e}")
        return None


def main():
    """
    Main function to find videos, run inference, and save summaries.
    """
    # Get absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, ACTIVITY_VIDEOS_PATH)
    v2xum_dir = os.path.join(base_dir, V2XUM_PROJECT_PATH)
    v2xum_script = os.path.join(v2xum_dir, V2XUM_INFERENCE_SCRIPT)

    if not os.path.isdir(videos_dir):
        print(f"Error: Video directory not found at '{videos_dir}'")
        return

    if not os.path.isfile(v2xum_script):
        print(f"Error: V2Xum inference script not found at '{v2xum_script}'")
        print("Please check the 'V2XUM_PROJECT_PATH' and 'V2XUM_INFERENCE_SCRIPT' variables.")
        return

    video_paths = find_videos(videos_dir)
    all_summaries = []

    for video_path in video_paths:
        summary = run_v2xum_inference(video_path, v2xum_dir)

        if summary:
            video_metadata = {
                "video_filename": os.path.basename(video_path),
                "video_path": os.path.abspath(video_path),
                "summary": summary,
                "summarization_timestamp_utc": datetime.utcnow().isoformat()
            }
            all_summaries.append(video_metadata)
            print(f"  Successfully generated summary for {os.path.basename(video_path)}.")
        else:
            # Log a failed summary as well, but with an empty summary field
            video_metadata = {
                "video_filename": os.path.basename(video_path),
                "video_path": os.path.abspath(video_path),
                "summary": "GENERATION_FAILED",
                "summarization_timestamp_utc": datetime.utcnow().isoformat()
            }
            all_summaries.append(video_metadata)


    # Save all summaries to a JSON file
    output_path = os.path.join(base_dir, OUTPUT_FILE)
    try:
        with open(output_path, "w") as f:
            json.dump(all_summaries, f, indent=4)
        print(f"\nAll summaries have been saved to: {output_path}")
    except IOError as e:
        print(f"\nError: Could not write to output file '{output_path}': {e}")


if __name__ == "__main__":
    main()
