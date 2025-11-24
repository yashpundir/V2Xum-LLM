import os
import json
import argparse
import random
from datetime import datetime
import torch
import clip
from PIL import Image
import sys
from tqdm import tqdm  # Progress bar library

# --- Path Setup ---
V2XUM_PROJECT_PATH = "V2Xum-LLM" 
base_dir = os.path.dirname(os.path.abspath(__file__))
v2xum_path = os.path.join(base_dir, V2XUM_PROJECT_PATH)
if v2xum_path not in sys.path:
    sys.path.insert(0, v2xum_path)

# --- V2Xum Imports ---
from v2xumllm.constants import IMAGE_TOKEN_INDEX
from v2xumllm.conversation import conv_templates, SeparatorStyle
from v2xumllm.model.builder import load_pretrained_model
from v2xumllm.utils import disable_torch_init
from v2xumllm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor

# --- Torchvision Imports ---
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

# --- Configuration ---
ACTIVITY_VIDEOS_PATH = "ActivityNet/Activity_Videos"
OUTPUT_FILE = "video_summaries_batch.json"
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}

def find_videos(directory_path):
    """Finds all video files in a directory with supported extensions."""
    video_files = []
    print(f"Searching for videos in: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    print(f"Found {len(video_files)} total videos in folder.")
    return video_files

def run_inference_on_video(model, tokenizer, clip_model, transform, video_path, query):
    """Runs inference for a single video."""
    # Note: We removed the print statement here to keep the progress bar clean
    try:
        # 1. Extract video frames
        video_loader = VideoExtractor(N=100)
        _, images = video_loader.extract({'id': None, 'video': video_path})

        # 2. Preprocess frames
        images = transform(images / 255.0)
        images = images.to(torch.float16)

        # 3. Encode with CLIP
        with torch.no_grad():
            features = clip_model.encode_image(images.to('cuda'))

        # 4. Prepare prompt
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], "<video>\n " + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # 5. Generate summary
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=features[None,].cuda(),
                do_sample=True,
                temperature=0.05,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)[0]
        summary = decoded_outputs.strip()
        
        if summary.endswith(stop_str):
            summary = summary[:-len(stop_str)]
        
        if "answer: " in summary:
             summary = summary.split("answer: ")[-1].strip()

        return summary

    except Exception as e:
        # Print error nicely above the progress bar
        tqdm.write(f"  [ERROR] Failed to process {os.path.basename(video_path)}: {e}")
        return None

def main(args):
    # --- 1. One-time Model Loading ---
    print("Loading models... This may take a moment.")
    disable_torch_init()
    
    tokenizer, model, _ = load_pretrained_model(args, args.stage2)
    model.to(device='cuda', dtype=torch.float16)

    clip_model, _ = clip.load(args.clip_path, device='cuda')
    clip_model.eval()
    print("Models loaded successfully.")

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # --- 2. Find Videos ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, ACTIVITY_VIDEOS_PATH)
    if not os.path.isdir(videos_dir):
        print(f"Error: Video directory not found at '{videos_dir}'")
        return
    
    video_paths = find_videos(videos_dir)
    
    # --- 3. Load Existing Progress (Resume Capability) ---
    output_path = os.path.join(base_dir, OUTPUT_FILE)
    all_summaries = []
    processed_filenames = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_summaries = data
                    for item in all_summaries:
                        if "video_filename" in item:
                            processed_filenames.add(item["video_filename"])
            print(f"Found existing output file. Resuming... ({len(processed_filenames)} videos already done).")
        except json.JSONDecodeError:
            print("Output file exists but is empty or corrupted. Starting fresh.")

    # Filter list to only process new videos
    videos_to_process = [v for v in video_paths if os.path.basename(v) not in processed_filenames]
    
    if not videos_to_process:
        print("All videos have already been processed! Exiting.")
        return

    print(f"Starting inference on {len(videos_to_process)} remaining videos.\n")

    # --- 4. Loop and Run Inference (With Progress Bar) ---
    query = "Please generate BOTH video and text summarization for this video."
    
    # tqdm creates the dynamic progress bar with ETA
    for video_path in tqdm(videos_to_process, desc="Summarizing", unit="video"):
        summary = run_inference_on_video(model, tokenizer, clip_model, transform, video_path, query)

        video_metadata = {
            "video_filename": os.path.basename(video_path),
            "video_path": os.path.abspath(video_path),
            "summary": summary if summary else "GENERATION_FAILED",
            "summarization_timestamp_utc": datetime.utcnow().isoformat()
        }
        all_summaries.append(video_metadata)

        # --- 5. Save Immediately After Each Video ---
        try:
            with open(output_path, "w") as f:
                json.dump(all_summaries, f, indent=4)
        except IOError as e:
            tqdm.write(f"Error saving file: {e}")

    print(f"\nProcessing complete. All summaries saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Video Summarization")
    
    # FIX: Use os.path.join to ensure paths are relative to V2XUM_PROJECT_PATH
    parser.add_argument("--clip_path", type=str, 
        default=os.path.join(V2XUM_PROJECT_PATH, "checkpoints/clip/ViT-L-14.pt"))
    
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5")
    
    # Point to the v2xumllm-7b folder we set up earlier
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, 
        default=os.path.join(V2XUM_PROJECT_PATH, "checkpoints/llava-vicuna-v1-5-7b-stage1/mm_projector.bin"))
    
    parser.add_argument("--stage2", type=str, 
        default=os.path.join(V2XUM_PROJECT_PATH, "checkpoints/v2xumllm-vicuna-v1-5-7b-stage2-e2"))
        
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)