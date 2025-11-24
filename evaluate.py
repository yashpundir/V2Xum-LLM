import json
import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
import re

# --- Configuration ---
PREDICTIONS_FILE = "video_summaries_batch copy.json"
# Point this to your ActivityNet captions file (e.g., 'val_1.json' or 'captions.json')
GROUND_TRUTH_FILE = "captions_1905_complete.json" 
# Point this to your folder of videos (Used to verify file existence)
VIDEOS_DIR = "V2Xum-LLM/Activity_Videos" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    print(f"Loading CLIP on {DEVICE}...")
    model, preprocess = clip.load("V2Xum-LLM/checkpoints/clip/ViT-L-14.pt", device=DEVICE)
    return model, preprocess

def parse_keyframes(summary_text):
    """
    Extracts keyframe indices [05, 10] from text.
    Returns sorted list of integers.
    """
    matches = re.findall(r'\[(.*?)\]', summary_text)
    indices = []
    for match in matches:
        # Split by comma, strip whitespace
        parts = [p.strip() for p in match.split(',')]
        for p in parts:
            if p.isdigit():
                indices.append(int(p))
    return sorted(list(set(indices)))

def get_frame_from_video(video_path, normalized_index, N=100):
    """
    Extracts a specific frame based on the 0-99 NORMALIZED index.
    
    Args:
        video_path (str): Path to the video file.
        normalized_index (int): The index from the model summary (0-99).
        N (int): The number of sample points the model was trained with (default 100).
        
    Returns:
        PIL.Image: The extracted frame, or None if failed.
    """
    try:
        # Initialize VideoReader
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # 1. Calculate the Raw Frame Index
        # We map the range [0, N-1] to [0, total_frames-1]
        # Logic: frame_idx = (norm_idx / 99) * (total_frames - 1)
        if N <= 1:
            raw_frame_idx = 0
        else:
            raw_frame_idx = int(round((normalized_index / (N - 1.0)) * (total_frames - 1)))
            
        # 2. Safety Clamp
        raw_frame_idx = max(0, min(total_frames - 1, raw_frame_idx))
        
        # 3. Extract Frame
        frame = vr[raw_frame_idx].asnumpy()
        return Image.fromarray(frame)
        
    except Exception as e:
        # Uncomment to debug specific video failures
        print(f"Error reading frame from {video_path}: {e}")
        return None

def get_gt_keyframes_from_timestamps(gt_entry, video_duration, target_count, N=100):
    """
    Converts GT segments to 0-99 indices.
    ADAPTIVE LOGIC: Samples exactly 'target_count' frames distributed across the GT segments.
    This ensures the Ground Truth density matches the Prediction density for fair F_CLIP comparison.
    """
    timestamps = gt_entry.get('timestamps', [])
    
    # Normalize format to list of lists
    segments = []
    if isinstance(timestamps, list):
        if len(timestamps) > 0 and isinstance(timestamps[0], list):
             segments = timestamps
        elif len(timestamps) == 2 and isinstance(timestamps[0], (int, float)):
             segments = [timestamps]
    
    if not segments or target_count == 0:
        return []

    # 1. Calculate total duration of all valid segments combined
    total_seg_duration = 0
    valid_segments = []
    for start, end in segments:
        if start >= end: continue
        duration = end - start
        valid_segments.append((start, end, duration))
        total_seg_duration += duration
    
    if total_seg_duration == 0: return []

    true_indices = []
    
    # 2. Distribute frames proportionally to segment length
    remaining_frames = target_count
    
    for i, (start, end, duration) in enumerate(valid_segments):
        # Calculate how many frames this segment deserves
        if i == len(valid_segments) - 1:
            # Last segment takes whatever is left to ensure exact count
            n_frames = remaining_frames
        else:
            ratio = duration / total_seg_duration
            n_frames = int(round(target_count * ratio))
            # Ensure at least 1 frame if segment exists and we have budget
            n_frames = max(1, n_frames) if n_frames == 0 and remaining_frames > 0 else n_frames
            # Don't take more than we have left
            n_frames = min(n_frames, remaining_frames)
        
        remaining_frames -= n_frames
        
        if n_frames > 0:
            # Sample evenly within this segment using linspace
            # We take points between start and end
            points = np.linspace(start, end, n_frames + 2)[1:-1]
            for p in points:
                if video_duration > 0:
                    norm_pos = p / video_duration
                    idx = int(norm_pos * (N - 1))
                    idx = max(0, min(N-1, idx))
                    true_indices.append(idx)

    return sorted(list(set(true_indices)))

def calculate_f_clip(pred_features, gt_features):
    """Computes F_CLIP score between two sets of CLIP embeddings."""
    if len(pred_features) == 0 or len(gt_features) == 0:
        return 0.0

    # Normalize features
    pred_features = pred_features / pred_features.norm(dim=1, keepdim=True)
    gt_features = gt_features / gt_features.norm(dim=1, keepdim=True)

    # Similarity Matrix (Cosine Similarity)
    sim_matrix = torch.mm(pred_features, gt_features.t())
    sim_matrix = torch.relu(sim_matrix) # Clamp negatives

    # Precision: For each pred, max sim with any GT
    precision_scores, _ = sim_matrix.max(dim=1)
    precision = precision_scores.mean().item()

    # Recall: For each GT, max sim with any pred
    recall_scores, _ = sim_matrix.max(dim=0)
    recall = recall_scores.mean().item()

    if (precision + recall) == 0:
        return 0.0
        
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score

def calculate_cross_f_clip(pred_v_feats, pred_t_feats, gt_v_feats, gt_t_feats):
    """
    Computes Cross-F_CLIP score.
    Formula: (F_CLIP(Pred_Video, GT_Text) + F_CLIP(GT_Video, Pred_Text)) / 2
    """
    # 1. Video(Pred) vs Text(GT)
    score_v_pred_t_gt = calculate_f_clip(pred_v_feats, gt_t_feats)
    
    # 2. Video(GT) vs Text(Pred)
    score_v_gt_t_pred = calculate_f_clip(gt_v_feats, pred_t_feats)
    
    return (score_v_pred_t_gt + score_v_gt_t_pred) / 2.0

def main():
    # --- Load Predictions ---
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Error: {PREDICTIONS_FILE} not found.")
        return
    with open(PREDICTIONS_FILE) as f:
        predictions = json.load(f)
    
    # --- Load GT (FIXED FOR YOUR LIST FORMAT) ---
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: {GROUND_TRUTH_FILE} not found.")
        return
        
    print("Loading Ground Truth...")
    with open(GROUND_TRUTH_FILE) as f:
        raw_gt = json.load(f)
    
    # Convert LIST to DICT for fast lookup
    # We map both "v_ID" and "ID" to the entry just in case
    ground_truth = {}
    
    # If it's a list (like your snippet), iterate and build the dict
    if isinstance(raw_gt, list):
        for item in raw_gt:
            vid_id = item.get('video_id', '') # e.g. v_QOlSCBRmfWY
            # Store as-is
            ground_truth[vid_id] = item
            # Also store without 'v_' prefix if present
            if vid_id.startswith('v_'):
                ground_truth[vid_id[2:]] = item
    
    # If it's a dict with 'database' key (Official format fallback)
    elif isinstance(raw_gt, dict) and 'database' in raw_gt:
        ground_truth = raw_gt['database']
    
    # If it's already a flat dict
    elif isinstance(raw_gt, dict):
        ground_truth = raw_gt

    model, preprocess = load_clip_model()
    
    f_clip_scores = []
    cross_f_clip_scores = []
    
    print(f"Evaluating {len(predictions)} videos...")
    
    for entry in tqdm(predictions):
        if entry['summary'] == "GENERATION_FAILED":
            continue

        # 1. Match Prediction to GT Entry
        vid_filename = entry['video_filename']
        vid_id_short = os.path.splitext(vid_filename)[0] # "v_12345" or "12345"
        
        # Lookup directly in our prepared dictionary
        gt_entry = ground_truth.get(vid_id_short)
        
        if not gt_entry:
            # Try adding 'v_' if missing
            if not vid_id_short.startswith('v_'):
                 gt_entry = ground_truth.get('v_' + vid_id_short)
        
        if not gt_entry:
            continue # Skip if still no GT found

        # --- Prepare Features ---
        
        # A. Predicted Video Features
        pred_indices = parse_keyframes(entry['summary'])
        if not pred_indices: continue
        
        pred_images = []
        for idx in pred_indices:
            img = get_frame_from_video(entry['video_path'], idx)
            if img: pred_images.append(preprocess(img).unsqueeze(0).to(DEVICE))
        if not pred_images: continue
        
        with torch.no_grad():
            pred_v_feats = model.encode_image(torch.cat(pred_images))

        # B. GT Video Features
        try:
            vr = VideoReader(entry['video_path'], ctx=cpu(0))
            duration = len(vr) / vr.get_avg_fps()
        except:
            continue
            
        # Sample GT frames adaptively
        gt_indices = get_gt_keyframes_from_timestamps(gt_entry, duration, target_count=len(pred_indices))
        if not gt_indices: continue
        
        gt_images = []
        for idx in gt_indices:
            img = get_frame_from_video(entry['video_path'], idx)
            if img: gt_images.append(preprocess(img).unsqueeze(0).to(DEVICE))
        if not gt_images: continue
        
        with torch.no_grad():
            gt_v_feats = model.encode_image(torch.cat(gt_images))

        # C. Predicted Text Features
        pred_text_clean = re.sub(r'\[.*?\]', '', entry['summary']).strip()
        if not pred_text_clean: continue
        
        with torch.no_grad():
            pred_t_tokenized = clip.tokenize([pred_text_clean[:300]], truncate=True).to(DEVICE)
            pred_t_feats = model.encode_text(pred_t_tokenized)

        # D. GT Text Features
        gt_sentences = gt_entry.get('sentences', [])
        if not gt_sentences: continue
        gt_text_full = " ".join(gt_sentences)
        
        with torch.no_grad():
            gt_t_tokenized = clip.tokenize([gt_text_full[:300]], truncate=True).to(DEVICE)
            gt_t_feats = model.encode_text(gt_t_tokenized)

        # --- Calculate Metrics ---
        score_f = calculate_f_clip(pred_v_feats, gt_v_feats)
        f_clip_scores.append(score_f)
        
        score_cross = calculate_cross_f_clip(pred_v_feats, pred_t_feats, gt_v_feats, gt_t_feats)
        cross_f_clip_scores.append(score_cross)

    print(f"\nResults computed on {len(f_clip_scores)} videos.")
    if f_clip_scores:
        print(f"Average F_CLIP:       {np.mean(f_clip_scores):.4f}")
        print(f"Average Cross-F_CLIP: {np.mean(cross_f_clip_scores):.4f}")
    else:
        print("No scores computed. Check GT file IDs.")

if __name__ == "__main__":
    main()