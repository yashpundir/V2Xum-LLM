import json
import re
import os
import numpy as np
from tqdm import tqdm
import evaluate

# --- Configuration ---
PREDICTIONS_FILE = "video_summaries_batch.json"
GROUND_TRUTH_FILE = "captions_1905_complete.json" # Or captions_1905.json

def clean_text(text):
    """Removes [01, 02] tags to leave just the natural language."""
    # Regex to remove [digits, digits]
    cleaned = re.sub(r'\[.*?\]', '', text)
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def main():
    # 1. Load Data
    print("Loading data...")
    with open(PREDICTIONS_FILE) as f:
        preds_data = json.load(f)
        
    with open(GROUND_TRUTH_FILE) as f:
        gt_data = json.load(f)
        
    # Convert GT list to dict for fast lookup
    gt_dict = {}
    if isinstance(gt_data, list):
        for item in gt_data:
            vid = item.get('video_id', '')
            gt_dict[vid] = item
            if vid.startswith('v_'): gt_dict[vid[2:]] = item
    
    # 2. Prepare Lists for Evaluation
    predictions = []
    references = []
    
    print("Aligning predictions with ground truth...")
    for pred_entry in tqdm(preds_data):
        if pred_entry['summary'] == "GENERATION_FAILED":
            continue
            
        vid_filename = pred_entry['video_filename']
        vid_id = os.path.splitext(vid_filename)[0]
        
        # Lookup GT
        gt_entry = gt_dict.get(vid_id)
        if not gt_entry:
            if vid_id.startswith('v_'): gt_entry = gt_dict.get(vid_id[2:])
            elif 'v_' + vid_id in gt_dict: gt_entry = gt_dict.get('v_' + vid_id)
            
        if not gt_entry:
            continue
            
        # Prepare Texts
        # Prediction: Clean up the tags
        pred_text = clean_text(pred_entry['summary'])
        
        # Reference: ActivityNet usually has multiple sentences. Join them.
        ref_sentences = gt_entry.get('sentences', [])
        ref_text = " ".join(ref_sentences)
        
        predictions.append(pred_text)
        references.append(ref_text)

    print(f"Evaluating on {len(predictions)} samples...")

    # 3. Load Metrics
    # Note: METEOR and CIDEr usually require specific setups or java.
    # ROUGE and BLEU are easier.
    
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    
    # 4. Compute
    print("Computing BLEU...")
    # BLEU-4
    bleu4_results = bleu.compute(predictions=predictions, references=references, max_order=4)
    
    print("Computing ROUGE...")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    
    print("Computing METEOR...")
    meteor_results = meteor.compute(predictions=predictions, references=references)
    
    # For CIDEr, it's a bit harder with 'evaluate' library directly.
    # We often use a specialized package or just report these 3 first.
    
    print("\n--- V2T Results ---")
    print(f"BLEU-4:  {bleu4_results['bleu'] * 100:.2f}")
    print(f"METEOR:  {meteor_results['meteor'] * 100:.2f}")
    print(f"ROUGE-L: {rouge_results['rougeL'] * 100:.2f}")

if __name__ == "__main__":
    main()