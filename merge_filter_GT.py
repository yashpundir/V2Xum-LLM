import json
import os

# --- Configuration ---
PREDICTIONS_FILE = "video_summaries_batch.json"

# List ALL your GT files here
GT_FILES = [
    "activitynet_captions_train.json",
    "activitynet_captions_val1.json", # <--- You need to download this
    "activitynet_captions_val2.json"  # <--- And maybe this
]

OUTPUT_FILE = "captions_1905_complete.json"

def main():
    # 1. Load processed IDs
    with open(PREDICTIONS_FILE, 'r') as f:
        preds = json.load(f)
    
    processed_ids = set()
    for p in preds:
        fname = p.get('video_filename', '')
        if fname:
            vid_id = os.path.splitext(fname)[0]
            processed_ids.add(vid_id)
            if vid_id.startswith('v_'): processed_ids.add(vid_id[2:])

    print(f"Looking for {len(processed_ids)} video IDs...")

    # 2. Merge GT Files
    merged_gt = {}
    for gt_path in GT_FILES:
        if not os.path.exists(gt_path):
            print(f"Skipping missing file: {gt_path}")
            continue
            
        print(f"Loading {gt_path}...")
        with open(gt_path, 'r') as f:
            data = json.load(f)
            
        # Handle database wrapper
        if isinstance(data, dict) and 'database' in data:
            db = data['database']
        else:
            db = data # Assume list or flat dict
            
        # Normalize to dict
        if isinstance(db, list):
            for item in db:
                vid = item.get('video_id', '')
                merged_gt[vid] = item
        elif isinstance(db, dict):
            merged_gt.update(db)

    # 3. Filter
    final_gt = []
    for vid_id, data in merged_gt.items():
        # Check if we processed this video
        if vid_id in processed_ids or f"v_{vid_id}" in processed_ids:
            final_gt.append(data)

    print(f"Found matching GT for {len(final_gt)} videos.")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_gt, f, indent=4)
    print(f"Saved merged GT to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()