import argparse
import json
import os
from datetime import datetime

import numpy as np
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm


# JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description="Run inference with Co-DETR model")
    
    # Add arguments
    parser.add_argument('--data_folder_root', type=str, default="/launch/data/coco", help='Root directory for the dataset')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--checkpoint_folder', type=str, default="/launch/output", help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run inference on (default: cuda:0)')
    parser.add_argument('--results_folder', type=str, default="/launch/output/inference", help='Folder to save the inference results')
    
    args = parser.parse_args()

    try:
        # Collect all the samples
        jpg_filepaths = []

        for root, dirs, files in os.walk(args.data_folder_root):
            for file in files:
                if file.endswith(".jpg"):
                    filepath = os.path.join(root, file)
                    jpg_filepaths.append(filepath)

        # Find the best_bbox checkpoint file
        checkpoint_files = [f for f in os.listdir(args.checkpoint_folder) if "best_bbox" in f and f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint file with 'best_bbox' found in the checkpoint folder.")
        if len(checkpoint_files) > 1:
            print(f"Found {len(checkpoint_files)} checkpoint files. Selecting {checkpoint_files[0]}.")
        checkpoint = checkpoint_files[0]

        model = init_detector(args.config_file, checkpoint, device=args.device)

        os.makedirs(args.results_folder, exist_ok=True)

        # Run inference on every single file
        results = {}
        for filepath in tqdm(jpg_filepaths, desc="Running Co-DETR inference."):
            result = inference_detector(model, filepath)
            results[filepath] = result

        # Store results as JSON in results_folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filepath = os.path.join(args.results_folder, f"inference_results_{timestamp}.json")

        

        with open(results_filepath, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()