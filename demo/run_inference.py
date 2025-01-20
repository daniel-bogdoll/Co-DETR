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
    # Collect all the samples
    data_folder_root = "/launch/data/coco"
    jpg_filepaths = []

    for root, dirs, files in os.walk(data_folder_root):
        for file in files:
            if file.endswith(".jpg"):
                filepath = os.path.join(root, file)
                jpg_filepaths.append(filepath)

    config_file = "/launch/projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py"
    checkpoint = "/launch/output/epoch_12.pth"
    device = "cuda:0"

    model = init_detector(config_file, checkpoint, device=device)

    results_folder = "/launch/output/inference"
    os.makedirs(results_folder, exist_ok=True)

    # Run inference on every single file
    results = {}
    for filepath in tqdm(jpg_filepaths, desc="Running Co-DETR inference."):
        result = inference_detector(model, filepath)
        results[filepath] = result
        break

    # Store results as JSON in results_folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filepath = os.path.join(results_folder, f"inference_results_{timestamp}.json")

    with open(results_filepath, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)

if __name__ == "__main__":
    main()