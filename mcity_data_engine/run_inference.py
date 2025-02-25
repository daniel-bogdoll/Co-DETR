import argparse
import json
import os
from datetime import datetime

import numpy as np
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    """Run Co-DETR inference on images and save results as JSON"""

    # Argument parsing
    parser = argparse.ArgumentParser(description="Run inference with Co-DETR model")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Co-DETR configuration",
    )
    parser.add_argument(
        "--data_folder_root",
        type=str,
        help="Root directory for the dataset",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="/launch/output/inference",
        help="Folder to save the inference results",
    )

    args = parser.parse_args()

    try:
        # Collect all the samples
        jpg_filepaths = []

        data_folder = os.path.join("data", args.data_folder_root)

        print(f"Collecting samples from {data_folder}")
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".jpg"):
                    filepath = os.path.join(root, file)
                    jpg_filepaths.append(filepath)

        checkpoint_path = os.path.join("hf_models", args.model_checkpoint)

        model = init_detector(args.config_file, checkpoint_path, device=args.device)

        os.makedirs(args.results_folder, exist_ok=True)

        # Run inference on every single file
        results = {}
        for filepath in tqdm(jpg_filepaths, desc="Running Co-DETR inference."):
            result = inference_detector(model, filepath)
            results[filepath] = result

        # Store results as JSON in results_folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filepath = os.path.join(
            args.results_folder, f"inference_results_{timestamp}.json"
        )

        with open(results_filepath, "w") as f:
            json.dump(results, f, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
