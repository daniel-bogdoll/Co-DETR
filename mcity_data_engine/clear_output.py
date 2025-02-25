import argparse
import os
import shutil


def move_best_model(config_file, dataset_name):
    """
    After training, move the best file from the standard output folder into a dedicated folder
    for the used dataset and Co-DETR config
    """
    output_folder_codetr = "output"
    param_config_name = os.path.splitext(os.path.basename(config_file))[0]

    # Find best model
    best_models = [
        f
        for f in os.listdir(output_folder_codetr)
        if "best_bbox" in f and f.endswith(".pth")
    ]
    if not best_models:
        print("Cannot find a checkpoint file with 'best_bbox' found.")
    else:
        if len(best_models) > 1:
            print(
                f"Found {len(best_models)} checkpoint files. Selecting {best_models[0]}."
            )
        checkpoint = best_models[0]
        checkpoint_path = os.path.join(output_folder_codetr, checkpoint)

        # Move best model
        output_dir = os.path.join(output_folder_codetr, "best")
        os.makedirs(output_dir, exist_ok=True)

        # Rename model file to format "config_dataset.pth"
        new_checkpoint_path = os.path.join(
            output_dir, f"{param_config_name}_{dataset_name}.pth"
        )
        shutil.move(checkpoint_path, new_checkpoint_path)
        print(f"Moved and renamed {checkpoint} to {new_checkpoint_path}")


def clear_output_folder():
    """
    After saving the best model file, clear the default output folder
    """
    try:
        if os.path.exists("output"):
            # Remove only the files directly in the 'output' directory
            for filename in os.listdir("output"):
                file_path = os.path.join("output", filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove file
                    print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error during clearing of output folder: {e}")


def main(config_file, dataset_name):
    print(f"Starting cleanup after Co-DETR training")
    # move_best_model(config_file, dataset_name)
    clear_output_folder()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clear output folder and process dataset."
    )
    parser.add_argument("--config_file", required=True, help="Path to the config file.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")

    args = parser.parse_args()
    main(args.config_file, args.dataset_name)
