# Add Importing files from different parallel folder
import sys
import yaml
import os

sys.path.append(
    "./"
)  # https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

from pathlib import Path
from src.data.data_display import display_predictions, get_data_paths
from src.data.load_data import load_data


config_path = "./flair-config.yml"  # Change to yours
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Creation of the train, val and test dictionnaries with the data file paths
d_train, d_val, d_test = load_data(config)

images = d_train["PATH_IMG"]

out_dir = Path(config["outputs"]["out_folder"], config["outputs"]["out_model_name"])
images_test = d_test["PATH_IMG"]
predictions = sorted(
    list(
        get_data_paths(
            Path(
                os.path.join(
                    out_dir, "predictions" + "_" + config["outputs"]["out_model_name"]
                )
            ),
            "PRED*.tif",
        )
    )
)

print(
    list(
        get_data_paths(
            Path(
                os.path.join(
                    out_dir, "predictions" + "_" + config["outputs"]["out_model_name"]
                )
            ),
            "PRED*.tif",
        )
    )
)
display_predictions(images_test, predictions, nb_samples=1)
# CMD: python ./visualizations/display_predictions.py
