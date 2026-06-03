"""
@author: Steven Ndung'u' and Hubert K

"""

### Georeference Masks

# Example found in the workstation workflow
# The aim of this script is to convert prediction patches saved
# in jp, png, etc to TIF format referencing the metadata of the
# original satelite imagery

import sys, os

sys.path.append("./")
import rasterio as rio
import pathlib
from make_dir import create_dir

#### Convert the PNG predictions to Rasters Tif format using the original image

# converts from png to tiff

PROJECT_ROOT = "D:\kanyamahanga\Bigwork\MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP"


def convert_img_to_tif_and_save(input_img, save_path, georef_img_tif, idx):
    # Input jpg/png image, to convert as geotiff
    # with rio.open(input_img, "r") as f:
    #     img = f.read([1])
    img = rio.open(str(input_img))
    img = img.read(1)

    # Input image for coordinate reference
    with rio.open(
        str(georef_img_tif)
        # + "/"
        # + str(input_img).replace("jpg", "tif").split("/")[-1]
    ) as naip:
        # open georeferenced.tif for writing

        with rio.open(
            str(save_path)
            + "/"
            + "{}.tif".format(str(input_img).split("\\")[-1].split(".")[0]),
            "w",
            driver="GTiff",
            count=1,
            height=img.shape[0],
            width=img.shape[1],
            dtype=img.dtype,
            crs=naip.crs,
            bounds=naip.bounds,
            transform=naip.transform,
        ) as dst:
            dst.write(img, indexes=1)

    ############## Set the paths #############


# path of png or jpg image predicted from smoothing algorithm
input_img = PROJECT_ROOT + "/results/predictions_Z14_UU/"
input_img = input_img
input_img = pathlib.Path(input_img)

# folder with original raster image (from original tif)
# Put the raster image you are trying to run predictions in this folder

georef_img_tif = PROJECT_ROOT + "/results/inputs_Z14_UU/"
georef_img_tif = pathlib.Path(georef_img_tif)
georef_img_tif = os.path.join(georef_img_tif)
# Path to save the outputs
save_path = PROJECT_ROOT + "/results/georeferenced_Z14_UU/"

# Create dir for saving predictions
output_dir = create_dir(save_path)
save_path = pathlib.Path(output_dir)
input_img, georef_img_tif, save_path

# Import the images, convert them to tif and save back in defined folder

images = list(input_img.glob("*"))
for idx, val in enumerate(images):
    # print("val:", val)
    georef_img_tif = (
        str(val).replace("predictions_Z14_UU", "inputs_Z14_UU").replace("PRED", "IMG")
    )
    # print("georef_img_tif:", georef_img_tif)
    convert_img_to_tif_and_save(str(val), save_path, georef_img_tif, idx)


# See the comments below for the next step of this prediction
#######################################################################################################################
#                      geospatial raster and vector formats
#                      run: python 3_solaris_raster_to_vector.py
#######################################################################################################################
