import numpy as np
import os
import sys
import cv2 as cv
from scipy.io import loadmat
import os
import tarfile
import shutil
from tqdm import tqdm
import requests
from sklearn.model_selection import train_test_split
import pandas as pd


def download_file(target_url, target_directory):
    """
    Function:
        - Creates a relative path `target_directory` in the current working directory if it doesn't exist yet
        - Downloads a file from `target_url` and puts it into the `target_directory`

    Arguments:
        - target_url (str): source URL of the file to be downloaded
        - target_directory (str): relative path of the folder into which the file is to be downloaded
    """

    # Get the filename from the target_url
    filename = target_url.split(os.path.sep)[-1]

    # Construct the target_path of the file
    target_path = os.path.join(target_directory, filename)

    # If the file has already been downloaded in the correct directory, terminate function
    if os.path.exists(target_path):
        print(f"The file '{filename}' has already been downloaded into '{target_directory}'")
        return

    # If the target_folder doesn't exist yet, create it
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)
        print(f"Created a new folder at '{target_directory}'")

    # Download the file into the target_folder
    print(f"Downloading '{filename}' from '{target_url}', please wait...")
    chunk_size = 1024
    r = requests.get(target_url, stream=True)
    size = int(r.headers['content-length'])
    print(f"File size = {size/chunk_size} KB.")
    with open(target_path, 'wb') as f:
        for data in tqdm(iterable=r.iter_content(chunk_size=chunk_size), total=size/chunk_size, unit='KB'):
            f.write(data)

    print(f"The file '{filename}' has been downloaded into '{target_directory}'")
    return

def extract_dataset(source_path, target_directory):
    """
    Function:
        - Unzips the tar file in `source_path` into `target_directory`

    Arguments:
        - source_path (str): relative path of the file to be unzipped
        - target_directory (str): relative path of the directory into which the tar file will be unzipped
    """
    # Extraction step
    tar = tarfile.open(source_path)
    tar.extractall(target_directory)
    tar.close()

    # Count the number of files/subfolders contained in the target_directory
    files_list = [file for file in os.listdir(target_directory)]
    print(f"Extraction complete!")
    print(f"'{source_path}' has been unzipped into '{target_directory}'; this directory now has {len(files_list)} files/subfolders")

def create_metadata_dataframe(path_to_cars_meta_mat='./metadata/devkit/cars_meta.mat', path_to_cars_train_annos_mat='./metadata/devkit/cars_train_annos.mat'):
    """
    Function:
        - Creates a dataframe from the two zipped metadata files provided as part of the downloaded Cars dataset;
          this dataframe will contain information about each image in the dataset, including file name, bounding box coordinates and car class

    Arguments:
        - path_to_cars_meta_mat (str): relative path to the file 'cars_meta.mat' unzipped from 'car_devkit.tgz'
        - path_to_cars_train_annos_mat (str): relative path to the file 'cars_trains_annos.mat' unzipped from 'car_devkit.tgz'
    Returns:
        - metadata (pd.DataFrame): a dataframe containing the metadata of the downloaded images
    """
    # Load the matlab file that contains the complete list of class names
    original_dict_metadata = loadmat(path_to_cars_meta_mat)

    # Create a dictionary `dict_classnumber_classname` that maps the class numbers to class names
    dict_classnumber_classname = {
        int(index+1): classname[0] for index, classname in enumerate(original_dict_metadata["class_names"][0])}

    # Load the matlab file that contains the metadata (bounding box details and class details) for each training sample
    original_dict_traindata = loadmat(path_to_cars_train_annos_mat)

    # Create an empty dictionary `dict_filename_metadata` that maps the file names and metadata
    dict_filename_metadata = {}

    # Populate `dict_filename_metadata` by looping through `original_dict_traindata`
    dict_filename_metadata = {image[5][0]: {"bbox_xmin": image[0][0][0], "bbox_ymin": image[1][0][0], "bbox_xmax": image[2][0][0], "bbox_ymax": image[3][0]
                                            [0], "classnumber": image[4][0][0], "classname": dict_classnumber_classname[image[4][0][0]]} for image in original_dict_traindata["annotations"][0]}

    # Convert `dict_filename_metadata` into a dataframe
    metadata = pd.DataFrame.from_dict(data=dict_filename_metadata, orient="index")

    # The "/" in 'Ram C/V Cargo Van Minivan 2012' turns out to be problematic so remove it
    metadata.loc[metadata["classname"] == 'Ram C/V Cargo Van Minivan 2012',
                 "classname"] = "Ram CV Cargo Van Minivan 2012"

    return metadata

def train_valid_test_split(metadata, train_size=0.6, valid_size=0.3):
    """
    Function:
        - Assign specified portions of the metadata dataframe into training, validation and testing sets

    Arguments:
        - metadata (pd.DataFrame): a dataframe containing the metadata of the downloaded images
        - train_size (float): the portion of the dataset to be assigned to the training set
        - valid_size (float): the portion of the dataset to be assigned to the validation set

    Returns:
        - final_df (pd.DataFrame): the metadata dataframe with a new column `category` that categorizes each sample into either the training, validation or testing set
    """

    assert train_size + valid_size <= 1, "training size + validation size has to be <= 1!"

    print(
        f"Splitting the dataframe into training, validation and testing (holdout) set with a ratio of {train_size:.2f} : {valid_size:.2f} : {1-train_size-valid_size:.2f}")

    # Split the df into training set and validation + testing set
    df_train, df_valid_test = train_test_split(
        metadata, train_size=train_size, random_state=88, stratify=metadata["classnumber"])
    df_train = df_train.assign(category="train")

    # Split the validation + testing set into validation set and testing set
    test_size = 1 - train_size - valid_size
    valid_size_adjusted = valid_size/(valid_size + test_size)

    df_valid, df_test = train_test_split(
        df_valid_test, train_size=valid_size_adjusted, random_state=88, stratify=df_valid_test["classnumber"])
    df_valid = df_valid.assign(category="valid")
    df_test = df_test.assign(category="test")

    # Combine the three dataframes
    df_final = pd.concat([df_train, df_valid, df_test], axis=0)

    total = metadata.shape[0]
    train = df_final["category"].value_counts()["train"]
    valid = df_final["category"].value_counts()["valid"]
    test = df_final["category"].value_counts()["test"]

    print()
    print(f"Out of {total:,} samples in the dataframe, \n{train:,} were assigned to the training set, \n{valid:,} to the validation set... \n...and {test:,} to the testing (holdout) set")
    return df_final

def crop_and_resize_images_using_metadata(metadata, source_directory="./data_download/cars_train", target_directory="./data_preprocessed"):
    """
    Function:
        - Crop and resize images in the source_directory and place the preprocessed images in the target_directory
    Arguments:
        - metadata (pd.DataFrame): a dataframe containing the metadata of the downloaded images
        - source_directory (str): relative path of the directory containing the images to be preprocessed
        - target_directory (str): relative path of the directory into which the preprocessed images are to be placed
        """

    image_list = os.listdir(source_directory)
    length_image_list = len(image_list)
    print(f"There are {length_image_list:,} images to preprocess in `{source_directory}`.")
    print("Preprocessing...")

    # If target_directory doesn't exist, create it
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    # Preprocess images in the source_path and place them in destination_path
    for index, filename in tqdm(enumerate(image_list)):

        xmin, ymin, xmax, ymax = metadata.loc[filename, [
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].values
        original_image_bgr = cv.imread(os.path.join(source_directory, filename))
        cropped_image = original_image_bgr[ymin:ymax, xmin:xmax]
        resized_image = cv.resize(cropped_image, (224, 224))

        cv.imwrite(os.path.join(target_directory, filename), resized_image)
    print(f"{length_image_list:,} images have been preprocessed and placed in `{target_directory}`")

def sort_images(metadata, directory="./data_preprocessed"):
    """
    Function:
        - Create folders `train`, `valid`  and `test` inside `directory`
        - In each of these three folders, create 196 subfolders that correspond to the 196 car classes
        - Sort images in the `directory` into their appropriate subfolders
    Argument:
        - metadata (pd.DataFrame): a dataframe containing the metadata of the downloaded images
        - directory: the directory containing images to be sorted
    """

    # Create folders `train`, `valid`  and `test` inside `directory` if they don't exist yet
    folders = metadata.category.unique()
    for folder in folders:
        if not os.path.exists(os.path.join(directory, folder)):
            os.makedirs(os.path.join(directory, folder))

        # Inside `train`, `valid`  and `test` subfolders, create one subfolder for every unique car class
        for classname in metadata.classname.unique():
            if not os.path.exists(os.path.join(directory, folder, classname)):
                os.makedirs((os.path.join(directory, folder, classname)))

        # Sort images in the `directory` into their appropriate subfolders
        for filename in metadata.loc[metadata["category"] == folder, :].index:
            classname = metadata.loc[filename, "classname"]
            # example source_path: ./data_preprocessed/00001.jpg
            source_path = os.path.join(directory, filename)
            # example target_path: ./data_preprocessed/train/Acura RL Sedan 2012/00001.jpg
            target_path = os.path.join(directory, folder, classname, filename)
            shutil.move(source_path, target_path)
