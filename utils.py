import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from google_images_download import google_images_download
from tqdm import tqdm

def insert_bounding_box(image, bounding_box_coordinates, color=(0, 0, 255), thickness=3, use_normalized_coordinates=True):
    """
    Function:
        - Adds a bounding box to the numpy representation of an image.
        - The bounding box coordinates can be specified in either absolute (pixel) or normalized coordinates.

    Arguments:
        - image (np.ndarray): a numpy representation of the image
        - bounding_box_coordinates (tuple): a tuple containing (ymin, xmin, ymax, xmax) of the bounding box.
        - color (str): string representation of the color of the bounding box. Default is 'red'.
        - thickness (int): line thickness of the bounding box. Default value is 4.
        - use_normalized_coordinates (bool): If True (default), treat coordinates as relative to the image dimension.  Otherwise treat
          coordinates as absolute.

    Returns:
        - image_with_bounding_box (np.ndarray): a numpy representation of the image with bounding box

    """
    # Get the image's width and height
    (height, width, _) = image.shape

    # Unpack the bounding box coordinates
    ymin, xmin, ymax, xmax = bounding_box_coordinates

    # If the coordinates are normalized, get the default value of the coordinates
    if use_normalized_coordinates:
        (xmin, xmax, ymin, ymax) = (xmin * width, xmax * width, ymin * height, ymax * height)

    # Draw the bounding box
    image_with_bounding_box = cv.rectangle(img=image,
                                           pt1=(int(xmin), int(ymin)),
                                           pt2=(int(xmax), int(ymax)),
                                           color=(255, 0, 0),
                                           thickness=3)
    # Return the modified np representation of the image into which the bounding box has been drawn
    return image_with_bounding_box

def show_image_from_directory(metadata, directory="./data_download/cars_train", index=0):
    """
    Function:
        - show a car image in a folder, along with its preprocessed versions
    Arguments:
        - metadata (pd.DataFrame): a dataframe containing the metadata of the downloaded images
        - directory (str): relative path to the folder that contains the images
        - index (int): the index of the image to be shown
    """

    # Get an image from the directory
    image_list = os.listdir(directory)
    filename = image_list[index]
    original_image_bgr = cv.imread(os.path.join(directory, filename))

    # Using the metadata dataframe provided, print the image's metadata
    print(f"Index: {index}")
    print(f"Filename: {filename}")
    classname = metadata.loc[filename, ["classname"]].values[0]
    print(f"Car class: {classname}")
    print("="*80)
    width, height, _ = original_image_bgr.shape
    print(f"Original width: {width}, original height: {height}")
    xmin, ymin, xmax, ymax = metadata.loc[filename, [
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].values
    print(
        f"Bounding box position (absolute) - xmin: {xmin} | ymin: {ymin} | xmax: {xmax} | ymax: {ymax}")
    print("="*80)

    # The preprocessing steps
    original_image_rgb = cv.cvtColor(original_image_bgr, cv.COLOR_BGR2RGB)
    image_with_bounding_box = insert_bounding_box(
        original_image_rgb.copy(), (ymin, xmin, ymax, xmax), use_normalized_coordinates=False)
    cropped_image = original_image_rgb[ymin:ymax, xmin:xmax]
    resized_image = cv.resize(cropped_image, (224, 224))

    # Visualize the original image as well as the three preprocessed images
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
    ax1.imshow(image_with_bounding_box, interpolation='nearest')
    ax1.set_title("1. Original Image with Bounding Box")
    ax2.imshow(cropped_image, interpolation='nearest')
    ax2.set_title("2. Cropped Image")
    ax3.imshow(resized_image, interpolation='nearest')
    ax3.set_title("3. Resized Image (224 x 224 pixels)")
    plt.show()

def crop_image_with_bounding_box(image_np, bounding_box_coordinates, use_normalized_coordinates=True):
    """
    Function:
        - Use the bounding box coordinates provided to crop an image.
        - The bounding box coordinates can be specified in either absolute (pixel) or normalized coordinates.

    Arguments:
        - image_np (nd_array): a numpy representation of the image
        - bounding_box_coordinates (tuple): a tuple containing (ymin, xmin, ymax, xmax) of the bounding box.
        - use_normalized_coordinates (bool): If True (default), treat coordinates as relative to the image dimension.  Otherwise treat
          coordinates as absolute.
    """
    # Get its width and height
    (im_height, im_width, _) = image_np.shape

    # Unpack the coordinates
    ymin,xmin,ymax,xmax = bounding_box_coordinates

    # Get the borders of the bounding box
    if use_normalized_coordinates:
        (xmin,xmax,ymin,ymax) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))

    # Cropping
    cropped_img_np = image_np[ymin:ymax, xmin: xmax]

    return cropped_img_np

def download_images(keywords, target_directory = "./data_unprocessed"):
    """
    Function:
        - Download images from Google using keywords provided
    Arguments:
        - keywords (list): list of keywords to be used in the search process
        - target_directory (str): relative path of the directory into which the images will be downloaded
    """

    for index, keyword in enumerate(keywords):
        print("============================================")
        print(f"Processing {index+1}th keyword out of {len(keywords)} keywords.")
        print(f"Downloading images using keyword '{keyword}'...")

        response = google_images_download.googleimagesdownload()
        arguments = {"keywords": keyword,
                     "format": "jpg",
                     "limit":100,
                     "print_urls":True,
                     "size": "medium",
                     "output_directory": target_directory}
        try:
            response.download(arguments)
        except:
            pass

def copy_images_between_two_nested_folders(source_directory = "./data_cropped_preprocessed2", target_directory = "./data_preprocessed/train"):
    """
    Function: transfer images  between two nested foloders
    Arguments:
        - source_directory: the directory containing images to be copied
        - target_directory: the directory into which the images will be placed
    """
    for root, directories, files in tqdm(os.walk(source_directory, topdown=False)):
        for i, file in tqdm(enumerate(files)):
            # Get the source_path
            source_path = os.path.join(root, file)

            # Get the  target_path
            classname = source_path.split(os.path.sep)[-2]
            target_path = os.path.join(target_directory,classname, file)
            shutil.copyfile(source_path, target_path)

def make_training_graph(csv_path, save_as_file = False, filename = "log1.html"):
    """
    Function:
        - draw an interactive graph of accuracy/validation accuracy vs epochs
    Arguments:
        - csv_path (str): relative path to the csv file that contains the training log of a model

    """
    training_log = pd.read_csv(csv_path)

    trace_acc = go.Scatter(x = training_log.index, y = training_log["acc"], mode = "lines", name = "Accuracy")
    trace_val_acc = go.Scatter(x = training_log.index, y = training_log.val_acc, mode = "lines", name = "Validation accuracy")

    data = [trace_acc, trace_val_acc]

    layout = go.Layout(title = "Training graph",
                       xaxis=dict(title='Epochs'),
                       yaxis=dict(title='Accuracy'),
                      yaxis2=dict(title='Loss', side = "right"),
                    width=600,
                    height=300)
    figure = go.Figure(data = data, layout = layout)
    if save_as_file:
        plot(figure, filename = filename)
    else:
        iplot(figure)
