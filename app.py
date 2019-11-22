
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import base64
import os
import matplotlib.pyplot as plt
import cv2
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import io
import gdown

from predict import *
from utils import *
from preprocess import *


from flask import Flask, render_template, url_for
server = Flask(__name__)


@server.route("/")
@server.route("/home")

def home():
    return render_template('index.html')


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/'
)

server = app.server
app.config['suppress_callback_exceptions']=True


print("\n...Checking if you have downloaded the model, please wait...\n")
CWD = os.getcwd()

MODEL_DIR_NAME = "models"
MODEL_CLF_NAME = "initial_classifer_augmented-0.97.hdf5"
MODEL_CLF_WEIGHT = "resnet152_weights_tf.h5"
MODEL_OBJ_NAME = "frozen_inference_graph.pb"

MODEL_DIR = os.path.join(CWD, MODEL_DIR_NAME)
MODEL_CLF_PATH = os.path.join(MODEL_DIR, MODEL_CLF_NAME)
MODEL_CLF_WEIGHT_PATH = os.path.join(MODEL_DIR, MODEL_CLF_WEIGHT)
MODEL_OBJ_PATH = os.path.join(MODEL_DIR, MODEL_OBJ_NAME)

if os.path.exists(MODEL_CLF_PATH) and os.path.exists(MODEL_CLF_WEIGHT_PATH) and os.path.exists(MODEL_OBJ_PATH):
    MODEL_AVAILABLE = True
else:
    print(f"Not all required models are available!")
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        print(f"Created a folder `{MODEL_DIR_NAME}` in current working directory that will host the models.")
    MODEL_AVAILABLE = False


if not MODEL_AVAILABLE:
    print("\n...Downloading models from GDrive (takes ~2 minutes)...\n")
    MODEL_CLF_URL = "https://drive.google.com/uc?id=1mOpZ3PG6VyulfLlUnQJdiysMF3T7SLE4"
    MODEL_CLF_WEIGHT_URL = "https://drive.google.com/uc?id=17nB4ZHpTSPkFiWd2-VINTB79Zx9z7Q_5"
    MODEL_OBJ_URL = "https://drive.google.com/uc?id=1D14F3YOBCYotojq_kGbK9aFW9PMIUUln"
    gdown.download(MODEL_CLF_URL, MODEL_CLF_PATH, quiet = False)
    gdown.download(MODEL_CLF_WEIGHT_URL, MODEL_CLF_WEIGHT_PATH, quiet = False)
    gdown.download(MODEL_OBJ_URL, MODEL_OBJ_PATH, quiet = False)
    print(f"\n..Trained models successfully downloaded to `{MODEL_DIR_NAME}` folder...\n")



################################### Load models #######################

# Classification model
print("\n...Loading models, please wait...\n")
model = load_pretrained_model(model_weights_path=MODEL_CLF_PATH)
#model = load_pretrained_model(model_weights_path = 'models/initial_classifer_augmented-0.97.hdf5')
model._make_predict_function()

# Object detection model
#PATH_TO_FROZEN_GRAPH = "models/frozen_inference_graph.pb"
#detection_graph = build_localization_model(PATH_TO_FROZEN_GRAPH)
detection_graph = build_localization_model(MODEL_OBJ_PATH)
########################### Load variables ##########################################

classnames = np.array(['AM General Hummer SUV 2000',
 'Acura RL Sedan 2012',
 'Acura TL Sedan 2012',
 'Acura TL Type-S 2008',
 'Acura TSX Sedan 2012',
 'Acura Integra Type R 2001',
 'Acura ZDX Hatchback 2012',
 'Aston Martin V8 Vantage Convertible 2012',
 'Aston Martin V8 Vantage Coupe 2012',
 'Aston Martin Virage Convertible 2012',
 'Aston Martin Virage Coupe 2012',
 'Audi RS 4 Convertible 2008',
 'Audi A5 Coupe 2012',
 'Audi TTS Coupe 2012',
 'Audi R8 Coupe 2012',
 'Audi V8 Sedan 1994',
 'Audi 100 Sedan 1994',
 'Audi 100 Wagon 1994',
 'Audi TT Hatchback 2011',
 'Audi S6 Sedan 2011',
 'Audi S5 Convertible 2012',
 'Audi S5 Coupe 2012',
 'Audi S4 Sedan 2012',
 'Audi S4 Sedan 2007',
 'Audi TT RS Coupe 2012',
 'BMW ActiveHybrid 5 Sedan 2012',
 'BMW 1 Series Convertible 2012',
 'BMW 1 Series Coupe 2012',
 'BMW 3 Series Sedan 2012',
 'BMW 3 Series Wagon 2012',
 'BMW 6 Series Convertible 2007',
 'BMW X5 SUV 2007',
 'BMW X6 SUV 2012',
 'BMW M3 Coupe 2012',
 'BMW M5 Sedan 2010',
 'BMW M6 Convertible 2010',
 'BMW X3 SUV 2012',
 'BMW Z4 Convertible 2012',
 'Bentley Continental Supersports Conv. Convertible 2012',
 'Bentley Arnage Sedan 2009',
 'Bentley Mulsanne Sedan 2011',
 'Bentley Continental GT Coupe 2012',
 'Bentley Continental GT Coupe 2007',
 'Bentley Continental Flying Spur Sedan 2007',
 'Bugatti Veyron 16.4 Convertible 2009',
 'Bugatti Veyron 16.4 Coupe 2009',
 'Buick Regal GS 2012',
 'Buick Rainier SUV 2007',
 'Buick Verano Sedan 2012',
 'Buick Enclave SUV 2012',
 'Cadillac CTS-V Sedan 2012',
 'Cadillac SRX SUV 2012',
 'Cadillac Escalade EXT Crew Cab 2007',
 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
 'Chevrolet Corvette Convertible 2012',
 'Chevrolet Corvette ZR1 2012',
 'Chevrolet Corvette Ron Fellows Edition Z06 2007',
 'Chevrolet Traverse SUV 2012',
 'Chevrolet Camaro Convertible 2012',
 'Chevrolet HHR SS 2010',
 'Chevrolet Impala Sedan 2007',
 'Chevrolet Tahoe Hybrid SUV 2012',
 'Chevrolet Sonic Sedan 2012',
 'Chevrolet Express Cargo Van 2007',
 'Chevrolet Avalanche Crew Cab 2012',
 'Chevrolet Cobalt SS 2010',
 'Chevrolet Malibu Hybrid Sedan 2010',
 'Chevrolet TrailBlazer SS 2009',
 'Chevrolet Silverado 2500HD Regular Cab 2012',
 'Chevrolet Silverado 1500 Classic Extended Cab 2007',
 'Chevrolet Express Van 2007',
 'Chevrolet Monte Carlo Coupe 2007',
 'Chevrolet Malibu Sedan 2007',
 'Chevrolet Silverado 1500 Extended Cab 2012',
 'Chevrolet Silverado 1500 Regular Cab 2012',
 'Chrysler Aspen SUV 2009',
 'Chrysler Sebring Convertible 2010',
 'Chrysler Town and Country Minivan 2012',
 'Chrysler 300 SRT-8 2010',
 'Chrysler Crossfire Convertible 2008',
 'Chrysler PT Cruiser Convertible 2008',
 'Daewoo Nubira Wagon 2002',
 'Dodge Caliber Wagon 2012',
 'Dodge Caliber Wagon 2007',
 'Dodge Caravan Minivan 1997',
 'Dodge Ram Pickup 3500 Crew Cab 2010',
 'Dodge Ram Pickup 3500 Quad Cab 2009',
 'Dodge Sprinter Cargo Van 2009',
 'Dodge Journey SUV 2012',
 'Dodge Dakota Crew Cab 2010',
 'Dodge Dakota Club Cab 2007',
 'Dodge Magnum Wagon 2008',
 'Dodge Challenger SRT8 2011',
 'Dodge Durango SUV 2012',
 'Dodge Durango SUV 2007',
 'Dodge Charger Sedan 2012',
 'Dodge Charger SRT-8 2009',
 'Eagle Talon Hatchback 1998',
 'FIAT 500 Abarth 2012',
 'FIAT 500 Convertible 2012',
 'Ferrari FF Coupe 2012',
 'Ferrari California Convertible 2012',
 'Ferrari 458 Italia Convertible 2012',
 'Ferrari 458 Italia Coupe 2012',
 'Fisker Karma Sedan 2012',
 'Ford F-450 Super Duty Crew Cab 2012',
 'Ford Mustang Convertible 2007',
 'Ford Freestar Minivan 2007',
 'Ford Expedition EL SUV 2009',
 'Ford Edge SUV 2012',
 'Ford Ranger SuperCab 2011',
 'Ford GT Coupe 2006',
 'Ford F-150 Regular Cab 2012',
 'Ford F-150 Regular Cab 2007',
 'Ford Focus Sedan 2007',
 'Ford E-Series Wagon Van 2012',
 'Ford Fiesta Sedan 2012',
 'GMC Terrain SUV 2012',
 'GMC Savana Van 2012',
 'GMC Yukon Hybrid SUV 2012',
 'GMC Acadia SUV 2012',
 'GMC Canyon Extended Cab 2012',
 'Geo Metro Convertible 1993',
 'HUMMER H3T Crew Cab 2010',
 'HUMMER H2 SUT Crew Cab 2009',
 'Honda Odyssey Minivan 2012',
 'Honda Odyssey Minivan 2007',
 'Honda Accord Coupe 2012',
 'Honda Accord Sedan 2012',
 'Hyundai Veloster Hatchback 2012',
 'Hyundai Santa Fe SUV 2012',
 'Hyundai Tucson SUV 2012',
 'Hyundai Veracruz SUV 2012',
 'Hyundai Sonata Hybrid Sedan 2012',
 'Hyundai Elantra Sedan 2007',
 'Hyundai Accent Sedan 2012',
 'Hyundai Genesis Sedan 2012',
 'Hyundai Sonata Sedan 2012',
 'Hyundai Elantra Touring Hatchback 2012',
 'Hyundai Azera Sedan 2012',
 'Infiniti G Coupe IPL 2012',
 'Infiniti QX56 SUV 2011',
 'Isuzu Ascender SUV 2008',
 'Jaguar XK XKR 2012',
 'Jeep Patriot SUV 2012',
 'Jeep Wrangler SUV 2012',
 'Jeep Liberty SUV 2012',
 'Jeep Grand Cherokee SUV 2012',
 'Jeep Compass SUV 2012',
 'Lamborghini Reventon Coupe 2008',
 'Lamborghini Aventador Coupe 2012',
 'Lamborghini Gallardo LP 570-4 Superleggera 2012',
 'Lamborghini Diablo Coupe 2001',
 'Land Rover Range Rover SUV 2012',
 'Land Rover LR2 SUV 2012',
 'Lincoln Town Car Sedan 2011',
 'MINI Cooper Roadster Convertible 2012',
 'Maybach Landaulet Convertible 2012',
 'Mazda Tribute SUV 2011',
 'McLaren MP4-12C Coupe 2012',
 'Mercedes-Benz 300-Class Convertible 1993',
 'Mercedes-Benz C-Class Sedan 2012',
 'Mercedes-Benz SL-Class Coupe 2009',
 'Mercedes-Benz E-Class Sedan 2012',
 'Mercedes-Benz S-Class Sedan 2012',
 'Mercedes-Benz Sprinter Van 2012',
 'Mitsubishi Lancer Sedan 2012',
 'Nissan Leaf Hatchback 2012',
 'Nissan NV Passenger Van 2012',
 'Nissan Juke Hatchback 2012',
 'Nissan 240SX Coupe 1998',
 'Plymouth Neon Coupe 1999',
 'Porsche Panamera Sedan 2012',
 'Ram C/V Cargo Van Minivan 2012',
 'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
 'Rolls-Royce Ghost Sedan 2012',
 'Rolls-Royce Phantom Sedan 2012',
 'Scion xD Hatchback 2012',
 'Spyker C8 Convertible 2009',
 'Spyker C8 Coupe 2009',
 'Suzuki Aerio Sedan 2007',
 'Suzuki Kizashi Sedan 2012',
 'Suzuki SX4 Hatchback 2012',
 'Suzuki SX4 Sedan 2012',
 'Tesla Model S Sedan 2012',
 'Toyota Sequoia SUV 2012',
 'Toyota Camry Sedan 2012',
 'Toyota Corolla Sedan 2012',
 'Toyota 4Runner SUV 2012',
 'Volkswagen Golf Hatchback 2012',
 'Volkswagen Golf Hatchback 1991',
 'Volkswagen Beetle Hatchback 2012',
 'Volvo C30 Hatchback 2012',
 'Volvo 240 Sedan 1993',
 'Volvo XC90 SUV 2007',
 'smart fortwo Convertible 2012'])


##################################################################################################


def show_resize_and_save_jpg(original_jpg_path, final_png_path):
    """
    Function:
        -converts uploaded jpeg to matplotlib-rendered png image
        -saves the png image in final_png_path
    Arguments:
        -original_jpg_path (str): the path to the jpeg image
        -final_png_path (str): path to the rendered png image
    """
    original_image_bgr = cv2.imread(original_jpg_path)
    # change the channel ordering
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

    resized_image_np = cv2.resize(original_image_rgb, (224, 224))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_image_rgb, interpolation='nearest')
    ax.set_title("Your original image:")
    fig.savefig(final_png_path)



def show_and_save_numpy(image_np_bb, image_np_cropped, final_png_path):
    """
    Function:
        -converts uploaded jpeg to matplotlib-rendered png image
        -saves the png image in final_png_path
    Arguments:
        -original_jpg_path (str): the path to the jpeg image
        -final_png_path (str): path to the rendered png image
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.imshow(image_np_bb, interpolation='nearest')
    ax1.set_title("Your image with bounding box added:")
    ax2.imshow(image_np_cropped, interpolation='nearest')
    ax2.set_title("Content of the bounding box to be inputted into the classifier:")
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(final_png_path)



def save_file(target_path, content=None):
    """
    Function: decodes and stores a file uploaded with Plotly Dash.
    """
    data = content.encode("utf8").split(b";base64,")[1]
    with open(target_path, "wb") as fp:
        fp.write(base64.decodebytes(data))


def encode_image(image_file):
    """
    Function: encodes a png image
    """
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())



###########################################################################################################

app.layout = html.Div(children=[
    html.H1(children="Autoclassifier", style={'textAlign': 'center'}),
    html.H2(children="Please upload an image of a car in JPEG format:", style={'textAlign': 'center'}),
    dcc.Upload(id='upload-image',
               children=[html.Button('Upload image')],
               style={'textAlign': 'center'}),           # Step 1: - Input: Upload image
    # - Function: Display image
    html.Div(id='img-1-div', style={'textAlign': 'center'}),
    html.Div(id='img-2-div', style={'textAlign': 'center'}),
    html.Div(id='prediction-div', style={'textAlign': 'center'}),
    html.P(),
    html.P(),
    html.Div(id='empty', style={'textAlign': 'center'})   # Output: predictions
])

###########################################################################################################

@app.callback(Output('img-1-div', 'children'),
              [Input('upload-image', 'contents')])
def display_image_1(contents):
    if contents is not None:
        print("Image uploaded, second callback function is activated")

        # Save it as jpeg
        jpg_path = "assets/image.jpg"
        save_file(target_path=jpg_path, content=contents)

        # Convert it to png and display it
        png_path = "assets/image.png"
        show_resize_and_save_jpg(original_jpg_path=jpg_path, final_png_path=png_path)
        print("First callback function is triggered")
        return [html.Img(src=encode_image(png_path)),
                html.Div(children = [html.P(
                                            id='notice-1',
                                            children='Localizing car, please wait (might take up to 10 seconds)...',
                                            style={'textAlign': 'center'}
                                            )
                                    ],
                         style={'textAlign': 'center'}
                         )
                ]

@app.callback(Output('img-2-div', 'children'),
              [Input('notice-1', 'children')],
              [State('upload-image', 'contents')])
def display_image_2(notice, contents):

    print("image uploaded, third callback function is activated")
    # Save it as jpeg
    jpg_path = "assets/image.jpg"
    #save_file(target_file_path=jpg_path, content=contents)

    image_np_bgr = cv.imread(jpg_path)
    image_np_rgb = cv.cvtColor(image_np_bgr, cv.COLOR_BGR2RGB)

    best_bounding_box_coordinates, best_bounding_box_probability = predict_bounding_box(image_np_rgb, detection_graph)
    print(best_bounding_box_probability)
    if best_bounding_box_probability is not None:
        best_bounding_box_coordinates_str = ",".join([str(coor) for coor in best_bounding_box_coordinates])

        image_np_with_bb = insert_bounding_box(image_np_rgb.copy(), best_bounding_box_coordinates)
        image_np_cropped = crop_image_with_bounding_box(image_np_rgb, best_bounding_box_coordinates)

        # Convert it to png and display it
        png_path = "assets/image_processed.png"
        show_and_save_numpy(image_np_with_bb, image_np_cropped, final_png_path=png_path)
        return [html.Img(id = "image-2",
                         src = encode_image(png_path)),
                html.Div(id = "hidden-contain-bb", children = best_bounding_box_coordinates_str, style = {'display': 'none'}),
                html.Div(children = [html.P(
                                            id='notice-2',
                                            children='Predicting car type, please wait (might take up to 10 seconds)...',
                                            style={'textAlign': 'center'}
                                            )
                                    ],
                         style={'textAlign': 'center'}
                         )
                ]

    else:
        return [html.Div(id = "hidden-contain-bb", children = [html.H1("Our model didn't detect a car in your picture, please upload another one!")])]


@app.callback(Output('prediction-div', 'children'),
              [Input('hidden-contain-bb', 'children')],
              [State('upload-image', 'contents')])
def display_prediction(coordinates,  _):

    if isinstance(coordinates,str):
        coordinates_list = [float(coor) for coor in coordinates.split(",")]
        print("Fourth callback function is triggered!")
        # Save it as jpeg
        jpg_path = "assets/image.jpg"
        #save_file(target_file_path=jpg_path, content=contents)
        print(coordinates_list)
        (ymin, xmin, ymax, xmax) = coordinates_list


        jpg_path = "assets/image.jpg"
        #save_file(target_file_path=jpg_path, content=contents)

        image_np_bgr = cv.imread(jpg_path)
        image_np_rgb = cv.cvtColor(image_np_bgr, cv.COLOR_BGR2RGB)

        im_height, im_width, _ = image_np_rgb.shape

        (xmin,xmax,ymin,ymax) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))

        image_np_truncated = image_np_rgb[ymin:ymax, xmin:xmax]

        image_np_resized = cv.resize(image_np_truncated, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        image_np_expanded = np.expand_dims(image_np_resized, 0)

        probabilities = model.predict(image_np_expanded)

        top_3_index = np.argsort(-probabilities)[0,:3]

        top_3_probabilities = probabilities[0,top_3_index]

        top_3_classes = classnames[top_3_index]


        return [html.P("Here is the top three predictions from the model:"),
                html.Div(children = [
                                    dcc.Graph(figure=go.Figure(
                                                                data = [go.Bar(x = top_3_classes,
                                                                               y = top_3_probabilities)],
                                                                layout = go.Layout(xaxis = dict(title = 'Class',
                                                                                              tickfont  =dict(size = 15,
                                                                                                            color = 'black')),
                                                                                 yaxis={'title': 'Probability'},
                                                                                 autosize=True,
                                                                                 width=600,
                                                                                 height=300
                                                                                 )

                                              ))
                                    ], style={'textAlign': 'center', "align-items": "center", 'display': 'inline-block', 'margin': 'auto'}),
                                ]
    ####
    else:
        return [html.Div(id = "hidden-contain-bb", children = [html.H1("Our model didn't detect a car in your picture, please upload another one!")])]


if __name__ == '__main__':
    app.run_server()
