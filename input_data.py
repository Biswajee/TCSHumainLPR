# import libraries
import urllib
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

Images = []
Plates = []


'''
readJSON : read the JSON content from the file and parse necessary information
'''
def readJSON():
    # Read the data
    data = pd.read_json('Indian_Number_plates.json', lines=True)
    pd.set_option('display.max_colwidth', -1)

    # Delete the empty column
    del data['extras']

    # Extract the points of the bounding boxes because thats what we want
    data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)

    # And drop the rest of the annotation info
    del data['annotation']
    return data


'''
downloadTraining : download the training data for the problem statement and return list of number plates
'''
def downloadTraining(df):

    for index, row in df.iterrows():

        # Get the image from the URL
        resp = urllib.request.urlopen(row[0])
        im = np.array(Image.open(resp))

        # We append the image to the training input array
        Images.append(im)  

        # Points of rectangle
        x_point_top = row[1][0]['x']*im.shape[1]
        y_point_top = row[1][0]['y']*im.shape[0]
        x_point_bot = row[1][1]['x']*im.shape[1]
        y_point_bot = row[1][1]['y']*im.shape[0]

        # Cut the plate from the image and use it as output
        carImage = Image.fromarray(im)
        plateImage = carImage.crop((x_point_top, y_point_top, x_point_bot, y_point_bot))
        Plates.append(np.array(plateImage))
    return Plates