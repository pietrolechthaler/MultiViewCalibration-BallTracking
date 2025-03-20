"""
Author: Pietro Lechthaler
Description: 
"""
import datetime
import json
import cv2
import numpy as np
import os
import copy
import utils.parameters as parameters
import sys


def read_chessboard_dimensions(folder):
    """
    Read the chessboard dimensions from a JSON file.
    @params:
        json_path (str): Path to the folder containing the json with chessboard dimensions.
    @return:
        tuple: Tuple containing the dimensions of the chessboard (rows,, columns)
    """
    # Load the JSON file
    filename = parameters.JSON_CHESSBOARD
    json_path = os.path.join(folder, filename)

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        
    rows = data['chessboard_dimensions']['rows']
    columns = data['chessboard_dimensions']['columns']

    return (rows, columns)