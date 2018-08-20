#######################################################################################
# tsprep.py : reads rasters files from multiple folders,
#              extract custom features and write values to raster
# Aug-20-2018
# author: Adane(Eddie) Bedada
# @adbe.gwu.edu
#######################################################################################


import numpy as np
import gdal
import glob
import os.path
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsprep import sRead


# If correctly provided, the the 5 rows of the data will be printed
my_df = sRead.ts_series(path)
print(my_df.head())
#

#
def calculateFeatures(self):
    '''
    calculateFeatures literally calculate features

    :param self: reads the dataframe created with ts_series
    Distributor is a tsfresh feature for parallel processing
    fc_parameters is a dictionary containin the features to be extracted
    :return: a dataframe with features
    '''

    my_df = sRead.ts_series(self)
    Distributor = MultiprocessingDistributor(n_workers=16,
                                             disable_progressbar=False,
                                             progressbar_title="Feature Extraction")


    fc_parameters = {
        "mean": None,
        "maximum": None
    }

    extracted_features = extract_features(my_df,
                                          default_fc_parameters=fc_parameters,
                                          column_sort="time",
                                          column_value="value",
                                          column_id="id",
                                          distributor=Distributor)
    return extracted_features


def features2array(self):
    '''
    features2array converts dataframe containing extracted features to array
    :param self: - collects meta data from the first image in the dataset
                        : meta data: rows, cols

    :return: numpy array
    '''


    raw_data = sRead.image(path)
    rows = raw_data[0].RasterXSize
    cols = raw_data[0].RasterYSize


    my_features = calculateFeatures(self)  # Calculate Features

    '''convert dataframe to array currently supported but gives incorrect output'''
    # df_features = my_features.drop(my_features.columns[1], axis=1) # drop row index
    # num_of_layers = df_features.shape[1]
    # matrix_features = df_features.values
    # f2array = matrix_features.reshape(cols, rows, num_of_layers)

    '''convert dataframe to array: depreciating but gives correct output'''

    feature_matrix = my_features.as_matrix(columns=my_features.columns[1:])
    ba = feature_matrix.shape[1]
    f2Array = feature_matrix.reshape(rows, cols, ba)
    return f2Array




def CreateTiff(Name, Array, driver, NDV, GeoT, Proj, DataType):
    '''
    CreateTiff stores each extracted feature as a band

    :param Name: Name out the output file
    :param Array: Array created by the script
    :param driver: Data format
    :param NDV:  No Data Value
    :param GeoT: Transform
    :param Proj: Projection
    :param DataType: Float ?
    :return: Raster file
    '''



    Array[np.isnan(Array)] = NDV

    rows = Array.shape[1]
    cols = Array.shape[0]
    band = Array.shape[2]

    driver = gdal.GetDriverByName('GTiff')

    DataSet = driver.Create(Name, rows, cols, band, gdal.GDT_Float32)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Proj)

    for i, image in range(Array.shape[2], 1):
        DataSet.GetRasterBand(i).WriteArray(image)
        DataSet.GetRasterBand(i).SetNoDataValue(NDV)

    DataSet.FlushCache()
    return Name


'''Two information that will be required from the user is 
the path to folders containing the rasters and output file name
'''

path="/Users/adbe/mmann/Demo_Data/temperature/"
output_file = r'TempFeatures.tiff'



# Feature Extractions begins here
f2Array = features2array(path)


# Get Meta Data from the raw data
raw_data = sRead.image(path)
GeoTransform = raw_data[0].GetGeoTransform()
driver = gdal.GetDriverByName('GTiff')

noData = f2Array[np.isnan(f2Array)]
Projection = raw_data[0].GetProjectionRef()
DataType = gdal.GDT_Byte


# Writes raster
CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType)


