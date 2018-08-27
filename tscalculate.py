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



path = "/Users/adbe/mmann/Demo_Data/temperature/"
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
        "maximum": None,
        "median":None,
        "minimum":None
    }

    extracted_features = extract_features(my_df,
                                          default_fc_parameters=fc_parameters,
                                          column_sort="time",
                                          column_value="value",
                                          column_id="id",
                                          distributor=Distributor)

    print(extracted_features.head())
    kr = pd.DataFrame(list(extracted_features.columns))
    kr.index += 1
    kr.to_csv("features_names.csv")
    return extracted_features


def features2array(self):
    '''
    features2array converts dataframe containing extracted features to array
    :param self: - collects meta data from the first image in the dataset
                        : meta data: rows, cols

    :return: numpy array
    '''
    rows, cols, num = sRead.image2array(path).shape


    my_df = calculateFeatures(self)  # Calculate Features

    '''convert dataframe to array currently supported but gives incorrect output'''



    #df_features = my_df.drop(my_df.columns[0], axis=1)
    print(my_df.shape)

    matrix_features = my_df.values
    num_of_layers = matrix_features.shape[1]
    print(matrix_features.shape[1])

    f2Array = matrix_features.reshape(rows, cols, num_of_layers)
    print(f2Array.shape)

    return f2Array



def CreateTiff(Name, Array, driver, NDV, GeoT, Proj, DataType):
    Array[np.isnan(Array)] = NDV

    rows = Array.shape[1]
    cols = Array.shape[0]
    band = Array.shape[2]

    driver = gdal.GetDriverByName('GTiff')

    DataSet = driver.Create(Name, rows, cols, band, gdal.GDT_Float32)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Proj)

    for i in range(band):
        DataSet.GetRasterBand(i + 1).WriteArray(Array[:, :, i])
        DataSet.GetRasterBand(i + 1).SetNoDataValue(noData)

    DataSet.FlushCache()
    return Name


def main(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType):
    # reverse array so the tif looks like the array
    CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType)  # convert array to raster


if __name__ == "__main__":

    output_file = "Temprature_Features.tiff"

    raw_data = sRead.image(path)
    f2Array = features2array(path)

    GeoTransform = raw_data[0].GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')

    noData = -9999

    Projection = raw_data[0].GetProjectionRef()
    DataType = gdal.GDT_Float32


    main(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType)
