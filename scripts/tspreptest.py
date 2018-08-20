import numpy as np
import gdal
import glob
import os.path
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsprep import sRead

# path on my local drive
path="/Users/adbe/mmann/Demo_Data/temperature/"
#read

#reader = tsprep.sRead()
my_df = sRead.ts_series(path)

#check
print(my_df.head(25))
#
# # #parallel-process
#
# Distributor = MultiprocessingDistributor(n_workers=16,
#                                          disable_progressbar=False,
#                                          progressbar_title="Feature Extraction")
#
# #calculate mean
#
# fc_parameters = {
#     "mean": None
# }
#
# extracted_features = extract_features(my_df,
#                                       default_fc_parameters = fc_parameters,
#                                       column_sort = "time",
#                                       column_value ="value",
#                                       column_id="id",
#                                       distributor=Distributor)
#
# extracted_features.describe()

def calculateFeatures(self):
    my_df = sRead.ts_series(self)
    Distributor = MultiprocessingDistributor(n_workers=16,
                                             disable_progressbar=False,
                                             progressbar_title="Feature Extraction")
    # calculate mean
    fc_parameters = {
        "mean": None
    }

    extracted_features = extract_features(my_df,
                                          default_fc_parameters=fc_parameters,
                                          column_sort="time",
                                          column_value="value",
                                          column_id="id",
                                          distributor=Distributor)
    return extracted_features

def features2array(self):
    raw_data = sRead.image(path)
    rows = raw_data[0].RasterXSize
    cols = raw_data[0].RasterYSize
    # read tsfresh output, drop index reshape as numpy array
    df_features = my_df.drop(my_df.columns[1], axis=1)
    num_of_layers = dropIndex.shape[1]
    matrix_features = df_features.values
    f2array = matrix_features.reshape(cols, rows, num_of_layers)
    print(f2array.shape)



#features2array(path)
#
# def array2raster(output_file, input_data, geo_transform, projection, data_type=gdal.GDT_Byte):
#
#     # Read meta data from the first raster layer
#
#     raw_data = sRead.image(path)
#     geo_transform = raw_data[0].GetGeoTransform()
#     projection = raw_data[0].GetProjection()
#     rows = raw_data[0].RasterXSize
#     cols = raw_data[0].RasterYSize
#
#     self.output_file = output_file
#     self.input_data = features2array()
#
#     driver = gdal.GetDriverByName('GTiff')
#
#     features = driver.Create(output_file, rows, cols, bands, gdal.GDT_Float32)
#     features.SetGeoTransform(geo_transform)
#     features.SetProjection(projection)
#
#
#  output_file = r'features.tiff'