'''
calculate.py: a module for extracting, evaluating and saving features
'''


import numpy as np
import pandas as pd
import os
import gdal
from pathlib import Path
from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_selection.relevance import calculate_relevance_table as crt
from tsraster.prep import image_to_series, image_to_array, read_images


def CreateTiff(Name, Array, driver, NDV, GeoT, Proj, DataType, path):
    '''
    Converts array to a single or multi band raster file

    :param Name: name of the output tiff file
    :param Array: numpy array to be converted to
    :param driver: output image (data) format
    :param NDV: no Data Value (-9999)
    :param GeoT: geographic transformation
    :param Proj: projection
    :param DataType: array data format
    :return: GeoTiff
    '''

    Array[np.isnan(Array)] = NDV

    rows = Array.shape[1]
    cols = Array.shape[0]
    band = Array.shape[2]
    noData = -9999
    driver = gdal.GetDriverByName('GTiff')
    Name_out = os.path.join(path,Name)
    print('tif:'+ Name_out)
    DataSet = driver.Create(Name_out, rows, cols, band, gdal.GDT_Float32)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Proj)

    for i in range(band):
        DataSet.GetRasterBand(i + 1).WriteArray(Array[:, :, i])
        DataSet.GetRasterBand(i + 1).SetNoDataValue(noData)

    DataSet.FlushCache()
    return Name


def calculateFeatures(path, parameters, reset_df, tiff_output=True):
    '''
    Calculates features or the statistical characteristics of time-series raster data.
    It can also save features as a csv file (dataframe) and/or tiff file.

    :param path: directory path to the raster files
    :param parameters: a dictionary of features to be extracted
    :param reset_df: boolean option for existing raster inputs as dataframe
    :param tiff_output: boolean option for exporting tiff file
    :return: extracted features as a dataframe and tiff file
    '''
  
    if reset_df == False:
        #if reset_df =F read in csv file holding saved version of my_df
    	    my_df = pd.read_csv(os.path.join(path,'my_df.csv'))
    else:
        #if reset_df =T calculate ts_series and save csv
        my_df = image_to_series(path)
        print('df: '+os.path.join(path,'my_df.csv'))
        my_df.to_csv(os.path.join(path,'my_df.csv'), chunksize=10000, index=False)
    
    Distributor = MultiprocessingDistributor(n_workers=6,
                                             disable_progressbar=False,
                                             progressbar_title="Feature Extraction")
    
    extracted_features = extract_features(my_df,
                                          default_fc_parameters=parameters,
                                          column_sort="time",
                                          column_value="value",
                                          column_id="id",
                                          distributor=Distributor)
    
    # deal with output location 
    out_path = Path(path).parent.joinpath(Path(path).stem+"_features")
    out_path.mkdir(parents=True, exist_ok=True)
    
    # get file prefix
    prefix = [f for f in os.listdir(path) if f.endswith('.tif')][0][0:4]
    
    # write out features to csv file
    print("features:"+os.path.join(out_path,'extracted_features.csv'))
    extracted_features.columns = [prefix + str(col) for col in extracted_features.columns]
    extracted_features.to_csv(os.path.join(out_path,'extracted_features.csv'), chunksize=10000)
    
    # write data frame
    kr = pd.DataFrame(list(extracted_features.columns))
    kr.index += 1
    kr.index.names = ['band']
    kr.columns = ['feature_name']
    kr.to_csv(os.path.join(out_path,"features_names.csv"))
    
    # write out features to tiff file
    if tiff_output == False:
    
        '''tiff_output is true and by default exports tiff '''
    
        return extracted_features
    else:
        # get image dimension from raw data
        rows, cols, num = image_to_array(path).shape
        # get the total number of features extracted
        matrix_features = extracted_features.values
        num_of_layers = matrix_features.shape[1]
        
        #reshape the dimension of features extracted
        f2Array = matrix_features.reshape(rows, cols, num_of_layers)
        output_file = 'extracted_features.tiff'  
        
        #Get Meta Data from raw data
        raw_data = read_images(path)
        GeoTransform = raw_data[0].GetGeoTransform()
        driver = gdal.GetDriverByName('GTiff')
        
        noData = -9999
        
        Projection = raw_data[0].GetProjectionRef()
        DataType = gdal.GDT_Float32
        
        #export tiff
        CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType, path=out_path)
        return extracted_features


def features_to_array(path, input_file):
    '''
     Converts a dataframe to array

    :param path:  directory path to the raster files
    :param input_file: features in dataframe
    :return: array with height and width similar to the input rasters
    '''

    rows, cols, num = image_to_array(path).shape
    my_df = pd.read_csv(input_file)


    #df_features = my_df.drop(my_df.columns[0], axis=1)
    matrix_features = my_df.values
    num_of_layers = matrix_features.shape[1]

    f2Array = matrix_features.reshape(rows, cols, num_of_layers)

    return f2Array


def exportFeatures(path, input_file, output_file,
                    driver = gdal.GetDriverByName('GTiff'),
                    noData = -9999, DataType = gdal.GDT_Float32):

   '''
   Saves features stored in a data frame as a mulit-band tiff file

   :param path: directory path to the raster files
   :param input_file: the features stored in pandas data frame
   :param output_file: the name of the output_file
   :param driver: data format of the output file
   :param noData: no data value
   :param DataType: array data format
   :return: tiff file of the exported features
   '''
   output_file = output_file
   raw_data = read_images(path)
   geoTransform = raw_data[0].GetGeoTransform()
   projection = raw_data[0].GetProjectionRef()
   f2Array = features_to_array(path, input_file)
   export_features = CreateTiff(output_file, f2Array, driver, noData, geoTransform, projection, DataType)

   return export_features




def checkRelevance(x, y, ml_task="auto", fdr_level=0.05):
    '''
    Checks the statistical relevance of features to the target data

    :param x: pandas dataframe containing the features extracted
    :param y: pandas series
    :return: dataframe
    '''

    # read files
    features = x
    target = y

    # drop id column
    features = features.drop(labels="id", axis=1)

    # calculate relevance
    relevance_test = crt(features,
                         target,
                         ml_task=ml_task,
                         fdr_level=fdr_level)

    return relevance_test

def checkRelevance2(x, y, ml_task="auto", fdr_level=0.05):
        '''
        :Checks the statistical significance of features selects only significant ones

        :param x: pandas dataframe containing the features extracted
        :param y: pandas series
        :return: dataframe
        '''
        # read files
        features = x
        target = y

        # drop id column
        features = features.drop(labels="id", axis=1)

        # calculate relevance
        relevance_test = crt(features,
                             target,
                             ml_task=ml_task,
                             fdr_level=fdr_level)

        # gather subset of relevant features
        relevant_feature_names = relevance_test.feature[relevance_test.relevant==True]
        relevant_features = features[relevant_feature_names]

        return relevance_test, relevant_features