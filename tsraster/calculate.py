'''
calculate.py: a module for extracting, evaluating and saving features
'''


import numpy as np
import pandas as pd
import rasterio
import os
import gdal
import glob
from pathlib import Path
from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor, LocalDaskDistributor
from tsfresh.feature_selection.relevance import calculate_relevance_table as crt
from tsraster.prep import image_to_series, image_to_array, read_images, image_to_series_window, image_to_array_window, read_images_window
import tsraster.prep  as tr
#from tsfresh.utilities.distribution import LocalDaskDistributor


def CreateTiff(Name, Array, driver, NDV, GeoT, Proj, DataType, path):
    '''
    Converts array to a single or multi band raster file in GeoTiff format

    :param Name: name of the output tiff file
    :param Array: numpy array to be converted to
    :param driver: output image (data) format
    :param NDV: no Data Value (-9999)
    :param GeoT: geographic transformation
    :param Proj: projection
    :param DataType: array data format
    :param path: file directory
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


def calculateFeatures(path, parameters, reset_df,raster_mask=None ,tiff_output=True,missing_value = -9999, workers = None):
    '''
    Calculates features or the statistical characteristics of time-series raster data.
    It can also save features as a csv file (dataframe) and/or tiff file.
    
    :param path: directory path to the raster files
    :param parameters: a dictionary of features to be extracted
    :param reset_df: boolean option for existing raster inputs as dataframe
    :param raster_mask: path to binary raster mask (default None)
    :param tiff_output: boolean option for exporting tiff file (default True)
    :param workers: number of parallel workers in multiprocessing pool (default None)
    :return: extracted features as a dataframe and tiff file
    '''
    
    if reset_df == False:
        #if reset_df =F read in csv file holding saved version of my_df
        my_df = tr.read_my_df(path)
            
    else:
        #if reset_df =T calculate ts_series and save csv
        my_df = image_to_series(path)
        print('df: '+os.path.join(path,'my_df.csv'))
        my_df.to_csv(os.path.join(path,'my_df.csv'), chunksize=10000, index=False)
    
    # mask rasters based on desired mask, if present
    if raster_mask is not None:
        my_df = tr.mask_df(raster_mask = raster_mask, 
                        original_df = my_df)
    
    #distribute processing across multiprocessing pool, if multiple workers are present
    if workers is not None:
        Distributor = MultiprocessingDistributor(n_workers=workers,
                                                 disable_progressbar=False,
                                                 progressbar_title="Feature Extraction")
        #Distributor = LocalDaskDistributor(n_workers=workers)
    else:
        Distributor = None
    
    extracted_features = extract_features(my_df, 
                                          default_fc_parameters = parameters,
                                          column_sort = "time",
                                          column_value = "value",
                                          column_id = "pixel_id",
                                          column_kind="kind", 
                                          #chunksize = 1000,
                                          distributor=Distributor
                                          )
    
    # change index name to match pixel and time period
    extracted_features.index.rename('pixel_id',inplace=True)
    extracted_features.reset_index(inplace=True, level=['pixel_id'])
    
    extracted_features['time'] = str(my_df.time.min())+"_"+str(my_df.time.max())
    extracted_features.set_index(['pixel_id', 'time'], inplace=True) 
     
    # unmask extracted features
    extracted_features = tr.unmask_from_mask(mask_df_output = extracted_features, 
                                          missing_value = -missing_value,
                                          raster_mask = raster_mask)
    
    # deal with output location 
    out_path = Path(path).parent.joinpath(Path(path).stem+"_features")
    out_path.mkdir(parents=True, exist_ok=True)
    
    # write out features to csv file
    print("features:"+os.path.join(out_path,'extracted_features.csv'))
    extracted_features.to_csv(os.path.join(out_path,'extracted_features.csv'), chunksize=10000)
    
    # write out feature names 
    kr = pd.DataFrame(list(extracted_features.columns))
    kr.index += 1
    kr.index.names = ['band']
    kr.columns = ['feature_name']
    kr.to_csv(os.path.join(out_path,"features_names.csv"))
    
    # write out features to tiff file
    if tiff_output == False:
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


def calculateFeatures_window(path, parameters, baseYear, reset_df = True,length = 3, offset = 0,raster_mask=None ,tiff_output=True, workers = None, outPath= "None"):
    
    '''
    Calculates features or the statistical characteristics of time-series raster data.
    It can also save features as a csv file (dataframe) and/or tiff file.
    
    :param path: directory path to the raster files
    :param parameters: a dictionary of features to be extracted
    :param length: number of prior years to evaluate (default 3)
    :param startYear: first year of interest
    :param lastYear: last year of interest
    :param offset: number of years by which to offset parameters from year of interest (default 1)
    :param reset_df: boolean option for existing raster inputs as dataframe
    :param raster_mask: path to binary raster mask (default None)
    :param tiff_output: boolean option for exporting tiff file (default True)
    :param workers: number of parallel workers in multiprocessing pool (default None)
    :return: extracted features as a dataframe and tiff file
    '''
    
  
    if reset_df == False:
        #if reset_df =F read in csv file holding saved version of my_df
        my_df = tr.read_my_df(path)
            
    else:
        #if reset_df =T calculate ts_series and save csv
        my_df = image_to_series_window(path, baseYear, length, offset)
        print('df: '+os.path.join(path,'my_df.csv'))
        my_df.to_csv(os.path.join(path,'my_df.csv'), chunksize=10000, index=False)
    
    # mask rasters based on desired mask, if present
    if raster_mask is not None:
        my_df = tr.mask_df(raster_mask = raster_mask, 
                        original_df = my_df)
    
    #distribute processing across multiprocessing pool, if multiple workers are present
    if workers is not None:
        Distributor = MultiprocessingDistributor(n_workers=workers,
                                                 disable_progressbar=False,
                                                 progressbar_title="Feature Extraction")
        #Distributor = LocalDaskDistributor(n_workers=workers)
    else:
        Distributor = None
    
    extracted_features = extract_features(my_df, 
                                          default_fc_parameters = parameters,
                                          column_sort = "time",
                                          column_value = "value",
                                          column_id = "pixel_id",
                                          column_kind="kind", 
                                          #chunksize = 1000,
                                          distributor=Distributor
                                          )
    
    # change index name to match pixel and time period
    extracted_features.index.rename('pixel_id',inplace=True)
    extracted_features.reset_index(inplace=True, level=['pixel_id'])
    
    extracted_features['time'] = str(my_df.time.min())+"_"+str(my_df.time.max())
    extracted_features.set_index(['pixel_id', 'time'], inplace=True) 
    
    # unmask extracted features
    extracted_features = tr.unmask_from_mask(mask_df_output = extracted_features, 
                                          missing_value = -9999,
                                          raster_mask = raster_mask)
    
    
    if length>0:
      baseName = str(baseYear + offset) + '_' + str(baseYear + length+ offset)
    elif length<0: 
      baseName = str(baseYear + length+ offset) + '_' + str(baseYear + offset)

    # deal with output location 
    if outPath == "None":
      out_path = Path(path).parent.joinpath(Path(path).stem+"_features")
      out_path.mkdir(parents=True, exist_ok=True)
      
      # write out features to csv file
      print("features:"+os.path.join(out_path,'extracted_features' + baseName +  '.csv'))
      extracted_features.to_csv(os.path.join(out_path,'extracted_features' + baseName +  '.csv'), chunksize=10000)
       # write out feature names 
      kr = pd.DataFrame(list(extracted_features.columns))
      kr.index += 1
      kr.index.names = ['band']
      kr.columns = ['feature_name']
      kr.to_csv(os.path.join(out_path,"features_names" + baseName +  ".csv"))
    elif outPath != "None":
      print("features:"+os.path.join(outPath,'extracted_features' + baseName +  '.csv'))
      extracted_features.to_csv(os.path.join(outPath,'extracted_features' + baseName +  '.csv'), chunksize=10000)
      kr = pd.DataFrame(list(extracted_features.columns))
      kr.index += 1
      kr.index.names = ['band']
      kr.columns = ['feature_name']
      kr.to_csv(os.path.join(outPath,"features_names" + baseName +  ".csv"))
   
    
    # write out features to tiff file
    if tiff_output == False:
        return extracted_features
    else:
        # get image dimension from raw data
        rows, cols, num = image_to_array(path).shape
        # get the total number of features extracted
        matrix_features = extracted_features.values
        num_of_layers = matrix_features.shape[1]
        
        #reshape the dimension of features extracted
        f2Array = matrix_features.reshape(rows, cols, num_of_layers)
        output_file = 'extracted_features'+ baseName +  '.tiff'  
        
        #Get Meta Data from raw data
        raw_data = read_images(path)
        GeoTransform = raw_data[0].GetGeoTransform()
        driver = gdal.GetDriverByName('GTiff')
        
        noData = -9999
        
        Projection = raw_data[0].GetProjectionRef()
        DataType = gdal.GDT_Float32
        
        #export tiff
        if outPath == "None":
          CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType, path=out_path)
        elif outPath != "None":
          CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType, path=outPath)
        return extracted_features

#def calculateFeatures2(path, parameters, mask=None, reset_df=True, tiff_output=True, 
#                           missing_value =-9999,workers=2):
#    '''
#    Calculates features or the statistical characteristics of time-series raster data.
#    It can also save features as a csv file (dataframe) and/or tiff file.
#    
#    :param path: directory path to the raster files
#    :param parameters: a dictionary of features to be extracted
#    :param reset_df: boolean option for existing raster inputs as dataframe
#    :param tiff_output: boolean option for exporting tiff file
#    :return: extracted features as a dataframe and tiff file
#    '''
#      
#    if reset_df == False:
#        #if reset_df =F read in csv file holding saved version of my_df
#        df_long = pd.read_csv(os.path.join(path,'df_long.csv'))
#        
#        # create example of original df to help unmask 
#        df_original = pd.read_csv(os.path.join(path,'df_original.csv') )
#        df_original = pd.DataFrame(index = pd.RangeIndex(start=0,
#                                                         stop=len(df_original),
#                                                         step=1), 
#                                             dtype=np.float32)
#        
#        # set index name to pixel id 
#        df_original.index.names = ['pixel_id']
#        
#    else:
#        #if reset_df =T calculate ts_series and save csv
#        df_long, df_original   = image_to_series2(path, 
#                                                  mask)
#        
#        print('df: '+os.path.join(path,'df_long.csv'))
#        df_long.to_csv(os.path.join(path,'df_long.csv'), 
#                     chunksize=10000, 
#                     index=False)
#    
#        df_original.to_csv(os.path.join(path,'df_original.csv'), 
#                     chunksize=10000, 
#                     index=True)
#    
#    # remove missing values from df_long
#    df_long = df_long[df_long['value'] != missing_value]
#    
#    # check if the number of observation per pixel are not identical
#    if ~df_long.groupby(['pixel_id','kind']).kind.count().all():
#        print('ERROR: the number of observation per pixel are not identical')
#        print('       fix missing values to have a uniform time series')
#        print(df_long.groupby(['time']).time.unique())
#        
#        return(df_long.groupby(['pixel_id','kind']).kind.count().all())
#     
#        
#    Distributor = MultiprocessingDistributor(n_workers=workers,
#                                             disable_progressbar=False,
#                                             progressbar_title="Feature Extraction")
#    #Distributor = LocalDaskDistributor(n_workers=2)
#    
#    extracted_features = extract_features(df_long,
#                                          #chunksize=10e6,
#                                          default_fc_parameters=parameters,
#                                          column_id="pixel_id", 
#                                          column_sort="time", 
#                                          column_kind="kind", 
#                                          column_value="value",
#                                          distributor=Distributor
#                                          )
#    
#    # extracted_features.index is == df_long.pixel_id
#    extracted_features.index.name= 'pixel_id'
#    
#    
#    #unmask extracted features to match df_original index 
#    extracted_features = pd.concat( [df_original, extracted_features], 
#                                            axis=1 )
#    
#    # fill missing values with correct 
#    extracted_features.fillna(missing_value, inplace=True)
#    
#    
#    # deal with output location 
#    out_path = Path(path).parent.joinpath(Path(path).stem+"_features")
#    out_path.mkdir(parents=True, exist_ok=True)
#     
#    # write out features to csv file
#    print("features:"+os.path.join(out_path,'extracted_features.csv'))
#    extracted_features.to_csv(os.path.join(out_path,'extracted_features.csv'), chunksize=10000)
#    
#    # write data frame
#    kr = pd.DataFrame(list(extracted_features.columns))
#    kr.index += 1
#    kr.index.names = ['band']
#    kr.columns = ['feature_name']
#    kr.to_csv(os.path.join(out_path,"features_names.csv"))
#    
#    
#    # write out features to tiff file
#    if tiff_output == False:
#    
#        '''tiff_output is true and by default exports tiff '''
#    
#        return extracted_features  
#    
#    else:
#         print('use export_features instead')
#        # get image dimension from raw data
#        rows, cols, num = image_to_array(path).shape
#        # get the total number of features extracted
#        matrix_features = extracted_features.values
#        num_of_layers = matrix_features.shape[1]
#        
#        #reshape the dimension of features extracted
#        f2Array = matrix_features.reshape(rows, cols, num_of_layers)
#        output_file = 'extracted_features.tiff'  
#        
#        #Get Meta Data from raw data
#        raw_data = read_images(path)
#        GeoTransform = raw_data[0].GetGeoTransform()
#        driver = gdal.GetDriverByName('GTiff')
#        
#        noData = -9999
#        
#        Projection = raw_data[0].GetProjectionRef()
#        DataType = gdal.GDT_Float32
#        
#        #export tiff
#        CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType, path=out_path)
#        return extracted_features

def multiYear_Window_Extraction(startYear, endYear, featureData_Path, feature_params, out_Path, mask, window_length = 3, window_offset = 0):
    '''
    Extracts summary statistics(features) from multiYear datasets within moving window, across years
    Outputs a series of annual dataFrames as CSV files
    
    :param startYear: year on which to start feature extraction
    :param endYear: year on which to end feature extraction
    :param featureData_Path: file path to data from which to extract features
    :param feature_params: summary statistics(features) to extract from data within each window
    :param out_Path: file path to location at which extracted features should be output as a csv
    :param window_length: length of window within which to extract features
    :param window_offset: number of years by which features pertaining to each year are offset from that year
    :param mask:  mask to apply to data prior to feature extraction
    :return: no return.  instead, feature data relative to each year of interest is saved as a .csv file at the out_Path location
              under the filename FD_Window_XXXX.csv 
    '''
    
    # read in variables that are time variant, extract summary features, and concatenate output
    for x in range(startYear, endYear+1):

        #get climate parameters for desired window relative to iterated year
        extracted_features_iter = calculateFeatures_window(path = featureData_Path, 
                                                  parameters = feature_params, 
                                                  baseYear = x,
                                                  length = window_length,
                                                  offset = window_offset,
                                                  reset_df=True,
                                                  raster_mask =  mask,
                                                  tiff_output=True,
                                                  workers = 1,
                                                  outPath = out_Path)


        #reset index of extracted features to combine with other datasets based on pixel ids
        extracted_features_iter.reset_index(inplace = True)
        

        extracted_features_iter.to_csv(out_Path + "FD_Window_" + str(x- window_length - offset) +"_" +  str(x - window_offset) + ".csv")



def multiYear_Window_Extraction2(startYears,  featureData_Path, feature_params, out_Path, mask):
    '''
    Extracts summary statistics(features) from multiYear datasets within moving window, across years
    Outputs a series of annual dataFrames as CSV files
    
    :param startYears: list of years on which to start feature extraction
    :param endYear: year on which to end feature extraction
    :param featureData_Path: file path to data from which to extract features
    :param feature_params: summary statistics(features) to extract from data within each window
    :param out_Path: file path to location at which extracted features should be output as a csv
    :param window_length: length of window within which to extract features
    :param window_offset: number of years by which features pertaining to each year are offset from that year
    :param mask:  mask to apply to data prior to feature extraction
    :return: no return.  instead, feature data relative to each year of interest is saved as a .csv file at the out_Path location
              under the filename FD_Window_XXXX.csv 
    '''
    
    # read in variables that are time variant, extract summary features, and concatenate output
    for x in range(len(startYears)-1):
        length = startYears[x+1] - startYears[x]
        baseYear = startYears[x+1]

          #get climate parameters for desired window relative to iterated year
        extracted_features_iter = calculateFeatures_window(path = featureData_Path, 
                                                  parameters = feature_params, 
                                                  baseYear = baseYear,
                                                  length = length,
                                                  offset = 0,
                                                  reset_df=True,
                                                  raster_mask =  mask,
                                                  tiff_output=True,
                                                  workers = 1,
                                                  outPath = out_Path)


        #reset index of extracted features to combine with other datasets based on pixel ids
        extracted_features_iter.reset_index(inplace = True)
        

        extracted_features_iter.to_csv(out_Path + "FD_Window_" + str(startYears[x]) +"_" + str(startYears[x+1]) + ".csv")

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
   :param driver: data format of the output file (default gdal.GetDriverByName('GTiff'))
   :param noData: no data value (default -9999)
   :param DataType: array data format (default gdal.GDT_Float32)
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
    :param ml_task: string indicating intended machine learning task.  May be set to "regression", "classification", or "auto", in which case the task is inferred from y (default "auto")
    :param fdr_Level: float indicating theoretical expected percentage of irrelevant features among all created features (default 0.05)
    :return: dataframe
    '''
    
    from  tsraster.prep import set_common_index
    
    # remove non-matching indexes
    features, target = set_common_index(a=x, b=y)
    
    #if features.index.names==['pixel_id', 'time']:
    #    features.index = features.index.droplevel(level='time')
    
    #drop specified labels from features dataframe
    features = features.drop(labels=["id",'index', 'pixel_id','time'], axis=1, errors ='ignore') 
    
    # calculate relevance
    relevance_test = crt(features,
                     target.squeeze(),  # convert back from df to series
                     ml_task=ml_task,
                     fdr_level=fdr_level)
    
    return relevance_test

def checkRelevance2(x, y, ml_task="auto", fdr_level=0.05):
        '''
        Checks the statistical significance of features selects only significant ones

        :param x: pandas dataframe containing the features extracted
        :param y: pandas series
        :return: 2 dataframes relevance_test, relevant_features
        '''

        from  tsraster.prep import set_common_index

        # remove non-matching indexes
        features, target = set_common_index(a=x, b=y)
        #if features.index.names==['pixel_id', 'time']:
        #    features.index = features.index.droplevel(level='time')
    
        #drop specified labels from features dataframe
        features = features.drop(labels=["id",'index', 'pixel_id','time'], 
                                 axis=1, 
                                 errors ='ignore')

        # calculate relevance
        relevance_test = crt(features,
                             target.squeeze(),
                             ml_task=ml_task,
                             fdr_level=fdr_level)

        # gather subset of relevant features
        relevant_feature_names = relevance_test.feature[relevance_test.relevant==True]
        X_relevant_features = features[relevant_feature_names]
         
        return relevance_test, X_relevant_features 

def Temporal_Interpolation(input_path, outPath, startYear, endYear, interval, nullValues = [], exampleRaster = "../Data/Examples/3month_ts/aet/aet-201201.tif"):
    Years = np.arange(startYear, endYear+1, interval)
    '''
    :param input_path: filepath to input files, including filename - replace 4 digits of year with XXXX in filename
    :param output_path: filepath to location where output files will be created, as well as template file name - - replace 4 digits of year with XXXX in filename
    :param startYear: first year to include in interpolation
    :param endYear: last year to include in interpolation
    :param interval: interval (in years) between subsequent files
    :param nullValues: values that should be treated as null or 0 (as list)
    :param exampleRast: raster to use for acquiring projection and extent
    :return: returns nothing: exports a series of .tif files with data values corresponding to linear regressions between preceding and following data
    '''
    
    
    with rasterio.open("../Data/Examples/3month_ts/aet/aet-201201.tif") as exampleRast:
        array = exampleRast.read()
        profile = exampleRast.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=0)
    
    for x in range(len(Years)- 1):
        earlyYear = Years[x]
        lateYear = Years[x+1]
        for y in range(0, interval):
            
            iterYear = earlyYear + y
           
            earlyRasterName = input_path.replace('XXXX', str(earlyYear))
            earlyRaster = tr.read_images(earlyRasterName)
            earlyRaster = earlyRaster[0].ReadAsArray()
            earlyRaster[earlyRaster <0.0] = 0.0
            for z in nullValues:
                    earlyRaster[earlyRaster == y] = 0.0
            
        
            lateRasterName = input_path.replace('XXXX', str(lateYear))
            lateRaster = tr.read_images(lateRasterName)
            lateRaster = lateRaster[0].ReadAsArray()
            lateRaster[lateRaster <0.0] = 0.0
            for z in nullValues:
                    lateRaster[lateRaster == y] = 0.0

            #calculate annual raster by weighting between prior and successive raster
            earlyWeight = interval - y
            lateWeight = y
            
            iterRaster = ((earlyRaster * earlyWeight) + (lateRaster * lateWeight))/interval
            iterRaster[iterRaster<0.0] = 0.0
           
            iterRaster = np.float32(iterRaster)
           
            iter_outPath = outPath.replace("XXXX", str(iterYear))
            with rasterio.open(iter_outPath, 'w', **profile) as exampleRast:
                exampleRast.write(iterRaster, 1)