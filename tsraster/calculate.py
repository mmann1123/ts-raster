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
import dask as dask
import copy
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
    '''Calculates features or the statistical characteristics of time-series raster data.
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


def calculateFeaturesDask(path, parameters, reset_df,raster_mask=None ,tiff_output=True,missing_value = -9999, workers = None):
    
  ''' Calculates features or the statistical characteristics of time-series raster data.
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
      Distributor = LocalDaskDistributor(n_workers=workers,
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



def calculateFeatures_window(path, parameters, baseYear, reset_df = True,length = 3, offset = 0,raster_mask=None ,tiff_output=True, workers = None, outPath= "None", dataTypes = "*"):
    
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
    :param dataTypes: string of the dataType of interest (e.g., aet, etc - may be * in cases where all dataTypes are desired) or list of strings consisting of the datatypes of interest
    :return: extracted features as a dataframe and tiff file
    '''
    
  
    if reset_df == False:
        #if reset_df =F read in csv file holding saved version of my_df
        my_df = tr.read_my_df(path)
            
    else:
        #if reset_df =T calculate ts_series and save csv
        my_df = image_to_series_window(path, baseYear, length, offset,  dataTypes)
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
      baseName = str(baseYear + offset) + '_' + str(baseYear + length+ offset -1)
    elif length<0: 
      baseName = str(baseYear + length+ offset + 1) + '_' + str(baseYear + offset)

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






def multiYear_Window_Extraction(startYears, featureData_Path, feature_params, out_Path, mask, length = 3, offset = 0, workers = None, dataTypes = "*"):
    '''
    Extracts summary statistics(features) from multiYear datasets within moving window, across years
    Outputs a series of annual dataFrames as CSV files
    
    :param startYears: list of years on which to start feature extraction
    :param featureData_Path: file path to data from which to extract features
    :param feature_params: summary statistics(features) to extract from data within each window
    :param out_Path: file path to location at which extracted features should be output as a csv
    :param length: length of window within which to extract features
    :param offset: number of years by which features pertaining to each year are offset from that year
    :param workers: number of workers
    :param mask:  mask to apply to data prior to feature extraction
    :param dataTypes: string of the dataType of interest (e.g., aet, etc - may be * in cases where all dataTypes are desired) or list of strings consisting of the datatypes of interest
    :return: no return.  instead, feature data relative to each year of interest is saved as a .csv file at the out_Path location
              under the filename FD_Window_XXXX.csv 
    '''
    
    # read in variables that are time variant, extract summary features, and concatenate output
    for x in startYears:

        #get climate parameters for desired window relative to iterated year
        extracted_features_iter = calculateFeatures_window(path = featureData_Path, 
                                                  parameters = feature_params, 
                                                  baseYear = x,
                                                  length = length,
                                                  offset = offset,
                                                  reset_df=True,
                                                  raster_mask =  mask,
                                                  tiff_output=True,
                                                  workers = workers,
                                                  outPath = out_Path,
                                                  dataTypes = dataTypes)


        #reset index of extracted features to combine with other datasets based on pixel ids
        extracted_features_iter.reset_index(inplace = True)
        

        if length>0:
          baseName = str(x + offset) + '_' + str(x + length+ offset -1)
        elif length<0: 
          baseName = str(x + length+ offset + 1) + '_' + str(x + offset)

        extracted_features_iter.to_csv(out_Path + "FD_Window_" + baseName + ".csv", index = False)

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

def Extract_features_Dask(df, 
                          parameters = {'mean':True, 'max': True, 'min': True},
                          dataTypes = ['ppt','tmx','aet','cwd','pet','pck'],
                         column_id = 'pixel_id'):
    '''
    Extract mean, max, and/or min values for each pixel from a dask dataFrame derived from monthly rasters by pixel_id
    Outputs a csv of extracted features, alongside a tif of each extracted feature

    :param df: input dask dataFrame from which features are to be extracted
    :param dataTypes: dictionary of three extraction types: defaults to outputting all three, some may be excluded by seting values to false
    :param column_id: id of column by which to identify pixels: defaults to 'pixel_id'
   
    '''

    df = df[[column_id]+ dataTypes]
    
    
    groupByList = []
    
    if parameters['mean'] == True:
        means = df.groupby('pixel_id').mean()
        groupByList.append(means)
    elif parameters['mean'] != True:
        means = None
    
    if parameters['max'] == True:
        maxs = df.groupby('pixel_id').max()
        groupByList.append(maxs)
    elif parameters['max'] != True:
        maxs = None
        
    if parameters['min'] == True:
        mins = df.groupby('pixel_id').min()
        groupByList.append(maxs)
    elif parameters['min'] != True:
        mins = None
        
    
    groupByFrames = dask.compute(means, maxs, mins)
    
    
    
    #using these groupByFrames, populate a new pandas df with data from each, with appropriate column headers
    is_outFrame_built = False # set outFrame to None so that it can be populated by pixel_id index when needed
    
    paramList = list(parameters.keys()) # turn dictionary keys into list for iteration
    for x in range(len(paramList)):
        if type(groupByFrames[x]) == pd.core.frame.DataFrame:
            for y in range(len(dataTypes)):
                if is_outFrame_built == False:
                    outFrame = groupByFrames[x].loc[:, []] #create blank dataFrame with index pixel_id 
                    is_outFrame_built = True
                outFrame[dataTypes[y] + '_' + paramList[x]] = groupByFrames[x][dataTypes[y]] #add data from a given feature & data type to outFrame
    return outFrame
  
    #using these groupByFrames, populate a new pandas df with data from each, with appropriate column headers
    is_outFrame_built = False # set outFrame to None so that it can be populated by pixel_id index when needed
    paramList = list(parameters.keys()) # turn dictionary keys into list for iteration
    for x in range(len(paramList)):
        if type(groupByFrames[x]) == pd.core.frame.DataFrame:
            for y in range(len(dataTypes)):
                if is_outFrame_built == False:
                    outFrame = groupByFrames[x].loc[:, []] #create blank dataFrame with index pixel_id 
                    is_outFrame_built = True
                outFrame[dataTypes[y] + '_' + paramList[x]] = groupByFrames[x][dataTypes[y]] #add data from a given feature & data type to outFrame
    return outFrame                

def multiYear_Window_Extraction_Dask_No_TSFresh(startYears, 
                              featureData_Path,  
                              out_Path, 
                              raster_mask = None,  
                              feature_params = {'mean':True, 'max': True, 'min': True, 'std': True},
                              length = 3, 
                              offset = 0, 
                              dataTypes = ['aet', 'cwd'],
                              chunks = 1000,
                              reset_df = True):
  
  '''Extracts summary statistics(features) from multiYear datasets within moving window, across years
  Outputs a series of annual dataFrames as CSV files using dask
  
  :param startYears: list of years on which to start feature extraction
  :param featureData_Path: file path to data from which to extract features
  :param feature_params: summary statistics(features) to extract from data within each window
  :param out_Path: file path to location at which extracted features should be output as a csv
  :param window_length: length of window within which to extract features
  :param window_offset: number of years by which features pertaining to each year are offset from that year
  :param raster_mask:  mask to apply to data prior to feature extraction
  :param dataTypes: list of dataTypes to be examined
  :param chunks: size of chunk to be used by dask
  :param reset_df: if False, attempt to open pre-existing feature data before recalculating it in each period of interest
          --Should be set to true if features or mask havew changed since previous versions were calculated
  :return: no return.  instead, feature data relative to each year of interest is saved as a .csv file at the out_Path location
            under the filename FD_Window_XXXX.csv 
  '''
  
  
  
  
  
  # read in variables that are time variant, extract summary features, and concatenate output
  for x in startYears:
      print(x)
      #create base part of name for extracted feature data
      if length>0:
        baseName = str(x + offset) + '_' + str(x + length+ offset -1)
      elif length<0: 
        baseName = str(x + length+ offset + 1) + '_' + str(x + offset)
      
      
      if reset_df == False:
          try:
              df_iter = pd.read_csv(out_path + "FD_Window_" + baseName + ".csv", index = False)
              new_df_Needed = False
          except:
              new_df_Needed = True
      
      elif reset_df != False:
          new_df_Needed = True
      
      
      
      if new_df_Needed == True: #read in climate data from rasters and extract features
          
          
          df_iter = tr.image_to_Dask_Dataframe(path = featureData_Path, 
                                       baseYear = x, 
                                       length = length, 
                                       offset = offset,
                                       dataTypes = dataTypes,
                                       examplePath = raster_mask,
                                       chunks = 1000)

          #get climate parameters for desired window relative to iterated year
          extracted_features_iter = Extract_features_Dask(df_iter, 
                            parameters = feature_params,
                            dataTypes = dataTypes,
                           column_id = 'pixel_id')
          
          #output extracted features to csv
          extracted_features_iter.to_csv(out_Path + "FD_Window_" + baseName + "_PreMasked.csv", index = False)
          


      
          #reset index of extracted features to combine with other datasets based on pixel ids
          extracted_features_iter.reset_index(inplace = True)
      
          # mask rasters based on desired mask, if present
          if raster_mask is not None:
              extracted_features_iter = tr.mask_df(raster_mask = raster_mask, 
                              original_df = extracted_features_iter, multiIndex = False)
      
  

      extracted_features_iter.to_csv(out_Path + "FD_Window_" + baseName + ".csv", index = False)



def calculateFeatures_window_Dask(path,
                                  parameters, 
                                  baseYear, 
                                  reset_df = True,
                                  length = 3, 
                                  offset = 0,
                                  raster_mask=None,
                                  tiff_output=True, 
                                  workers = 7, 
                                  outPath= "None"):
    
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
      Distributor = LocalDaskDistributor(n_workers=workers)
      #Distributor = LocalDaskDistributor(n_workers=workers)
  else:
      Distributor = None
  
  extracted_features = extract_features(my_df, 
                                        default_fc_parameters = parameters,
                                        column_sort = "time",
                                        column_value = "value",
                                        column_id = "pixel_id",
                                        column_kind="kind", 
                                        chunksize = 1000,
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
    baseName = str(baseYear + offset) + '_' + str(baseYear + length+ offset -1)
  elif length<0: 
    baseName = str(baseYear + length+ offset + 1) + '_' + str(baseYear + offset)

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

def multiModel_Window_Extraction(startYears, 
                                featureData_Path,  
                                out_Path, 
                                raster_mask = None,
                                exampleRasterPath = "C:/Users/isaac/Documents/wildfire_FRAP_working/wildfire_FRAP/Data/Examples/buffer/StatePoly_buf.tif",
                                feature_params = {'mean':True, 'max': True, 'min': True},
                                length = 1, 
                                offset = 0, 
                                dataTypes = ['ppt','tmx','aet','cwd','pet','pck'],
                                chunks = 1000,
                                reset_df = True):
    '''
    Extracts summary statistics(features) from multiYear datasets within moving window, across years
    Outputs a series of annual dataFrames as CSV files using dask
    
    :param startYears: list of years on which to start feature extraction
    :param featureData_Path: file path to data from which to extract features
    :param feature_params: summary statistics(features) to extract from data within each window
    :param out_Path: file path to location at which extracted features should be output as a csv
    :param window_length: length of window within which to extract features
    :param window_offset: number of years by which features pertaining to each year are offset from that year
    :param raster_mask:  mask to apply to data prior to feature extraction
    :param dataTypes: list of dataTypes to be examined
    :param chunks: size of chunk to be used by dask
    :param reset_df: if False, attempt to open pre-existing feature data before recalculating it in each period of interest
            --Should be set to true if features or mask havew changed since previous versions were calculated
    :return: no return.  instead, feature data relative to each year of interest is saved as a .csv file at the out_Path location
              under the filename FD_Window_XXXX.csv 
    '''
    
    

    
    
    
    
    # read in variables that are time variant, extract summary features, and concatenate output
    for x in startYears: #iterate across years
        print(x)
        for y in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]: #iterate across months


            if reset_df == False:
                try:
                    df_iter = pd.read_csv(out_path + "FD_Window_" +  str(x) + "_MultiModel" + ".csv", index = False)
                    new_df_Needed = False
                except:
                    new_df_Needed = True

            elif reset_df != False:
                new_df_Needed = True



            if new_df_Needed == True: #read in climate data from rasters and extract features


                df_iter = tr.image_to_Dask_Dataframe(path = featureData_Path, 
                                             baseYear = x, 
                                             length = length, 
                                             offset = 0,
                                             dataTypes = dataTypes,
                                             chunks = 1000, 
                                             month = y, 
                                             examplePath = exampleRasterPath)

                #get climate parameters for desired window relative to iterated year
                extracted_features_iter = Extract_features_Dask(df_iter, 
                                  parameters = feature_params,
                                  dataTypes = dataTypes,
                                 column_id = 'pixel_id')


                #reset index of extracted features to combine with other datasets based on pixel ids
                extracted_features_iter.reset_index(inplace = True)


                #output extracted features to csv
                extracted_features_iter.to_csv(out_Path + "FD_Window_" + str(x) + "_" + str(y) + "_MultiModel.csv", index = False)
                
                #output extracted features to raster
                for dataType in dataTypes:
                    for feature in feature_params:
                        tr.seriesToRaster(extracted_features_iter.loc[:, [dataType + "_"+ feature] ], exampleRasterPath, out_Path + feature + "/" + dataType + "-" +  str(x) + y + '.tif', noData = -9999)



def multi_year_Annual_Summaries(inpath, outpath, startYear, endYear, length, 
                         id_val = 'pixel_id', 
                         timeCode = 'time',
                         mean_out = True,
                         max_out = True,
                         min_out = True,
                         std_out = True):
    '''Extract multi-year minima, maxima, and mean values from annual features that have been previously exracted
    :param inpath: filepath to annual data, including filename with 4-digit years replaced by XXXX
        example - inpath = "C:/Users/isaac/Documents/wildfire_FRAP_working/wildfire_FRAP/Data/Examples/HadGES2_oneYear_Examples/extracted_featuresXXXX_XXXX.csv"
    :param outpath: filepath to output files (requires that subfolders Annual_means, Annual_maxima, and Annual minima have also been created within it)
    :param startYear: year on which to begin iterating
    :param endYear: year on which to cease iterating
    :param length: length of time (i.e. # of years, hind-looking) over which to calculate minima, maxima, and means
    :param id_val: column name used to identify pixels in extracted features
    :param timecode: column name used to identify timecode for extracted features in original data
    :param mean_out: Boolean value to determine whether multiYear means should be calculated
    :param max_out: Boolean value to determine whether multiYear maxima should be calculated
    :param min_out: Boolean value to determine whether multiYear minima should be calculated




    '''
    
    for iterYear in range(startYear, endYear+1, 1):
        
        for y in range(length):
            inpath_iter = inpath.replace('XXXX', str(iterYear - y))
            annData = pd.read_csv(inpath_iter)
            columns = annData.columns.tolist()
            del annData[timeCode]
            
            if y ==0:
                stackData = copy.deepcopy(annData)
            elif y>0:
                stackData = pd.concat([stackData, copy.deepcopy(annData)])
            
            if mean_out ==True:
                meanData = stackData.groupby(by = [id_val]).mean()
                meanData[timeCode] = str(iterYear-y) + '01_' + str(iterYear) + '12'
                meanData.reset_index(inplace = True)
                meanData = meanData[columns]
                iter_outpath = outpath + 'Annual_means/extracted_features' + str(iterYear - length+1) + '_' + str(iterYear) + '.csv'
                meanData.to_csv(iter_outpath, index = False)
            
            if mean_out ==True:
                maxData = stackData.groupby(by = [id_val]).max()
                maxData[timeCode] = str(iterYear-y) + '01_' + str(iterYear) + '12'
                maxData.reset_index(inplace = True)
                maxData = maxData[columns]
                iter_outpath = outpath + 'Annual_maxima/extracted_features' + str(iterYear - length+1) + '_' + str(iterYear) + '.csv'
                maxData.to_csv(iter_outpath, index = False)
                
            if min_out == True:
                minData = stackData.groupby(by = [id_val]).min()
                minData[timeCode] = str(iterYear-y) + '01_' + str(iterYear) + '12'
                minData.reset_index(inplace = True)
                minData = minData[columns]
                iter_outpath = outpath + 'Annual_minima/extracted_features' + str(iterYear - length+1) + '_' + str(iterYear) + '.csv'
                minData.to_csv(iter_outpath, index = False)
            
            if std_out == True:
                stdData = stackData.groupby(by = [id_val]).std()
                stdData[timeCode] = str(iterYear-y) + '01_' + str(iterYear) + '12'
                stdData.reset_index(inplace = True)
                stdData = stdData[columns]
                iter_outpath = outpath + 'Annual_std/extracted_features' + str(iterYear - length+1) + '_' + str(iterYear) + '.csv'
                stdData.to_csv(iter_outpath, index = False)
            
    
            


