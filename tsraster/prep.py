'''
prep.py : reads and prepares raster files for time series feature extraction

authors: m.mann & a.bedada
'''


import numpy as np
import glob
import os.path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import gdal


def image_names(path):
    '''
    Reads raster files from multiple folders and returns their names

    :param path: directory path
    :return: names of the raster files
    '''
    images = glob.glob("{}/**/*.tif".format(path), recursive=True)
    image_name = [os.path.basename(tif).split('.')[0]
                  for tif in images]

    return image_name

def read_images(path):
    '''
    Reads a set of associated raster bands from a file.
    Can read one or multiple files stored in different folders.

    :param path: file name or directory path
    :return: raster files opened as GDALDataset
    '''

    if os.path.isdir(path):
        images = glob.glob("{}/**/*.tif".format(path), recursive=True)
        raster_files = [gdal.Open(f, gdal.GA_ReadOnly) for f in images]
    else:
        raster_files = [gdal.Open(path, gdal.GA_ReadOnly)]

    return raster_files

def image_to_array(path):
    '''
    Converts images inside multople folders to stacked array

    :param path: directory path
    :return: stacked numpy array
    '''

    raster_array = np.stack([raster.ReadAsArray()
                             for raster in read_images(path)],
                             axis=-1)

    return raster_array


def image_to_series(path):
    '''
    Converts images to one dimensional  array with axis labels

    :param path: directory path
    :return: pandas series
    '''

    rows, cols, num = image_to_array(path).shape
    data = image_to_array(path).reshape(rows*cols, num)

    # create index
    index = pd.RangeIndex(start=0, stop=len(data), step=1)#[str(i)  for i in range(1, len(data) + 1)]

    # create wide df with images as columns
    df = pd.DataFrame(data=data[0:,0:],
                      index=index, 
                      dtype=np.float32, 
                      columns=image_names(path))

    #reindex aand sort columns
    df2 = df.reindex(sorted(df.columns), axis=1)
    # stack columns as 1d array
    df2 = df2.stack().reset_index()
    # create a time series column
    df2['time'] = df2['level_1'].str.split('-').str[1]
    df2['kind'] = df2['level_1'].str.split('-').str[0]

    #rename all columns
    df2.columns =['id', 'level_1', 'value', 'time','kind']

    return df2

def image_to_series2(path, mask=None,missing_value=None):
    '''
    Converts images to one dimensional  array with axis labels
    
    :param path: directory path
    :return: pandas series
    '''
    
    # find all unique variables in path
    unique_variables = list(set([os.path.basename(i).split('-')[0] 
                 for i in glob.glob("{}/**/*.tif".format(path),
                                    recursive=True) ]))
    
    # convert to array
    rows, cols, num = image_to_array(path).shape
    data = image_to_array(path).reshape(rows*cols, num)
    
    # create index
    index = pd.RangeIndex(start=0, stop=len(data), step=1) 
    
    # create wide df with images as columns
    df_original = pd.DataFrame(data=data[0:,0:],
                      index=index, 
                      dtype=np.float32, 
                      columns=image_names(path))
 
    # add row id
    df_original['pixel_id'] = index
    
    if mask == None:
        df_mask = df_original
    else:
        df_mask = mask_df(original_df=df_original,raster_mask=mask)
    
    # remove any more missing values 
    #if missing_value != None:
    #    df_mask = df_mask[df_mask['value'] != missing_value]
    
    # convert to long format
    df_long = pd.wide_to_long(df_mask, unique_variables, i="pixel_id", j="time",sep='-',)
    
    # set pixel_id and year multi-index as columns
    # stack to long format 
    df_long = df_long.stack( ).reset_index()
    df_long.columns = ["pixel_id","time", "kind", "value"]
    
    # sort into correct format for feature extraction
    
    df_long.sort_values(['pixel_id', 'kind','time'], 
                   ascending=[True, True,True], 
                   inplace=True)
    
    # create empty df to use an example for unmasking 
    df_original = pd.DataFrame(index=df_original.index)
    
    return df_long, df_original 


def targetData(file):
    '''
    Reads and prepares the target data for prediction.

    :param file: raster file name
    :return: One-dimensional ndarray with axis
    '''

    # read image as array and reshape its dimension
    rows, cols, num = image_to_array(file).shape
    data = image_to_array(file).reshape(rows * cols)

    # create an index for each pixel
    index = pd.RangeIndex(start=0, stop=len(data), step=1)
    # convert N-dimension array to one dimension array
    df = pd.Series(data=data, index=index, dtype=np.int8, name='Y')

    return df

def poly_rasterizer(poly,raster_ex, raster_path_prefix, buffer_poly_cells=0):
    '''
    Rasterizes polygons by assigning a value 1.
    It can also add a buffer at a distance that is multiples of the example raster resolution

    :param poly: polygon to to convert to raster
    :param raster_ex: example tiff
    :param raster_path_prefix: directory path to the output file
    :param buffer_poly_cells: buffer size
    :return: a GeoTiff raster
    '''

    # check if polygon is already geopandas dataframe if so, don't read again
    if ('poly' in locals()):
        if not(isinstance(poly, gpd.geodataframe.GeoDataFrame)):
            poly = gpd.read_file(poly)
    else:
        poly = poly

    # create column of ones to rasterize for presence (1) of fire
    poly['ONES'] = 1

    # get example metadata
    with rasterio.open(raster_ex) as src:
        array = src.read()
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=0)
        out_arr = src.read(1) # get data from first band, this gets updated in write
        out_arr.fill(0) #set all values of raster to zero

    # reproject polygon to match crs of raster
    poly = poly.to_crs(src.crs)

    # buffer polygon to avoid edge effects
    if buffer_poly_cells != 0:
        poly['geometry'] = poly.buffer(buffer_poly_cells*src.res[0] ) # this creates an empty polygon geoseries

    # Write to tif, using the same profile as the source
    with rasterio.open(raster_path_prefix+'.tif', 'w', **profile) as dst:

        # generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(poly.geometry, poly.ONES))

        #rasterize shapes
        burned_value = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=dst.transform)
        dst.write(burned_value,1)


def mask_df(raster_mask, original_df):
    '''
    Reads in raster mask and subsets dataframe by mask index

    :param raster_mask: tif containing (0,1) mask
    :param original_df: a path to a pandas dataframe or series to mask
    :return: masked tiff
    '''

    # convert mask to pandas series
    index_mask = targetData(raster_mask)
    index_mask = index_mask[index_mask == 1]

         # check if polygon is already geopandas dataframe if so, don't read again
    if not(isinstance(original_df, pd.core.series.Series)) and \
            not(isinstance(original_df, pd.core.frame.DataFrame)):
        original_df = pd.read_csv(original_df)
    else:
        original_df = original_df

    # limit to matching index from index_mask
    original_df = original_df[original_df.index.isin(index_mask.index)]

    return original_df

def unmask_df(original_df, mask_df_output):
    '''
    Unmasks a dataframe with the raster file used for masking

    :param original_df: tif containing (0,1) mask
    :param mask_df_output: a path to a pandas dataframe or series to mask
    :return: unmasked output
    '''

         # check if polygon is already geopandas dataframe if so, don't read again
    if not(isinstance(original_df, pd.core.series.Series)) and \
            not(isinstance(original_df, pd.core.frame.DataFrame)):
        original_df = pd.read_csv(original_df)
    else:
        original_df = original_df

    # replace values based on masked values
    original_df.update(mask_df_output)

    return original_df

def check_mask(raster_mask, raster_input_ex):
    '''
    Checks that mask and input rasters have identical properties

    :param raster_mask: full path and prefix for raster name
    :param raster_input_ex: int specifying number of cells to buffer polygon with, 0 for no buffer
    :return: raster
    '''
    mask_list = []
    ex_list = []
    test_list = ['Mask','Resolution','Bounds','Shape']

    with rasterio.open(raster_mask) as mask:
        mask_list = [mask.crs,mask.res,mask.bounds,mask.shape]

    with rasterio.open(raster_input_ex) as ex:
        ex_list = [ex.crs,ex.res,ex.bounds, ex.shape]

    for i in range(0,len(mask_list)):
        if mask_list[i] == ex_list[i]:
            print(test_list[i]+": passed")
        else:
            print(test_list[i]+": FAILED")

    # close rasters
    mask.close()
    ex.close()


