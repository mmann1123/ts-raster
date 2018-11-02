'''
prep.py : reads and prepares raster files for time series feature extraction
'''


import numpy as np
import glob
import os.path
import pandas as pd
from tsfresh import extract_features

try:
    import gdal
    from osgeo import gdal
except ImportError:
    raise ImportError('GDAL must be installed')

class sRead:

    def image_names(self):

        '''

        image_names: reads images and returns the name of the file
             image_files - a for loop to connect a directory to the file ending with .tif
             image_names -  a for loop to split and return name
        '''

        images = glob.glob("{}/**/*.tif".format(self), recursive=True)

        image_name = [os.path.basename(tif).split('.')[0]
                      for tif in images]


        return image_name

    def image(self):
        #read images from sub-directories
        images = glob.glob("{}/**/*.tif".format(self), recursive=True)
        raster_files = [gdal.Open(f, gdal.GA_ReadOnly) for f in images]
        return raster_files


    def image2array(self):
        '''
            image2array: reads images, stack as bands and returns array
                image - a for loop to connect a directory to the file ending with .tif
                raster_files -  open images with gdal
                raster_array - converts each raster to array and stack them together by columns

            return: array
        '''

        #stack images as bands
        raster_array = np.stack([raster.ReadAsArray()
                                 for raster in sRead.image(self)],
                                axis=-1)

        return raster_array


    def ts_series(self):

        '''
        ts_series: reads array and returns multi index data frame for time series
            data - 3 dimensional images shaped to 2d
            index - row id for each pixel
            df - 2d array returned as data frame
            df2 - dataframe stacked for multi index

        return: dataframe
        '''

        rows, cols, num = sRead.image2array(self).shape
        data = sRead.image2array(self).reshape(rows*cols, num)


        index = [str(i)
                 for i in range(1, len(data) + 1)]


        df = pd.DataFrame(data=data[0:,0:],
                          index=index, columns=sRead.image_names(self))


        df2 = df.reindex(sorted(df.columns), axis=1) #sort by month
        df2 = df2.stack().reset_index()

        df2['time'] = df2['level_1'].str.split('-').str[1] # extract time
        df2.columns =['id', 'kind', 'value', 'time']

        '''tsfresh doesn't accept na values '''

        df3 = df2.replace(df2.value[0], 0)

        return df3

    def image2series(self):

        '''
        ts_series: reads array and returns multi index data frame for time series
            data - 3 dimensional images shaped to 2d
            index - row id for each pixel
            df - 2d array returned as data frame
            df2 - dataframe stacked for multi index

        return: dataframe
        '''

        rows, cols, num = sRead.image2array(self).shape
        data = sRead.image2array(self).reshape(rows*cols, num)

        # create index
        index = [str(i)
                 for i in range(1, len(data) + 1)]

        # convert array to dataframe
        # change dtype from float64 to integer
        df = pd.DataFrame(data=data[0:,0:],
                          index=index, dtype=np.int8, columns=sRead.image_names(self))

        #reindex columns
        df2 = df.reindex(sorted(df.columns), axis=1) #sort by month

        # stack n-columns (months) into one column
        df2 = df2.stack().reset_index()

        # rename column
        df2['time'] = df2['level_1'].str.split('-').str[1] # extract time
        df2.columns =['id', 'kind', 'value', 'time']

        '''tsfresh doesn't accept na values '''

        #replace -Inf with 0
        #df3 = df2.replace(df2.value[0], 0)

        return df2


    def targetData(self):
        '''
        ts_series: reads and converts arrays to Series
        return: pd.Series
        '''

        # read image
        rows, cols, num = sRead.image2array(self).shape

        # reshape array to 1D
        data = sRead.image2array(self).reshape(rows * cols)

        # create index
        index = pd.RangeIndex(start=0, stop=len(data), step=1)  

        # convert array pd.Series
        df = pd.Series(data=data, index=index, dtype=np.int8, name='Y')
     
        return df
    
    def poly_rasterizer(poly,raster_ex, raster_path_prefix, buffer_poly_cells=0):
    '''
    :poly_rasterizer: Function rasterizes polygons assigning the value 1, it \\
    can also add a buffer at a distance that is multiples of the example raster resolution
    :param raster_path_prefix: full path and prefix for raster name 
    :param buffer_poly_cells: int specifying number of cells to buffer polygon with, 0 for no buffer
    :return: raster
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