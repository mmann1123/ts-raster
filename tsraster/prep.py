'''
prep.py : reads and prepares raster files for time series feature extraction
'''


import numpy as np
import gdal
import glob
import os.path
import pandas as pd
from tsfresh import extract_features

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
