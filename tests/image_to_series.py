'''
prints out the names of raster files inside the directory
'''

from tsraster.prep import image_to_series

#file directory
path = "../docs/img/temperature/"
series_df = image_to_series(path)

print(series_df)