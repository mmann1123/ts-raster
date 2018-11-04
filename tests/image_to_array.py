'''
prints out the names of raster files inside the directory
'''

from tsraster.prep import image_to_series

#file directory
path = "../docs/img/temperature/"
data= image_to_series(path)

print(data.shape)