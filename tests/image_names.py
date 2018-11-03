'''
prints out the names of raster files inside the directory
'''

from tsraster.prep import image_names

#file directory
path = "../docs/img/temperature/"
file_names = image_names(path)

print(file_names)