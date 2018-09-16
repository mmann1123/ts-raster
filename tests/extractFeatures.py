'''
this test script produces a tiff file containing extracted Features
'''

import os.path
from tsraster.calculate import extractFeatures

#file directory
input_files_path = "../docs/img/temperature/"

# output file name
output_file = "my_features.tif"


extractFeatures(input_files_path, output_file)

