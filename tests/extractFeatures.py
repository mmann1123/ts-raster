'''
this test script produces a tiff file containing extracted Features
'''

import os.path

from tsraster.calculate import exportFeatures
import pandas as pd

#file directory
path = "../docs/img/temperature/"
input_file = pd.read_csv("../docs/img/temperature/extracted_features.csv")

#name of output file
output_file = "extracted_features.tiff"

exportFeatures(path=path, input_file=input_file, output_file=output_file)

