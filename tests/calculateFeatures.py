
'''this test script produces a data-frame containing extracted features '''

from tsraster.calculate import calculateFeatures

#file directory
input_files_path = "../docs/img/temperature/"

#run
my_features = calculateFeatures(input_files_path)

 #optional:: export
my_features.to_csv("df_features.csv")