
'''this test script produces array of extracted features '''

from tsraster.calculate import features2array

#file directory
input_files_path = "../docs/img/temperature/"

#run
my_features = features2array(input_files_path)

#check
print(my_features.shape)
