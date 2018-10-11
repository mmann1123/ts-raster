
'''this test script returns p=value showing the significance of features '''

from tsraster.calculate import checkRelevance

# read csv extracted features
features = pd.read_csv("test_features.csv")

#path to target data
target_variable = "../docs/img/target_data/"

#read target data
target_data = sRead.targetData(target_variable)

#check for relevance
relevance = checkRelevance(features, target_data)