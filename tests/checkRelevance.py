
'''this test script returns p=value showing the significance of features '''

from tsraster.calculate import checkRelevance2
from tsraster.prep import targetData
import pandas as pd

# read csv extracted features
features = pd.read_csv("../../extracted_features.csv")
#path to target data
target_variable = "../docs/img/target_data/"
#read target data
target_data = targetData(target_variable)
#check for relevance
relevance = checkRelevance2(features, target_data)

print(relevance)