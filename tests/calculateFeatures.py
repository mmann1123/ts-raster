
'''this test script produces a data-frame containing extracted features '''

from tsraster.calculate import calculateFeatures

#file directory
path = "../docs/img/temperature/"

parameters = {
    "mean": None,
    "maximum": None,
    "median":None,
    "minimum":None,
    "skewness":None,
    "sum_values":None
}
group_1_results = calculateFeatures(path, parameters, reset_df=True)
