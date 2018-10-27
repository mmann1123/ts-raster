
'''this test script produces a data-frame containing extracted features '''

from tsraster.calculate_v2 import calculateFeatures

#file directory
path = "../docs/img/temperature/"

parameters = {
    "mean": None,
    "maximum": None,
    "median":None,
    "minimum":None,
    "mean_abs_change":None,
    "mean_change":None,
    "quantile":[{"q": 0.15},{"q": 0.05},{"q": 0.85},{"q": 0.95}],
    "skewness":None,
    "sum_values":None
}
group_1_results = calculateFeatures(path=path, parameters=parameters, reset_df=False, tiff_output=True)
