''' this test script convert rasters to time-series data format '''

from tsraster.prep import sRead as tr


#file directory
input_files_path = "../docs/img/temperature/"

#run
ts_df = tr.ts_series(input_files_path)

#check
print(ts_df.head())