from tsraster.prep import mask_df, unmask_df, check_mask
import pandas as pd
#path to files
raster_mask="../../wildfire_FRAP/Data/Examples/3month/aet-198401.tif"

original_df ="../../wildfire_FRAP/Data/Examples/3month/my_df.csv"

#run task
masked_data = mask_df(raster_mask, original_df)


original_df_file = pd.read_csv(original_df)
print("Size of unmasked data: ", original_df_file.shape)
print("Size of masked data: ", masked_data.shape)
