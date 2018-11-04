from tsraster.prep import mask_df, unmask_df, check_mask
import pandas as pd

#path to files
raster_mask="../../wildfire_FRAP/Data/Examples/3month/aet-198401.tif"

original_df ="../../wildfire_FRAP/Data/Examples/3month/my_df.csv"

#run tasks
masked_data = mask_df(raster_mask, original_df)

#check maksing
original_df_file = pd.read_csv(original_df)
print("Check Masking")
print("Size of original data: ", original_df_file.shape)
print("Size of masked data: ", masked_data.shape)

print("Check Unmasking")
#check unmaksing
unmasked_data = unmask_df(original_df, masked_data)
print("Size of original data: ", original_df_file.shape)
print("Size of unmasked data: ", unmasked_data.shape)
