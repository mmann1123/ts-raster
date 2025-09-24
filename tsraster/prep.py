"""
prep.py : reads and prepares raster files for time series feature extraction

authors: m.mann & a.bedada
"""

import numpy as np
import glob
import os.path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from osgeo import gdal
from re import sub
from pathlib import Path


def set_df_mindex(df):
    """
    Returns dataframe with pixel_id and time index
    """
    df.set_index(["pixel_id", "time"], inplace=True)
    return df


def set_df_index(df):
    df.set_index(["pixel_id"], inplace=True)
    return df


def reset_df_index(df):
    df.reset_index(inplace=True)
    return df


def set_common_index(a, b):
    a = reset_df_index(if_series_to_df(a))
    b = reset_df_index(if_series_to_df(b))
    index_value = (
        a.columns.intersection(b.columns).intersection(["pixel_id", "time"]).tolist()
    )
    a.set_index(index_value, inplace=True)
    b.set_index(index_value, inplace=True)
    return a, b


def read_my_df(path):
    my_df = pd.read_csv(os.path.join(path, "my_df.csv"))
    my_df = set_df_mindex(my_df)  # sort
    # add columns needed for tsfresh
    my_df = reset_df_index(my_df)
    return my_df


def path_to_var(path):
    """
    Returns variable name from path to folder of tifs
    """
    return [
        sub(r"[^a-zA-Z ]+", "", os.path.basename(x).split(".")[0])
        for x in glob.glob("{}/**/*.tif".format(path), recursive=True)
    ][0]


def image_names(path):
    """
    Reads raster files from multiple folders and returns their names

    :param path: directory path
    :return: names of the raster files
    """
    images = glob.glob("{}/**/*.tif".format(path), recursive=True)
    image_name = [os.path.basename(tif).split(".")[0] for tif in images]

    # handle single tif case
    if len(image_name) == 0:
        image_name = [os.path.basename(path).split(".")[0]]

    return image_name


def read_images(path):
    """
    Reads a set of associated raster bands from a file.
    Can read one or multiple files stored in different folders.

    :param path: file name or directory path
    :return: raster files opened as GDALDataset
    """

    if os.path.isdir(path):
        images = glob.glob("{}/**/*.tif".format(path), recursive=True)
        raster_files = [gdal.Open(f, gdal.GA_ReadOnly) for f in images]
    else:
        raster_files = [gdal.Open(path, gdal.GA_ReadOnly)]

    return raster_files


def image_to_array(path):
    """
    Converts images inside multiple folders to stacked array

    :param path: directory path
    :return: stacked numpy array
    """

    raster_array = np.stack(
        [raster.ReadAsArray() for raster in read_images(path)], axis=-1
    )

    return raster_array


def image_to_series(path):
    """
    Converts images to one dimensional  array with axis labels

    :param path: directory path
    :return: pandas series
    """
    data = image_to_array(path)
    rows, cols, num = data.shape
    data = data.reshape(rows * cols, num)

    # create index
    index = pd.RangeIndex(start=0, stop=len(data), step=1, name="pixel_id")

    # create wide df with images as columns
    df = pd.DataFrame(
        data=data[0:, 0:], index=index, dtype=np.float32, columns=image_names(path)
    )

    # reindex and sort columns
    df2 = df.reindex(sorted(df.columns), axis=1)
    # stack columns as 1d array
    df2 = df2.stack().reset_index()
    # create a time series column
    df2["time"] = df2["level_1"].str.split("[- _]").str[1]
    df2["kind"] = df2["level_1"].str.split("[- _]").str[0]

    # set multiindex
    df2.set_index(["pixel_id", "time"], inplace=True)

    # rename all columns
    df2.columns = ["level_1", "value", "kind"]
    df2.drop(["level_1"], axis=1, inplace=True)

    # add columns needed for tsfresh
    df2.reset_index(inplace=True, level=["pixel_id", "time"])
    #    df2['pixel_id'] = df2.index.get_level_values('pixel_id')
    #    df2['time'] = df2.index.get_level_values('time')
    return df2


def image_to_series_simple(file, dtype=np.int8):
    """
    Reads and prepares single raster file

    :param file: raster file name
    :param dtype: numpy data type to return (default:np.int8)
    :return: One-dimensional ndarray with axis
    """

    # read image as array and reshape its dimension
    rows, cols, num = image_to_array(file).shape
    data = image_to_array(file).reshape(rows * cols)

    # create an index for each pixel
    index = pd.RangeIndex(start=0, stop=len(data), step=1, name="pixel_id")
    # convert N-dimension array to one dimension array
    df = pd.Series(data=data, index=index, dtype=dtype, name="value")

    return df


def poly_rasterizer(poly, raster_ex, raster_path_prefix, buffer_poly_cells=0):
    """
    Rasterizes polygons by assigning a value 1.
    It can also add a buffer at a distance that is multiples of the example raster resolution

    :param poly: polygon to to convert to raster
    :param raster_ex: example tiff
    :param raster_path_prefix: directory path to the output file example: 'F:/Boundary/StatePoly_buf'
    :param buffer_poly_cells: buffer size in cell count example: 1 = buffer by one cell
    :return: a GeoTiff raster
    """

    # check if polygon is already geopandas dataframe if so, don't read again
    if "poly" in locals():
        if not (isinstance(poly, gpd.geodataframe.GeoDataFrame)):
            poly = gpd.read_file(poly)
    else:
        poly = poly

    # create column of ones to rasterize for presence (1)
    poly["ONES"] = 1

    # get example metadata
    with rasterio.open(raster_ex) as src:
        array = src.read()
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=0)
        out_arr = src.read(1)  # get data from first band, this gets updated in write
        out_arr.fill(0)  # set all values of raster to zero

    # reproject polygon to match crs of raster
    poly = poly.to_crs(src.crs)

    # buffer polygon to avoid edge effects
    if buffer_poly_cells != 0:
        poly["geometry"] = poly.buffer(
            buffer_poly_cells * src.res[0]
        )  # this creates an empty polygon geoseries

    # Write to tif, using the same profile as the source
    with rasterio.open(raster_path_prefix + ".tif", "w", **profile) as dst:

        # generator of geom, value pairs to use in rasterizing
        shapes = ((geom, value) for geom, value in zip(poly.geometry, poly.ONES))

        # rasterize shapes
        burned_value = features.rasterize(
            shapes=shapes, fill=0, out=out_arr, transform=dst.transform
        )
        dst.write(burned_value, 1)


def poly_rasterizer_year_group(
    poly,
    raster_exmpl,
    raster_path_prefix,
    year_col_name="YEAR_",
    year_sub_list=range(1980, 1990),
):
    """
    Rasterizes polygons by assigning a value 1 to pixel. Utilizes year column to create
    an aggregated polygon across multiple year groups.


    :param poly: polygon to to convert to raster
    :param raster_ex: example tiff to base output on
    :param raster_path_prefix: directory path to the output file example: 'F:/Boundary/StatePoly_buf'
    :param year_col_name: column storing year to compare year_sub_list to
    :param year_sub_list: an int year, range(), or list of start end dates [1951, 1955]
    :return: a GeoTiff raster
    """

    # year or year groups must be forced into a list or range
    if type(year_sub_list) == int:
        year_sub_list = [year_sub_list]
    elif type(year_sub_list) == range:
        year_sub_list = year_sub_list
    elif type(year_sub_list) == list:
        # convert to range so all years are rasterized
        year_sub_list = range(year_sub_list[0], year_sub_list[1] + 1)

    # check if polygon is already geopandas dataframe if so, don't read again
    if not ("polys" in locals()):
        polys = gpd.read_file(poly)
    if "polys" in locals():
        if not (isinstance(polys, gpd.geodataframe.GeoDataFrame)):
            polys = gpd.read_file(poly)
    else:
        polys = poly

    # subset to year and convert to integer
    polys = polys[polys.loc[:, year_col_name].isin([str(i) for i in year_sub_list])]

    # create column of ones to rasterize for presence (1) of fire
    polys["ONES"] = 1

    # get example metadata
    with rasterio.open(raster_exmpl) as src:
        array = src.read()
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=0)
        out_arr = src.read(1)  # get data from first band, this gets updated in write

    # Write to tif, using the same profile as the source
    with rasterio.open(
        raster_path_prefix
        + str(year_sub_list[0])
        + "_"
        + str(year_sub_list[-1])
        + ".tif",
        "w",
        **profile
    ) as dst:

        # generator of geom, value pairs to use in rasterizing
        shapes = ((geom, value) for geom, value in zip(polys.geometry, polys.ONES))

        # rasterize shapes
        rasterized_value = features.rasterize(
            shapes=shapes, fill=0, out=out_arr, transform=dst.transform
        )
        dst.write(rasterized_value, 1)


def poly_to_series(poly, raster_ex, field_name, nodata=-9999, plot_output=True):
    """
    Rasterizes polygons by assigning a value 1.
    It can also add a buffer at a distance that is multiples of the example raster resolution

    :param poly: polygon to to convert to raster
    :param raster_ex: example tiff
    :param raster_path_prefix: directory path to the output file example: 'F:/Boundary/StatePoly_buf'
    :param nodata: (int or float, optional) – Used as fill value for all areas not covered by input geometries.
    :param nodata: (True False, optional) – Plot rasterized polygon data?

    :return: a pandas dataframe with a named column of rasterized data
    """

    # check if polygon is already geopandas dataframe if so, don't read again
    if "poly" in locals():
        if not (isinstance(poly, gpd.geodataframe.GeoDataFrame)):
            poly = gpd.read_file(poly)
    else:
        poly = poly

    # get example metadata
    with rasterio.open(raster_ex) as src:
        array = src.read()
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=nodata)
        out_arr = src.read(1)  # get data from first band, this gets updated in write
        out_arr.fill(nodata)  # set all values of raster to missing data value

    # reproject polygon to match crs of raster
    poly = poly.to_crs(src.crs)

    # generator of geom, value pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(poly.geometry, poly[field_name]))

    # rasterize shapes
    burned_value = features.rasterize(
        shapes=shapes, fill=nodata, out=out_arr, transform=src.transform
    )

    if plot_output == True:
        import matplotlib.pyplot as plt

        plt_burned_value = burned_value.copy()
        plt_burned_value[plt_burned_value == nodata] = np.NaN
        plt.imshow(plt_burned_value)
        plt.set_cmap("Reds")
        plt.colorbar()
        plt.show()

    # convert to array
    rows, cols = burned_value.shape
    data = burned_value.reshape(rows * cols, 1)

    # create index
    index = pd.RangeIndex(start=0, stop=len(data), step=1, name="pixel_id")

    # create wide df with images as columns
    df = pd.DataFrame(
        data=data[:, :], index=index, dtype=np.float32, columns=[field_name]
    )

    return df


def mask_df(raster_mask, original_df, missing_value=-9999, reset_index=True):
    """
    Reads in raster mask and subsets dataframe by mask index

    :param raster_mask: tif containing (0,1) mask where 1's are retained
    :param original_df: a path to a pandas dataframe, a series to mask, or a list of 2 dfs
    :param missing_value: additional missing values to be masked out
    :param reset_index: if true, any df index will be reset (added as columns to df)

    :return: masked df
    """

    # convert mask to pandas series keep only cells with value 1
    index_mask = image_to_series_simple(raster_mask)
    index_mask = index_mask[index_mask == 1]

    # if original_df is list concatenate by index
    if type(original_df) == list:
        list_flag = True
        first_df_shape = if_series_to_df(original_df[0]).shape

        try:
            original_df = pd.concat(original_df, axis=1, ignore_index=False)
        except:
            print(
                "time index missing in one element, merging list elements using only pixel_id index"
            )
            original_df = [
                set_df_index(reset_df_index(if_series_to_df(df))) for df in original_df
            ]
            original_df = pd.concat(original_df, axis=1, ignore_index=False)
            original_df = reset_df_index(original_df)
    else:
        list_flag = False

    # check if polygon is already geopandas dataframe if so, don't read again
    if not (isinstance(original_df, pd.core.series.Series)) and not (
        isinstance(original_df, pd.core.frame.DataFrame)
    ):
        original_df = read_my_df(original_df)

    # limit to matching pixels in index from index_mask
    try:
        original_df = original_df.iloc[
            original_df.index.get_level_values("pixel_id").isin(index_mask.index)
        ]
    except KeyError:
        # set multiindex
        original_df.set_index(["pixel_id", "time"], inplace=True)
        original_df = original_df.iloc[
            original_df.index.get_level_values("pixel_id").isin(index_mask.index)
        ]

    # remove any more missing values
    if missing_value != None:
        # inserts nan in missing value locations
        try:
            original_df = original_df[original_df.iloc[:, :] != missing_value]
        except:
            original_df = original_df[original_df.iloc[:] != missing_value]

        original_df.dropna(inplace=True)

    if list_flag == True:
        # split back out list elements
        a, b = (
            original_df.iloc[:, range(first_df_shape[1])],
            original_df.iloc[:, first_df_shape[1] :],
        )

        # reset index as columns
        if reset_index == True:
            a = reset_df_index(if_series_to_df(a))
            b = reset_df_index(if_series_to_df(b))
        return a, b

    else:
        # reset index as columns
        if reset_index == True:
            original_df = reset_df_index(if_series_to_df(original_df))
        return original_df


def unmask_df(original_df, mask_df_output):
    """
    Unmasks a dataframe with the raster file used for masking

    :param original_df: a data frame with the correct unmasked index values
    :param mask_df_output: a path to a pandas dataframe or series to mask
    :return: unmasked output
    """

    # check if df is already dataframe if so, don't read again
    if not (isinstance(original_df, pd.core.series.Series)) and not (
        isinstance(original_df, pd.core.frame.DataFrame)
    ):
        original_df = read_my_df(original_df)
    else:
        original_df = original_df

    # cover series to dataframes
    original_df = if_series_to_df(original_df)
    mask_df_output = if_series_to_df(mask_df_output)

    # find common index and set
    original_df, mask_df_output = set_common_index(a=original_df, b=mask_df_output)

    # limit original_df to col # of mask_df and change names to match
    original_df = original_df.iloc[:, : mask_df_output.shape[1]]
    original_df.columns = mask_df_output.columns
    original_df["value"] = -9999

    try:
        # replace values based on masked values, iterate through kind if multiple features
        for knd in mask_df_output["kind"].unique():
            original_df.update(mask_df_output[mask_df_output["kind"] == knd])
    except:
        # replace values based on masked values for non long form data types
        original_df.update(mask_df_output)

    return original_df


def unmask_from_mask(mask_df_output, raster_mask, missing_value=-9999):
    """
    Unmasks a multiindex dataframe with the raster file used for masking

    :param mask_df_output: a path to a pandas dataframe or series to mask with matching (multi)index values
    :param raster_mask: path to a rask max where 0 values are treated as missing
    :param missing_value: value assigned to missing values generally used for writing raster tifs
    :return: unmasked output
    """

    # set up df with correct index to unmask to
    unmask_df = if_series_to_df(image_to_series_simple(raster_mask, dtype=np.float32))
    unmask_df[unmask_df.value == 0] = missing_value
    unmask_df.reset_index(inplace=True)
    time_index = mask_df_output.reset_index().time.unique()[0]
    unmask_df["time"] = time_index
    unmask_df = set_df_mindex(unmask_df)

    # add placeholders for unmasked values
    for name in mask_df_output.columns:
        unmask_df[name] = unmask_df["value"]
    unmask_df.drop(columns=["value"], inplace=True)

    try:
        # replace values based on masked values, iterate through kind if multiple features
        for knd in mask_df_output["kind"].unique():
            unmask_df.update(mask_df_output[mask_df_output["kind"] == knd])
    except:
        # replace values based on masked values for non long form data types
        unmask_df.update(mask_df_output)

    return unmask_df


def check_mask(raster_mask, raster_input_ex):
    """
    Checks that mask and input rasters have identical properties

    :param raster_mask: full path and prefix for raster name
    :param raster_input_ex: int specifying number of cells to buffer polygon with, 0 for no buffer
    :return: raster
    """
    mask_list = []
    ex_list = []
    test_list = ["Mask", "Resolution", "Bounds", "Shape"]

    with rasterio.open(raster_mask) as mask:
        mask_list = [mask.crs, mask.res, mask.bounds, mask.shape]

    with rasterio.open(raster_input_ex) as ex:
        ex_list = [ex.crs, ex.res, ex.bounds, ex.shape]

    for i in range(0, len(mask_list)):
        if mask_list[i] == ex_list[i]:
            print(test_list[i] + ": passed")
        else:
            print(test_list[i] + ": FAILED")

    # close rasters
    mask.close()
    ex.close()


def combine_extracted_features(path, write_out=True, index_col=0):
    """
    Combines multiple extracted_features.csv files and assigns year prefix
    based on subfolder names.

    Folder structure assumed as follows:
         Precip>
                monthly1990-1995>
                    extracted_features.csv
                    extracted_features.tif
                monthly1996-2000>
                    extracted_features.csv
                    extracted_features.tif

    :param path: path to parent directory holding folders containing extracted features. (Example: Test)
    :param write_out: Should combined df be written to csv
    :param index_col: position of index in extracted_features.csv to be combined (default: 0, otherwise use None)
    :return: merged df containing all extracted_features.csv data with assigned year prefix
    """

    # get paths of all extracted_features.csv files
    all_files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(path)
        for name in files
        if name.endswith(("features.csv"))
    ]

    # extract numeric values from parent folder name
    parent_folder_years = [sub(r"\D", "", parent_folder) for parent_folder in all_files]
    print("Combining folder year names", parent_folder_years)

    # data read generator add year prefix to all column names  REMOVE?
    df_from_each_file = (
        pd.read_csv(all_files[i], index_col=index_col)
        .drop(["time"], errors="ignore", axis=1)
        .add_suffix("-" + parent_folder_years[i])
        for i in range(len(all_files))
    )

    # create joined df with all extraced_features data
    concatenated_df = pd.concat(df_from_each_file, axis=1, ignore_index=False)

    # set index to match others
    concatenated_df.index.names = ["pixel_id"]

    # deal with output location
    out_path = Path(path).parent.joinpath(Path(path).stem + "_features")
    out_path.mkdir(parents=True, exist_ok=True)

    # write combined extracted features data
    if write_out == True:
        concatenated_df.to_csv(
            os.path.join(out_path, "combined_extracted_features_df.csv"),
            chunksize=50000,
            index=False,
        )

    return concatenated_df


def combine_target_rasters(path, target_file_prefix, dep_var_name="Y", write_out=True):
    """
    Combines multiple extracted_features.csv files and assigns year prefix
    based on subfolder names.

    Folder structure assumed as follows:
         Path>
                target_2000-2005.tif
                target_2006-2010.tif
                target_2011-2016.tif


    :param path: path to parent directory holding folders containing extracted features. (Example: Test)
    :param target_file_prefix: prefix to search for in path (ex above: "target_")
    :param dep_var_name: column name to assign (default: "Y")
    :param write_out: Should combined df be written to csv
    :return: merged df containing all extracted_features.csv data with assigned year prefix
    """

    targets = glob.glob(
        ("{}/**/" + target_file_prefix + "*.tif").format(path), recursive=True
    )
    targets_years = [sub(r"\D", "", i) for i in targets]

    # rename columns with Y- prefix
    series_from_each_file = [
        image_to_series_simple(targets[i]).rename("Y-" + targets_years[i])
        for i in range(len(targets_years))
    ]

    # create joined df with all target data
    concatenated_df = pd.concat(series_from_each_file, axis=1, ignore_index=False)

    # deal with output location
    out_path = Path(path).parent.joinpath(Path(path).stem + "_target")
    out_path.mkdir(parents=True, exist_ok=True)

    # write combined extracted features data
    if write_out == True:
        print("writing file to ", out_path)
        concatenated_df.to_csv(
            os.path.join(out_path, "combined_target_df.csv"),
            chunksize=50000,
            index=False,
        )

    return concatenated_df


def wide_to_long_target_features(target, features, sep="-"):
    """
    Reads in target and feature data in wide format and returns long format

    :param target: target (Y) data wide format multiple years
    :param features: attribute (X) data wide format multiple years
    :return: target, attribute both in long format
    """
    # get variables to convert to long by removing dates at end of name
    target_stubs = list(
        set([sub(sep + r"\d+", "", i) for i in target.columns if i != "pixel_id"])
    )
    features_stubs = list(
        set([sub(sep + r"\d+", "", i) for i in features.columns if i != "pixel_id"])
    )

    target["pixel_id"] = target.index
    features["pixel_id"] = features.index

    target_ln = pd.wide_to_long(
        target, i="pixel_id", j="time", stubnames=target_stubs, sep=sep
    )
    features_ln = pd.wide_to_long(
        features, i="pixel_id", j="time", stubnames=features_stubs, sep=sep
    )

    if target_ln.index.equals(features_ln.index):
        print("converted to long, indexes match")
    else:
        print("index values did not match, make sure data is for same time period")
        return 0

    return target_ln, features_ln


def if_series_to_df(obj):
    # cover series to dataframes
    if isinstance(obj, pd.core.series.Series):
        obj = pd.DataFrame(data=obj, index=obj.index)
    return obj


def panel_lag_1(original_df, col_names, group_by_index="pixel_id"):
    """
    Adding temporal lag to df for selected columns

    :param original_df: any dataframe
    :param col_names: column names to add a lag to
    :param group_by_index:
    :return: original_df and lagged values with nans removed
    """

    # avoid duplicate column names
    col_names = list(set(col_names))

    # sort by pixel id, time
    original_df.sort_index(inplace=True)

    # check if all groups have same # of observations
    if not (original_df.count(level=group_by_index).all().all()):
        raise ("Data panel doesnt have balanced number of observations across groups")
        return 0

    # add lag
    for col in col_names:
        original_df = pd.concat(
            [
                original_df,
                original_df.loc[:, col]
                .groupby(by=[group_by_index])
                .shift(1)
                .rename(col + "_1"),
            ],
            axis=1,
        )

    # remove any columns that only have nan
    original_df.dropna(axis=1, how="all", inplace=True)
    original_df.dropna(inplace=True)

    return original_df
