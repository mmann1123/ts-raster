Description of modules
==========================================

| ts-raster : python package for feature extraction and analysis from raster data

**prep.py: module for converting raster files to dataframe**

    prep.image_names  >>  gets the name of the files

    prep.image2array >> converts images n dimensional array

    prep.ts_series >> converts array to time-series dataframe


**sample.py: get sample pixel values**

    sample.sample >> extracts sample points from raster

**calculate.py: extracts features from rasters**

    calculate.calculateFeatures >> extracts features based on parameters requested
    calculate.features2array >> converts features stored in dataframe to array
    calculate.CreateTiff >> creates a raster containing features calculated
    calculate.extractFeatures >> converts features to raster (name provided by user)

