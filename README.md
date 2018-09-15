# ts-raster
--------------------
ts-raster is a python package for extracting and analyzing of time-series characterstics from raster data. The feature extraction follows the footsteps of approaches developed in the python package <a href="https://github.com/blue-yonder/tsfresh">tsfresh</a>. 

- input : historical raster data (e.g. Monthly temperature data (2000-2018) 
- Extracted Feature: Mean, minimum, maximum, standard deviation... characterstics for all the data 
- output: CSV or Raster files of each

For analysis, several machine learning models as well as an ensemble modeling technique are incorporated. 

### Input Data Structure

The input raster files from which features will be extracted have to prepared as:

    temprature
        2005
            temp-200501.tif 
            temp-200502.tif
            temp-200503.tif ...
        2006
            temp-200601.tif
            temp-200602.tif
            temp-200603.tif...
        2007
            ...
        
 The variable, in this case, is "temprature" and under 2005, 2006 and 2007 monthly temprature data is stored.


### Installation

    git clone https://github.com/adbeda/ts-raster
    cd ts-raster
    pip install -e .


#### Disclaimer
ts-raster is developed as part of a research project for california wild-fire modeling.