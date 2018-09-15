# ts-raster
--------------------
ts-raster is a python package for extracting and analyzing of time-series characterstics from raster data. The feature extraction follows the footsteps of approaches developed in the python package <a href="https://github.com/blue-yonder/tsfresh">tsfresh</a>. 

- input : historical raster data (e.g. Monthly temperature data (2000-2018) 
- Extracted Feature: Mean, minimum, maximum, standard deviation... characterstics for all the data 
- output: data frame(CSV) or (array)Raster files of each

For analysis, several machine learning models as well as an ensemble modeling technique are incorporated. 


### Installation

    git clone https://github.com/adbeda/ts-raster
    cd ts-raster
    pip install -e .




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
        
A simple example:


```python

from tsraster.prep import sRead as tr
from tsraster.calculate import calculateFeatures


#directory
path = "../docs/img/temperature/"


image_name = tr.image_names(path)
print(image_name)
```

    ['tmx-200601', 'tmx-200603', 'tmx-200602', 'tmx-200703', 'tmx-200702', 'tmx-200701', 'tmx-200501', 'tmx-200502', 'tmx-200503']


Convert each image to array and stack them as bands


```python

rasters = tr.image2array(path)

rasters[0]
rasters.shape
```

    (1120, 872, 9)



```python
ts_features = calculateFeatures(path)
```

    Feature Extraction: 100%|██████████| 80/80 [01:18<00:00,  1.02it/s]


    variable  value__maximum  value__mean  value__median  value__minimum
    id                                                                  
    1.0                  0.0          0.0            0.0             0.0
    2.0                  0.0          0.0            0.0             0.0
    3.0                  0.0          0.0            0.0             0.0
    4.0                  0.0          0.0            0.0             0.0
    5.0                  0.0          0.0            0.0             0.0


Four features characterizing/summerising all input rasters are produced. 

