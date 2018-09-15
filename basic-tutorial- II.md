
## Feature extraction

This documents show how to extract important timeseries charcterstics from raster files.


```python
import os.path

import matplotlib.pyplot as plt
%matplotlib inline

from rasterio.plot import show

import tsraster
from tsraster.prep import sRead as tr

from tsraster.calculate import calculateFeatures
from tsraster.calculate import features2array
```

connect to the data directory


```python
path = "../docs/img/temperature/"
```

read the images and convert the arrays to a time-series dataframe 


```python
rasters = tr.ts_series(path)
```

Lets take a look at the time-series data


```python
rasters.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>kind</th>
      <th>value</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>tmx-200501</td>
      <td>0.0</td>
      <td>200501</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>tmx-200502</td>
      <td>0.0</td>
      <td>200502</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>tmx-200503</td>
      <td>0.0</td>
      <td>200503</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>tmx-200601</td>
      <td>0.0</td>
      <td>200601</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>tmx-200602</td>
      <td>0.0</td>
      <td>200602</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>tmx-200603</td>
      <td>0.0</td>
      <td>200603</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>tmx-200701</td>
      <td>0.0</td>
      <td>200701</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>tmx-200702</td>
      <td>0.0</td>
      <td>200702</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>tmx-200703</td>
      <td>0.0</td>
      <td>200703</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>tmx-200501</td>
      <td>0.0</td>
      <td>200501</td>
    </tr>
  </tbody>
</table>
</div>



Accordingly, each pixel is identified by *id*. Since there are 9 bands (3 images per year), **tsraster** assigns each 9 first pixels the id number 1 and differentiate each by their respective year and month. Hence, the *time* column organizes and orders pixels by time. The *value* column represent the pixel value

We can summerize the data and examine its pattern.


```python
rasters.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.789760e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.049381e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.123363e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.250000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.343750e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.900000e+01</td>
    </tr>
  </tbody>
</table>
</div>



Let's go ahead and extract features


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


The feature extraction swift through all bands and calculate values such maximum, minimum and mean, median. 
In this instance, 4 features of temprature data are generated for 9 rasters representing 3 months of 3 years of data.

Let's take a look at the summary of these features.


```python
ts_features.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>variable</th>
      <th>value__maximum</th>
      <th>value__mean</th>
      <th>value__median</th>
      <th>value__minimum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>976640.000000</td>
      <td>976640.000000</td>
      <td>976640.000000</td>
      <td>976640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.174632</td>
      <td>7.053721</td>
      <td>5.764061</td>
      <td>3.676642</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.604448</td>
      <td>9.836145</td>
      <td>8.230023</td>
      <td>5.625607</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-0.569444</td>
      <td>-4.375000</td>
      <td>-7.250000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>31.062500</td>
      <td>17.548611</td>
      <td>14.000000</td>
      <td>8.687500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>49.000000</td>
      <td>30.666666</td>
      <td>27.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>



Next, convert these features to array, visualize or create a tiff


```python
# first, get the original dimension/shape of image 
og_rasters = tr.image2array(path)
rows, cols, nums = og_rasters.shape


# convert df to matrix array
matrix_features = ts_features.values
num_of_layers = matrix_features.shape[1]


f2Array = matrix_features.reshape(rows, cols, num_of_layers)
print(f2Array.shape)

```

    (1120, 872, 4)


visualize features


```python
fig, ax = plt.subplots(2,2,figsize=(16,10))

for i in range(0,f2Array.shape[2]):
    img = rasters[:,:,i]
    i = i+1
    plt.subplot(1,4,i)
    plt.imshow(img, cmap="Greys")

```


![png](output_20_0.png)

