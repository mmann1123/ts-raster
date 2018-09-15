'''

 sample.py : Sample Any number of points and get their location, and respective pixel value
            - Supports satellite images with multiple bands

'''



import numpy as np
import pandas as pd
import json
import geojson
import random
import os

try:
    import gdal, ogr, osr
    from osgeo import ogr
except ImportError:
    raise ImportError('GDAL must be installed')

import rasterio


def sample(raster, input, output, n_samples, epsg=3310):
    '''
    :param raster: the raster data from which sample will be taken
    :param input: boundary of the sample area
    :param output: location of sample points
    :param n_samples: samples taken from the pixels
    :return: Data Frame
    '''

    # check data type
    dataType = os.path.basename(input).split('.')[1]

    if dataType == "shp":

        driver = ogr.GetDriverByName("ESRI Shapefile")
        file = driver.Open(input, 0)
        layer = file.GetLayer()
        env = layer.GetExtent()
        polygon = ogr.Geometry(ogr.wkbGeometryCollection)
        xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]


        for feature in layer:
            geom = feature.GetGeometryRef()
            ring = geom.GetGeometryRef(0)
            polygon.AddGeometry(ring)


    else:

        #get boundary values from json
        with open(input) as f:
            data = json.load(f)
        for feature in data['features']:
            geom = feature['geometry']
            geom = json.dumps(geom)
            polygon = ogr.CreateGeometryFromJson(geom)

        env = polygon.GetEnvelope()
        xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]



    # Read raster
    src_ds = gdal.Open(raster)
    geoT = src_ds.GetGeoTransform()


    num_points = n_samples
    counter = 0
    rows = []

    # write random points to vector-point
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    outDataSource = outDriver.CreateDataSource(output)
    outLayer = outDataSource.CreateLayer(output, geom_type=ogr.wkbPoint)



    for i in range(0, num_points):
        i += 1

        '''
        If random point (i) is inside the boundary: 
        store the location and extract pixel values
        '''

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(random.uniform(xmin, xmax),
                       random.uniform(ymin, ymax))


        if point.Within(polygon):
            multipoint.AddGeometry(point)
            counter += 1

            featureDefn = outLayer.GetLayerDefn()
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(point)
            outLayer.CreateFeature(outFeature)

            mx, my = point.GetX(), point.GetY()
            px = int((mx - geoT[0]) / geoT[1])
            py = int((my - geoT[3]) / geoT[5])
            xcount = 1
            ycount = 1


            ext = []

            # Extract pixels values for all bands
            for i in range(0, src_ds.RasterCount):
                i += 1
                dump = src_ds.GetRasterBand(i).ReadAsArray(px, py, xcount, ycount)
                extract = [x for x in dump if x != None]
                result = extract[0]
                ext.append(result[0])
            rows.append(ext)


    # Return data frame
    df = pd.DataFrame(rows)


    # Create projection (prj file)
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(epsg)


    spatialRef.MorphToESRI()
    prj_full_path = os.path.split(output)
    prj_path = os.path.split(output)[0]
    prj_file_name = os.path.split(output)[1]


    prj_full_path = os.path.basename(output).split('.')[0]
    k = prj_full_path + '.prj'
    file = open(k, 'w')
    r = os.path.join(prj_path, k)
    file = open(r, 'w')


    file.write(spatialRef.ExportToWkt())
    file.close()


    # write to csv
    text = prj_full_path + '.txt'
    r = os.path.join(prj_path, text)
    df.to_csv(r)



    return df




