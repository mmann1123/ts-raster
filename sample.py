#############################################################################################
# sample.py : Sample Any number of points and get their location, and respective pixel value
#             - Supports satellite images with multiple bands
# Aug-26-2018
# author: Adane(Eddie) Bedada
# @adbe.gwu.edu
#############################################################################################


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
    raise ImportError('OGR must be installed')

import rasterio


def sample(raster, input, output, n_samples):
    '''

    :param raster: the raster data from which sample will be taken
    :param input: boundary of the sample area
    :param output: location of sample points
    :param n_samples: samples taken from the pixels
    :return: Data Frame
    '''

    src_ds = gdal.Open(raster)
    geoT = src_ds.GetGeoTransform()


    with open(input) as f:
        data = json.load(f)
    for feature in data['features']:
        geom = feature['geometry']
        geom = json.dumps(geom)
        polygon = ogr.CreateGeometryFromJson(geom)


    env = polygon.GetEnvelope()
    xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]

    num_points = n_samples
    counter = 0


    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    outDataSource = outDriver.CreateDataSource(output)
    outLayer = outDataSource.CreateLayer(output, geom_type=ogr.wkbPoint)


    for i in range(0, num_points):
        i += 1

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

            outFeature = None
        outDataSource = None

        # To do: change hard cord on spatialRef
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(3310)

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


        mx, my = point.GetX(), point.GetY()
        px = int((mx - geoT[0]) / geoT[1])
        py = int((my - geoT[3]) / geoT[5])

        ext = []


        for i in range(0, src_ds.RasterCount):
            i += 1
            extracted = src_ds.GetRasterBand(i).ReadAsArray(px, py, 1, 1)
            extracted != None
            result = pd.DataFrame(extracted)
            ext.append(result)


        df = pd.DataFrame(ext, index=None).T
        text = prj_full_path + '.csv'
        df.to_csv(text)

        # to do: update return values
        return (df)

