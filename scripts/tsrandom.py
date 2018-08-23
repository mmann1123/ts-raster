######################################################################$$$$$$$$$$$$$$$##
#  tsrandom.py:- Generates any number of points from a vector file

# Aug-21-2018
# author: Adane(Eddie) Bedada
# @adbe.gwu.edu
#######################################################################################



import numpy as np
import pandas as pd
import json
import geojson
import random
try:
    import ogr, osr
    from osgeo import ogr
except ImportError:
    raise ImportError('OGR must be installed')


with open('/Users/adbe/mmann/Demo_Data/shp/California.geojson') as f:

    data = json.load(f)
for feature in data['features']:
    geom = feature['geometry']
    geom = json.dumps(geom)
    polygon = ogr.CreateGeometryFromJson(geom)

env = polygon.GetEnvelope()
xmin, ymin, xmax, ymax = env[0],env[2],env[1],env[3]

num_points = 1000

counter = 0

multipoint = ogr.Geometry(ogr.wkbMultiPoint)
outDriver = ogr.GetDriverByName('ESRI Shapefile')

outDataSource = outDriver.CreateDataSource('Cali-pts-prj.shp')
outLayer = outDataSource.CreateLayer('Cali-pts-prj.shp', geom_type=ogr.wkbPoint)



for i in range(num_points):
    while counter < num_points:

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


spatialRef = osr.SpatialReference()
spatialRef.ImportFromEPSG(3310)

spatialRef.MorphToESRI()
file = open('Cali-pts-prj.prj', 'w')
file.write(spatialRef.ExportToWkt())
file.close()