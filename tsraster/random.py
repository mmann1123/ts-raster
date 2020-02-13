'''
random.py:- Generates random points from a vector file
              Input: geojson of boundary
              output: points.shp

'''

'''
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


    with open(path) as f:

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

    outDataSource = outDriver.CreateDataSource('points.shp')
    outLayer = outDataSource.CreateLayer('points.shp', geom_type=ogr.wkbPoint)

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


spatialRef.MorphToESRI()
file = open('points.prj', 'w')
file.write(spatialRef.ExportToWkt())
file.close()
'''



###### POISSON

# Poisson Randomization for subsampling with minimum distance threshold between samples
import numpy as np
import matplotlib.pyplot as plt
from tsraster.prep import image_to_array
import rasterio
import pandas as pd



def get_cell_coords(pt, a):
    """Get the coordinates of the cell that pt = (x,y) falls in.

    :param pt: tuple of selected point coordinates (as floats, not as integers) 
    :param a: width of meta-cell (not raster array cell)
    """

    return int(pt[0] // a), int(pt[1] // a)


def get_neighbours(coords, mask_Array, cells, a):
    """Return the indexes of points in cells neighbouring cell at coords.

    For the cell at coords = (x,y), return the indexes of points in the cells
    with neighbouring coordinates illustrated below: ie those cells that could 
    contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

    :param coords: coordinates of meta-cell in which point is located
    :param mask_array: numpy array derived from raster_mask file
    :param cells: dictionary of tuples contasining coordinates for each point
    :param a: width of meta-cell (not raster array cell)
    

    """

    dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
            (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
            (-1,2),(0,2),(1,2),(0,0)]
   
    # Number of meta-cells in the x- and y-directions of the grid
    height, width = mask_Array.shape
    ny, nx = int(height / a) + 1, int(width / a) + 1
    
    neighbours = []
    for dx, dy in dxdy:
        neighbour_coords = coords[0] + dx, coords[1] + dy
        if not (0 <= neighbour_coords[0] < nx and
                0 <= neighbour_coords[1] < ny):
            # We're off the grid: no neighbours here.
            continue
        neighbour_cell = cells[neighbour_coords]
        if neighbour_cell is not None:
            # This cell is occupied: store this index of the contained point.
            neighbours.append(neighbour_cell)
    return neighbours

def point_valid(pt, mask_Array, samples, cells, r):
    """Is pt a valid point to emit as a sample?

    It must be no closer than r from any other point: check the cells in its
    immediate neighbourhood.

    :param pt: tuple of selected point coordinates (as floats, not as integers)
    :param mask_array: numpy array derived from raster_mask file
    :param cells: dictionary of tuples contasining coordinates for each point
    :param r: minimum distance (in raster cells) between selected points 
    :param a: width of meta-cell (not raster array cell)
    """

     # Cell side length
    a = r/np.sqrt(2)

    
    
    if len(samples) >0:
        cell_coords = get_cell_coords(pt, a)
        for idx in get_neighbours(cell_coords, mask_Array, cells, a):
            nearby_pt = samples[idx]
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < r**2:
                # The points are too close, so pt is not a candidate.
                return False
    #test if point falls within mask
    if mask_Array[int(pt[1]), int(pt[0])]:
        return True
    # All points tested: if we're here, pt is valid
    else: return False

def get_point(k, r, refpt, mask_Array, samples, cells):
    """Try to find a candidate point relative to refpt to emit in the sample.

    We draw up to k points from the annulus of inner radius r, outer radius 2r
    around the reference point, refpt. If none of them are suitable (because
    they're too close to existing points in the sample), return False.
    Otherwise, return the pt.
    
    :param k: number of attempts to select a point around each reference point before marking it as inactive
    :param r: minimum distance (in raster cells) between selected points 
    :param mask_array: numpy array derived from raster_mask file
    :param samples: list of tuples representing points that have already been selected and validated
    :param cells: dictionary of tuples contasining coordinates for each point
    

    """
    height, width = mask_Array.shape

    # Cell side length
    a = r/np.sqrt(2)

    i = 0
    while i < k:
        rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
        pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt, mask_Array, samples, cells,r):
            return pt
        i += 1
    # We failed to find a suitable point in the vicinity of refpt.
    return False

def get_initial(mask_Array, cells, samples, k, r):
    '''Select a point completely at random from within potential array space (barring masked areas)

    :param mask_array: numpy array derived from raster_mask file
    :param samples: list of tuples representing points that have already been selected and validated
    :param cells: dictionary of tuples contasining coordinates for each point
    :param height: height of param_mask array
    :param width: width of param_mask array
    :param k: number of attempts to select a point around each reference point before marking it as inactive
    :param r: minimum distance (in raster cells) between selected points 
    :param a: width of meta-cell (not raster array cell)
    '''
   
    height, width = mask_Array.shape

    # Cell side length
    a = r/np.sqrt(2)

    i = 0
    while i < k:
        pt = (np.random.uniform(0, width), np.random.uniform(0, height))
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt, mask_Array, samples, cells, r):
            return pt
        i += 1
    return False


def Poisson_Subsample(raster_mask, outFile, k = 50, r = 50):
    '''
Create raster of cells to be selected (populated as ones) in a raster of background value zero

:param raster_mask: name of raster mask - provides dimensions for subsample, and also masks unusable areas - 
        Remaining sample area is assumed to be contiguous
:param outFile: path and name of output mask consisting of a rater image with values of 1 for selected pixels, 
        and 0 for all other pixels
:param k: number of attempts to select a point around each reference point before marking it as inactive
:param r: minimum distance (in raster cells) between selected points 
:return:  list which includes an array of all masked & unnmasked cells, and a dictionary of all selected points.
            Also saves the a raster consisting of 0s for all non-selected points, and 1s for all selected points
            to the outFile location.
'''

    with rasterio.open(raster_mask) as exampleRast:
        mask_Array = exampleRast.read()
        profile = exampleRast.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=0)

    #convert to 2-dimensional numpy array, instead of 3 dimensional array with depth 1
    mask_Array = mask_Array[0]
    
    #open raster mask, get rectangular dimensions of potential subsampling area
    height, width =  rasterio.open(raster_mask).shape
    outRaster = np.zeros((height, width))
    
    # meta-Cell side length
    a = r/np.sqrt(2)

    # Number of meta-cells in the x- and y-directions of the grid
    nx, ny = int(width / a) + 1, int(height / a) + 1

    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    # Initialize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's coordinates in the
    # samples list (or None if the cell is empty).
    cells = {coords: None for coords in coords_list}

    #create blank list to populate with pt values as they are created
    samples = []

    # Pick a random point to start with.
    pt = get_initial(mask_Array, cells, samples, k, r)

    #add in initial point to samples list
    samples += [pt]
    
    # ... and it is active, in the sense that we're going to look for more points
    # in its neighbourhood.
    active = [0]

    nsamples = 1
    # As long as there are points in the active list, keep trying to find samples.
    while active:
        # choose a random "reference" point from the active list.
        idx = np.random.choice(active)
        refpt = samples[idx]
        # Try to pick a new point relative to the reference point.
        pt = get_point(k, r, refpt, mask_Array, samples, cells)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            nsamples += 1
            active.append(len(samples)-1)
            cells[get_cell_coords(pt, a)] = len(samples) - 1
            outRaster[int(pt[1]), int(pt[0])] = 1
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)
    
    outRaster = np.float32(outRaster)
    with rasterio.open(outFile, 'w', **profile) as subSample:
        subSample.write(outRaster, 1)

    return outRaster, cells




#### test_train
def TestTrain_GroupMaker(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups = [10], preset_GroupVar = None):
    '''
    :param combined_Data:  multivariate data for explaining target data - may be filename or csv
    :param combined_Data: target data - may be filename or csv
    :param varsToGroupBy: variable(s) on which to build groups for testing/training
    :param groupVars: variable(s) to name those groups
    :param testGroups: number of randomly assigned groups to provide for each variable
    :return: modified forms of combined_Data and target_Data that include the randomly allocated groups
            (labeled according to groupVars)
    ''' 

    if type(varsToGroupBy) is str:
        varsToGroupBy = [varsToGroupBy]
    if type(groupVars) is str:
        groupVars is [groupVars]
    if type(testGroups) is not list:
        testGroups = [testGroups]
    
    #filter out preset group vars
    varsToGroupBy = list(filter(lambda a: a!= preset_GroupVar, varsToGroupBy))
    groupVars = list(filter(lambda a: a!= preset_GroupVar, groupVars))
        
    if type(combined_Data) is str:
            combined_Data = pd.read_csv(combined_Data)
        
    if type(target_Data) is str:
        target_Data = pd.read_csv(target_Data)
    
    
    for x in range(len(varsToGroupBy)):
        
        if combined_Data.index.name == varsToGroupBy[x]:
            combined_Data.reset_index(inplace = True)

        if target_Data.index.name == varsToGroupBy[x]:
            target_Data.reset_index(inplace = True)

        
        #get single copy of unique values to group by 
        groupSelector = combined_Data.loc[:, [varsToGroupBy[x]]]
        groupSelector = groupSelector.drop_duplicates()
        groupSelector.reset_index(inplace = True, drop = True)

        #create random values
        groupSelector['random_values'] = np.random.randint(0,99999999, size=len(groupSelector))
        groupSelector["random_order"] = groupSelector.random_values.rank()


        #randomly order all values, place them into equally sized groups 
        #(in cases of a number of groups that cannot be divided evenly, the last group will include one extra value)
        groupSelector[groupVars[x]] = groupSelector.random_order.apply(lambda y: y / len(groupSelector) * testGroups[x]* 0.9999999) # the 0.999 is to prevent the top-ranked value from being placed in its own class
        groupSelector[groupVars[x]] = groupSelector[groupVars[x]].map(int)
        groupSelector.reset_index(inplace = True)

        groupSelector = groupSelector.loc[:, [varsToGroupBy[x], groupVars[x]]]
        


        combined_Data = pd.merge(combined_Data, groupSelector, on = [varsToGroupBy[x]], how = "left")
        test= combined_Data.dropna()

        target_Data = pd.merge(target_Data, groupSelector, on = [varsToGroupBy[x]], how = "left")

    if type(preset_GroupVar) == str:
        preset_GroupVar = [preset_GroupVar]
    
    if type(preset_GroupVar) == list:
        varsToGroupBy +=  preset_GroupVar
        groupVars += preset_GroupVar
        for y in range(len(preset_GroupVar)):
            testGroups += [len(list(set(combined_Data[preset_GroupVar[y]])))]
            target_Data[preset_GroupVar[y]] = combined_Data[preset_GroupVar[y]]
            
    
    return combined_Data, target_Data, varsToGroupBy, groupVars, testGroups
 

 
