# Poisson Randomization for subsampling with minimum distance threshold between samples
import numpy as np
import matplotlib.pyplot as plt
from tsraster.prep import image_to_array
import rasterio



def get_cell_coords(pt):
    """Get the coordinates of the cell that pt = (x,y) falls in."""

    return int(pt[0] // a), int(pt[1] // a)


def get_neighbours(coords):
    """Return the indexes of points in cells neighbouring cell at coords.

    For the cell at coords = (x,y), return the indexes of points in the cells
    with neighbouring coordinates illustrated below: ie those cells that could 
    contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

    """

    dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
            (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
            (-1,2),(0,2),(1,2),(0,0)]
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

def point_valid(pt, mask_Array):
    """Is pt a valid point to emit as a sample?

    It must be no closer than r from any other point: check the cells in its
    immediate neighbourhood.

    """

    cell_coords = get_cell_coords(pt)
    for idx in get_neighbours(cell_coords):
        nearby_pt = samples[idx]
        # Squared distance between or candidate point, pt, and this nearby_pt.
        distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
        if distance2 < r**2:
            # The points are too close, so pt is not a candidate.
            return False
    #test if point falls within mask
    if mask_Array[[int(pt[1]), int(pt[0])]]:
        return False
    # All points tested: if we're here, pt is valid
    else: return True

def get_point(k, refpt, mask_Array):
    """Try to find a candidate point relative to refpt to emit in the sample.

    We draw up to k points from the annulus of inner radius r, outer radius 2r
    around the reference point, refpt. If none of them are suitable (because
    they're too close to existing points in the sample), return False.
    Otherwise, return the pt.

    """
    i = 0
    while i < k:
        rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
        pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt, Mask_Array):
            return pt
        i += 1
    # We failed to find a suitable point in the vicinity of refpt.
    return False

def get_inital(raster_mask, mask_Array):
    '''Select a point completely at random from within potential array space (barring masked areas)'''
    i = 0
    while i < k:
        pt = (np.random.uniform(0, width), np.random.uniform(0, height))
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt, mask_Array):
            return pt
        i += 1
    return False


def Poisson_Subsample(raster_mask, k = 50, r = 50):
    '''
    Create raster of cells to be selected (populated as ones) in a raster of background value zero

    :param raster_mask: name of raster mask - provides dimensions for subsample, and also masks unusable areas - IS ASSUMED TO BE CONTIGUOUS
    :param k: number of attempts to select a point around each reference point before marking it as inactive
    :param r: minimum distance (in raster cells) between selected points 
    '''

    with rasterio.open(raster_mask) as exampleRast:
        mask_Array = exampleRast.read()
        profile = exampleRast.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=0)

    #open raster mask, get rectangular dimensions of potential subsampling area
    height, width =  rasterio.open(raster_mask).shape
    outRaster = np.zeros((height, width))
    
    # Cell side length
    a = r/np.sqrt(2)

    # Number of meta-cells in the x- and y-directions of the grid
    nx, ny = int(width / a) + 1, int(height / a) + 1

    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    # Initialize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's coordinates in the
    # samples list (or None if the cell is empty).
    cells = {coords: None for coords in coords_list}

    # Pick a random point to start with.
    pt = get_initial(raster_mask)

    samples = [pt]
    
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
        pt = get_point(k, refpt, mask_Array)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            nsamples += 1
            active.append(len(samples)-1)
            cells[get_cell_coords(pt)] = len(samples) - 1
            outRaster[int(pt[1]), int(pt[0])] = 1
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)
    return outRaster, cells