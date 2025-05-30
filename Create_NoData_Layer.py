import argparse
import warnings
import numpy as np
from osgeo import gdal,gdalconst,osr
import os
import subprocess
import glob
import geopandas as gpd

'''

'''

def get_extent(gt,cols,rows):
    '''
    Return list of corner coordinates from a geotransform
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    '''
    Reproject a list of x,y coordinates.
    x and y are in src coordinates, going to tgt
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def get_raster_extents(raster,global_local_flag='global'):
    '''
    Get global or local extents of a raster
    '''
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
    local_ext = get_extent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if src_srs.GetAttrValue('AUTHORITY',1) == '4326':
        global_ext = local_ext
    else:
        global_ext = reproject_coords(local_ext,src_srs,tgt_srs)
    x_local = [item[0] for item in local_ext]
    y_local = [item[1] for item in local_ext]
    x_min_local = np.nanmin(x_local)
    x_max_local = np.nanmax(x_local)
    y_min_local = np.nanmin(y_local)
    y_max_local = np.nanmax(y_local)
    x_global = [item[0] for item in global_ext]
    y_global = [item[1] for item in global_ext]
    x_min_global = np.nanmin(x_global)
    x_max_global = np.nanmax(x_global)
    y_min_global = np.nanmin(y_global)
    y_max_global = np.nanmax(y_global)
    if global_local_flag.lower() == 'global':
        return x_min_global,x_max_global,y_min_global,y_max_global
    elif global_local_flag.lower() == 'local':
        return x_min_local,x_max_local,y_min_local,y_max_local
    else:
        return None


def buffer_gdf(gdf,buffer_distance,cap_style='square',join_style='mitre',resolution=4):
    '''
    Buffer a gdf by a given distance
    '''
    gdf_3857 = gdf.to_crs('EPSG:3857')
    gdf_3857_buffered = gdf_3857.buffer(buffer_distance,cap_style=cap_style,join_style=join_style,resolution=resolution)
    gdf_buffered = gdf_3857_buffered.to_crs('EPSG:4326')
    return gdf_buffered


def main(args):
    args = parse_arguments(args)
    input_raster = args.raster
    output_file = args.output_file
    coastline_file = args.coastline
    buffer_distance = args.buffer
    intermediate_res = args.resolution
    reverse_flag = args.reverse

    if not os.path.isfile(input_raster):
        raise RuntimeError(f'Input raster file does not exist: {os.path.basename(input_raster)}')
    
    src = gdal.Open(input_raster, gdalconst.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f'Could not open input raster: {os.path.basename(input_raster)}')
    
    nodata_value = src.GetRasterBand(1).GetNoDataValue()
    if nodata_value is None:
        raise RuntimeError(f'No NoData value found in raster: {os.path.basename(input_raster)}')

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_ext = os.path.splitext(output_file)[1]
    if output_file_ext.lower() not in ['.shp', '.geojson']:
        raise RuntimeError(f'Output file must be a shapefile or GeoJSON: {os.path.basename(output_file)}')

    raster_resolution = np.max(np.abs([src.GetGeoTransform()[1],src.GetGeoTransform()[5]]))


    if reverse_flag == True:
        calc_str = f'A == {nodata_value}'
    else:
        calc_str = f'A != {nodata_value}'
    calc_command = f'gdal_calc.py --quiet --overwrite -A {input_raster} --calc="{calc_str}" --outfile="{os.path.join(*[output_dir,"tmp_binary.tif"])}" --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER"'
    subprocess.run(calc_command,shell=True,check=True)
    
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)
    if epsg_code not in ['4326', '4269']:
        if raster_resolution < intermediate_res:
            resample_command = f'gdalwarp --quiet -overwrite -tr {intermediate_res} {intermediate_res} {os.path.join(*[output_dir,"tmp_binary.tif"])} {os.path.join(*[output_dir,"tmp_binary_resampled.tif"])} -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER"'
            subprocess.run(resample_command,shell=True,check=True)
        else:
            os.rename(os.path.join(*[output_dir,"tmp_binary.tif"]), os.path.join(*[output_dir,"tmp_binary_resampled.tif"]))
        reproject_command = f'gdalwarp --quiet -overwrite -t_srs EPSG:4326 -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" {os.path.join(*[output_dir,"tmp_binary_resampled.tif"])} {os.path.join(*[output_dir,"tmp_binary_resampled_4326.tif"])}'
        subprocess.run(reproject_command,shell=True,check=True)
    else:
        os.rename(os.path.join(*[output_dir,"tmp_binary.tif"]), os.path.join(*[output_dir,"tmp_binary_resampled_4326.tif"]))

    if coastline_file is not None:
        #If coastline clip is requested we need to create an intermediate file first
        tmp_output_file = f'{os.path.join(*[output_dir,"tmp_outline"])}{output_file_ext}'
        polygonize_command = f'gdal_polygonize.py -q -overwrite {os.path.join(*[output_dir,"tmp_binary_resampled_4326.tif"])} {tmp_output_file}'
        subprocess.run(polygonize_command,shell=True,check=True)
        lon_min,lon_max,lat_min,lat_max = get_raster_extents(os.path.join(*[output_dir,"tmp_binary_resampled_4326.tif"]),global_local_flag='global')
        buffer_distance_degrees = buffer_distance * 1.1 / (6378137 * np.pi / 180) #add 10% for buffer
        clip_coast_command = f'ogr2ogr --quiet {os.path.join(*[output_dir,"tmp_coast.shp"])} {coastline_file} -clipsrc {lon_min-buffer_distance_degrees} {lat_min-buffer_distance_degrees} {lon_max+buffer_distance_degrees} {lat_max+buffer_distance_degrees}'
        subprocess.run(clip_coast_command,shell=True,check=True)
        if buffer_distance > 0:
            gdf = gpd.read_file(tmp_output_file)
            gdf_buffered = buffer_gdf(gdf, buffer_distance)
            gdf_buffered.to_file(tmp_output_file)
        clip_raster_command = f'ogr2ogr --quiet {output_file} {os.path.join(*[output_dir,"tmp_outline"])}{output_file_ext} -clipsrc {os.path.join(*[output_dir,"tmp_coast.shp"])}'
        subprocess.run(clip_raster_command,shell=True,check=True)
    else:
        #If coastline clip is not requested we can polygonize directly to requested output file
        polygonize_command = f'gdal_polygonize.py -q -overwrite {os.path.join(*[output_dir,"tmp_binary_resampled_4326.tif"])} {output_file}'
        subprocess.run(polygonize_command,shell=True,check=True)
        if buffer_distance > 0:
            gdf = gpd.read_file(output_file)
            gdf_buffered = buffer_gdf(gdf, buffer_distance)
            gdf_buffered.to_file(output_file)
    
    # Clean up temporary files
    tmp_files = []
    glob_patterns = ['tmp_binary.tif', 'tmp_binary_resampled.tif', 'tmp_binary_resampled_4326.tif', 'tmp_coast.*','tmp_outline.*']
    for p in glob_patterns:
        tmp_files.extend(glob.glob(os.path.join(output_dir, p)))
    for tmp_file in tmp_files:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster',help='Input raster',required=True)
    parser.add_argument('--output_file',help='Output file name',default='tmp.shp')
    parser.add_argument('--coastline',help='Coastline vector file',default=None)
    parser.add_argument('--buffer',help='Buffer distance in meters',default=1e3,type=float)
    parser.add_argument('--resolution',help='Intermediate spatial resolution in meters',default=10,type=float)
    parser.add_argument('--reverse',help='Reverse action, i.e. create layer where there is data',action='store_true',default=False)
    return parser.parse_args(args)


if __name__ == '__main__':
    import sys
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    warnings.filterwarnings('ignore',category=FutureWarning)
    gdal.UseExceptions()
    main(sys.argv[1:])
