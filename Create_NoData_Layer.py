import argparse
import warnings
import numpy as np
from osgeo import gdal,gdalconst,osr
import os
import subprocess
import glob
import sys
sys.path.insert(0,'../DEM')

from dem_utils import raster_to_geotiff,get_raster_extents,resample_raster


def main(args):
    args = parse_arguments(args)
    input_raster = args.raster
    output_file = args.output_file
    coastline_file = args.coastline
    buffer_val = args.buffer
    intermediate_resolution = args.resolution
    min_size = args.min_size
    # reverse_flag = args.reverse


    if not os.path.isfile(input_raster):
        raise RuntimeError(f'Input raster file does not exist: {os.path.basename(input_raster)}')
    
    src = gdal.Open(input_raster, gdalconst.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f'Could not open input raster: {os.path.basename(input_raster)}')
    
    nodata_value = src.GetRasterBand(1).GetNoDataValue()
    if nodata_value is None:
        raise RuntimeError(f'No NoData value found in raster: {os.path.basename(input_raster)}\nTip: Use gdal_edit.py -a_nodata <value> {os.path.basename(input_raster)} to set a NoData value.')

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_ext = os.path.splitext(output_file)[1]
    if output_file_ext.lower() not in ['.shp', '.geojson']:
        raise RuntimeError(f'Output file must be a shapefile or GeoJSON: {os.path.basename(output_file)}')


    output_dir = os.path.dirname(output_file)
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    output_binary_file = os.path.join(output_dir,'tmp_binary.tif')
    output_zeros_array = os.path.join(output_dir,'zeros_array.tif')
    output_binary_buffered_file = os.path.join(*[output_dir,'tmp_binary_buffered.tif'])
    output_binary_buffered_flipped_file = os.path.join(*[output_dir,'tmp_binary_buffered_flipped.tif'])
    output_binary_buffered_flipped_4326_file = os.path.join(*[output_dir,'tmp_binary_buffered_flipped_4326.tif'])
    osm_clipped_file = os.path.join(output_dir,'tmp_coast_clipped.shp')

    output_polygon_full = os.path.join(output_dir,'tmp_polygon_full.shp')
    # output_polygon_clipped = os.path.join(output_dir,'tmp_polygon_clipped.shp')
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)

    x_size = src.RasterXSize
    y_size = src.RasterYSize
    x_min = src.GetGeoTransform()[0]
    y_max = src.GetGeoTransform()[3]
    x_max = x_min + x_size * src.GetGeoTransform()[1]
    y_min = y_max + y_size * src.GetGeoTransform()[5]

    x_min_buffered = x_min - buffer_val
    x_max_buffered = x_max + buffer_val
    y_min_buffered = y_min - buffer_val
    y_max_buffered = y_max + buffer_val

    if nodata_value > 1e38:
        calc_eq = 'A < 1e38'
    elif nodata_value == 0:
        calc_eq = 'A != 0'
    elif nodata_value == -9999:
        calc_eq = 'A != -9999'
    else:
        raise ValueError(f'Unsupported nodata value: {nodata_value}')
    calc_command = f'gdal_calc.py --quiet --overwrite -A {input_raster} --calc="{calc_eq}" --outfile="{output_binary_file}" --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER"'
    subprocess.run(calc_command,shell=True,check=True)

    x_array_buffered = np.arange(x_min_buffered,x_max_buffered+intermediate_resolution,intermediate_resolution)
    y_array_buffered = np.arange(y_min_buffered,y_max_buffered+intermediate_resolution,intermediate_resolution)
    arr = np.zeros((len(y_array_buffered),len(x_array_buffered)),dtype=np.uint8)

    raster_to_geotiff(x_array_buffered,y_array_buffered,arr,epsg_code,output_zeros_array)
    resample_raster(output_binary_file,output_zeros_array,output_binary_buffered_file,resample_method='nearest',compress=True,nodata=0,quiet_flag=True,dtype='int')

    # flip_command = f'gdal_calc.py --quiet --overwrite -A {output_binary_buffered_file} --calc="-1*(A-1)" --outfile="{output_binary_buffered_flipped_file}" --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --NoDataValue=0'
    flip_command = f'gdal_calc.py --quiet --overwrite -A {output_binary_buffered_file} --calc="A==0" --outfile="{output_binary_buffered_flipped_file}" --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --type Byte'
    subprocess.run(flip_command,shell=True,check=True)


    lon_min,lon_max,lat_min,lat_max = get_raster_extents(output_binary_buffered_file,global_local_flag='global')
    # print(f'Global extent: {lon_min}, {lon_max}, {lat_min}, {lat_max}')

    osm_clip_command = f'ogr2ogr -overwrite --quiet -f "ESRI Shapefile" -clipsrc {lon_min} {lat_min} {lon_max} {lat_max} {osm_clipped_file} {coastline_file}'
    subprocess.run(osm_clip_command,shell=True,check=True)

    reproject_command = f'gdalwarp --quiet -overwrite -t_srs EPSG:4326 -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" {output_binary_buffered_flipped_file} {output_binary_buffered_flipped_4326_file}'
    subprocess.run(reproject_command,shell=True,check=True)


    if coastline_file is not None:
        polygonize_command = f'gdal_polygonize.py -overwrite -mask {output_binary_buffered_flipped_4326_file} -q {output_binary_buffered_flipped_4326_file} {output_polygon_full}'
        subprocess.run(polygonize_command,shell=True,check=True)
        clip_command = f'ogr2ogr -overwrite --quiet {output_file} {output_polygon_full} -clipsrc {osm_clipped_file}'
        subprocess.run(clip_command,shell=True,check=True)
    else:
        polygonize_command = f'gdal_polygonize.py -overwrite -mask {output_binary_buffered_flipped_4326_file} -q {output_binary_buffered_flipped_4326_file} {output_file}'
        subprocess.run(polygonize_command,shell=True,check=True)

    if min_size > 0:
        import geopandas as gpd
        gdf = gpd.read_file(output_file)
        gdf = gdf[gdf.to_crs('EPSG:3857').area > min_size].reset_index(drop=True)
        gdf.to_file(output_file)

    # Clean up temporary files
    tmp_files = []
    glob_patterns = ['tmp_binary.tif','zeros_array.tif', 'tmp_binary_buffered.tif', 'tmp_binary_buffered_flipped.tif','tmp_binary_buffered_flipped_4326.tif',
                     'tmp_coast_clipped.*','tmp_polygon_full.*']
    for p in glob_patterns:
        tmp_files.extend(glob.glob(os.path.join(output_dir, p)))
    for tmp_file in tmp_files:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster',help='Input raster',required=True)
    parser.add_argument('--output_file',help='Output file name',default='tmp_nodata.shp')
    parser.add_argument('--coastline',help='Coastline vector file',default=None)
    parser.add_argument('--buffer',help='Buffer distance in meters',default=1e4,type=float)
    parser.add_argument('--resolution',help='Intermediate spatial resolution in meters',default=10,type=float)
    parser.add_argument('--min_size',help='Minimum size of final polygon\' features',default=0,type=float)
    # parser.add_argument('--reverse',help='Reverse action, i.e. create layer where there is data',action='store_true',default=False)
    return parser.parse_args(args)


if __name__ == '__main__':
    # import sys
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    warnings.filterwarnings('ignore',category=FutureWarning)
    gdal.UseExceptions()
    main(sys.argv[1:])
