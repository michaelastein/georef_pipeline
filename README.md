
A pipeline to
- find all images containing a given point/anomaly
- georeference all of these images and calculate the average gps position
- display the gps position on a CAD map or on Google maps




Start the programm from the terminal with 
`python feature_matching.py --anomalies batch`


For only georeferencing:
`python georeferncing.py`




**Important!**
It really speeds up to downsize the images, for example a width of 500px is more than enough.
But then you need to give the original image size in the terminal, otherwise the anomaly matching won't work (as the anomaly algorithm was run on the original size -> the logs contain pixels corresponding to the original sizes).
`python feature_matching.py --anomalies batch --image-size 1200 700`

If you have the image log .csv file in the same folder as the images, it will be automatically chosen (if it is the only .csv file in the folder --> keep the anomaly csv file in a different folder)


The Georeferncing uses camera parameters of the WIRIS IR Camera. If you want to use images from another camera, please change focal_length_mm and sensor_width in georeferencing.py

The calculations (how the images relate to each other & feature matching) will be automatically saved to homographies.pkl, so you can reload them later with --homographies homographies.pkl.






This tool allows you to detect and match features in images and optionally handle anomalies. You can either select a single anomaly manually using a GUI or process multiple anomalies automatically from a CSV file. The tool supports scaling for resized images and can integrate Digital Elevation Models (DEM) and LiDAR data to improve georeferencing and altitude calculations.

`python feature_matching.py [OPTIONS]`


Parameters

--algorithm
    Choose the feature detection/matching algorithm.
    Options: SIFT, AKAZE, ORB, BRISK, KAZE
    Default: None (script may use an internal default)
    Example:    --algorithm SIFT



-a, --anomalies
    Specify anomaly handling mode.
    Options:
    none – Do not process anomalies (default).
    single – Launch a GUI to manually select a single anomaly in an image.
    batch – Process multiple anomalies automatically from a CSV.

    Example:    --anomalies single

-H, --homographies
    Load precomputed homographies and image metadata from a file. ( So you don't need to do the heavy computations again for the same image files)
    Default: None
    Example: --homographies homographies.pkl
    If the file is not in the same folder as the feature_matching.py, use the full path, e.g. "C:\Users\Micha\internship\georef\homographies.pkl"


-d, --dem
    Load a Digital Elevation Model (DEM) file for georeferencing or altitude calculations.
    Default: None
    Example:  --dem terrain.dem
    If the file is not in the same folder as the feature_matching.py, use the full path, e.g. "C:\Users\Micha\internship\georef\terrain.dem"



-l, --lidar
    Load a LiDAR LAZ file to improve GPS or altitude estimates.
    Default: None
    Example: --lidar data.laz
    If the file is not in the same folder as the feature_matching.py, use the full path, e.g. "C:\Users\Micha\internship\georef\data.laz"


-i, --image-size
    Specify the original size of the images (width × height).
    Needed if images were resized, to correctly scale coordinates from CSV.
    Input: Two integers separated by a space: WIDTH HEIGHT
    Example: --image-size 4000 3000


-c, --cad
    Path to the CAD file (GeoJSON) of the solar plant. This is used for plotting or georeferencing anomalies relative to the plant layout.
    Default: None
    Example:  --cad solar_plant.geojson
    If the file is not in the same folder as the feature_matching.py, use the full path, e.g. "C:\Users\Micha\internship\georef\solar_plant.geojson"





Example Command

Launch the tool with SIFT feature matching, a single anomaly GUI, and the original image size:
`python feature_matching.py --algorithm SIFT --anomalies single --image-size 4000 3000`





Notes

- GUI pixel selections are automatically scaled to the original image size if specified.

- Manual pixel coordinate input in the GUI should be in the original image size (as in the anomaly logs).

- Possible further improvements to implement:

    - flying the drone with LIDAR to get a more accurate heigth above ground. I think this will have a great impact on acuracy of the georeferencing
    - find the right image size that balances computation speed and quality of results. Reducing the input image size has a great impact on computation duration of the         homography. Don't forget to add the original image size as a parameter in the terminal if you sized the images down
    - Use the GPU for some of the calculations to speed things up
    - further parallelize the anomaly and anomaly batch processing, as it uses only a small fraction of available CPU at the moment 
    - Take the height of the solar panels above ground into account, maybe with LIDAR scans from the drone
    - take into account the physical distance and possible rotation between camera, imu, GPS and LIDAR to make it more accurate; consider the gimbal
    - put more effort into finding out the best feature matching algorithm, I only decided on some examplary results I've seen
    - remove the images from when the drone turns


