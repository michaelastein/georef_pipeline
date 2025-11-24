
A pipeline to
- find all images containing a given point/anomaly
- georeference all of these images and calculate the average gps position
- display the gps position on a CAD map or on Google maps




Start the programm from the terminal with 
`python feature_matching.py --anomalies batch`


For only georeferencing:
`python georef_new.py`




Important!
It really speeds up to downsize the images, for example a width of 500px is more than enough.
But then you need to give the original image size in the terminal, otherwise the anomaly matching won't work.
`python feature_matching.py --anomalies batch --image-size 1200 700`

If you have the image log .csv file in the same folder as the images, it will be automatically chosen (if it is the only .csv file in the folder --> keep the anomaly csv file in a different folder)







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



Example Command

Launch the tool with SIFT feature matching, a single anomaly GUI, and the original image size:
`python feature_matching.py --algorithm SIFT --anomalies single --image-size 4000 3000`





Notes

- GUI pixel selections are automatically scaled to the original image size if specified.

- Manual coordinate input in the GUI should be in the original image size.

