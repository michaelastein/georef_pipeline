
A pipeline to
- find all images containing a given point/anomaly
- georeference all of these images and calculate the average gps position
- display the gps position on a CAD map or on Google maps

Start the programm from the terminal with 
 `python feature_matching.py` or `python feature_matching.py -no-gui`


For only georeferencing:
`python georef_new.py`









To be able to use your images for this code, you first need to annotate the meta data from the .csv file to the images directly:

# Make sure your terminal is in the folder with images and CSV
$csvFile = "merged_tiff_pi_new.csv"

# Import the CSV (comma-separated)
$rows = Import-Csv $csvFile

foreach ($row in $rows) {

    $imageFile = $row.wiris_image.Trim()
    
    if (-not $imageFile) {
        Write-Host "Skipping row: image filename missing"
        continue
    }
    
    if (-not (Test-Path $imageFile)) {
        Write-Host "File not found: $imageFile"
        continue
    }

    # Determine GPS refs
    $latRef = if ([double]$row.Latitude -ge 0) { "N" } else { "S" }
    $lonRef = if ([double]$row.Longitude -ge 0) { "E" } else { "W" }

    # Build ImageDescription string
    $description = "Yaw=$($row.GimbalYawE), Pitch=$($row.pitch_agisoft), Roll=$($row.roll), RelativeAlt=$($row.CHeight)"

    Write-Host "Processing $imageFile ..."

    # Annotate image with ExifTool (wrap numeric values in quotes)
    exiftool `
        "-GPSLatitude=$($row.Latitude)" `
        "-GPSLatitudeRef=$latRef" `
        "-GPSLongitude=$($row.Longitude)" `
        "-GPSLongitudeRef=$lonRef" `
        "-GPSAltitude=$($row.alt)" `
        "-GPSAltitudeRef=0" `
        "-ImageDescription=$description" `
        -overwrite_original "$imageFile"
}
