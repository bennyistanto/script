library(raster)
library(ncdf4)
prec <- getData("worldclim", res = 10, var = "prec")
tmax <- getData("worldclim", res = 10, var = "tmax")
tmax[] <- tmax[]/10 # tmax is scaled by 10
# Change NA values to -999
prec[is.na(prec)] <- tmax[is.na(tmax)] <- -999

# make sure the rasters are identical
compareRaster(prec, tmax, res = TRUE, orig = TRUE) ## TRUE

# output filename
filename <- "testMultiDim.nc"

# Longitude and Latitude data
xvals <- unique(values(init(prec, "x")))
yvals <- unique(values(init(prec, "y")))
nx <- length(xvals)
ny <- length(yvals)
lon <- ncdim_def("longitude", "degrees_east", xvals)
lat <- ncdim_def("latitude", "degrees_north", yvals)

# Missing value to use
mv <- -999

# Time component
time <- ncdim_def(name = "Time", 
                  units = "months", 
                  vals = 1:12, 
                  unlim = TRUE,
                  longname = "Month_of_year")

# Define the precipitation variables
var_prec <- ncvar_def(name = "precipitation",
                      units = "mm/month",
                      dim = list(lon, lat, time),
                      longname = "Monthly_Total_Precipitation",
                      missval = mv,
                      compression = 9)

# Define the temperature variables
var_temp <- ncvar_def(name = "temperature",
                      units = "degC",
                      dim = list(lon, lat, time),
                      longname = "Monthly_Avg_Temperature_degC",
                      missval = mv,
                      compression = 9)

# Add the variables to the file
ncout <- nc_create(filename, list(var_prec, var_temp), force_v4 = TRUE)
print(paste("The file has", ncout$nvars,"variables"))
print(paste("The file has", ncout$ndim,"dimensions"))

# add some global attributes
ncatt_put(ncout, 0, "Title", "MultiDimesionsalNCDFTest")
ncatt_put(ncout, 0, "Source", "Some example data from the raster package")
ncatt_put(ncout, 0, "References", "See the raster package")
ncatt_put(ncout, 0, "Created on", date())

# Place the precip and tmax values in the file
# need to loop through the layers to get them 
# to match to correct time index
for (i in 1:nlayers(prec)) { 
  #message("Processing layer ", i, " of ", nlayers(prec))
  ncvar_put(nc = ncout, 
            varid = var_prec, 
            vals = values(prec[[i]]), 
            start = c(1, 1, i), 
            count = c(-1, -1, 1))
  ncvar_put(ncout, var_temp, values(tmax[[i]]), 
            start = c(1, 1, i), 
            count = c(-1, -1, 1))
}
# Close the netcdf file when finished adding variables
nc_close(ncout)
