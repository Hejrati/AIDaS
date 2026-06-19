#
#rm(list = ls())
############################
### USING THIS SCRIPT... ###
############################
#
#
#
#####
# 1 #
#####
# Enter the name of each ANALYZE file here.
#  Omit the ".hdr"/".img" extension.
# You should have four ANALYZE images:
#
#  LIGHT, DARK, LIGHT_MARKED, and DARK_MARKED.
#  Each _MARKED image (each slice of each image) should:
#     (1) be 8-bit, and max set to 230, other than:
#     (2) have the RPE marked as 255
#     (3) have a line showing the center of the fovea marked as 243
#  Moreover, the first slice of the DARK_MARKED image should have the following marked:
#      254 is the ELM
#      253 is the ONL/OPL border
#      252 is the INL/IPL border
#      250 is the RNFL/GCL border
#      249 is the RNFL/vitreous border
#
# IMAGE.INDEX.LIGHT and IMAGE.INDEX.DARK should list the order of images.
# if, for instance, there were 5 dark images taken with no omissions,
# then IMAGE.INDEX.DARK=c(1,2,3,4,5)
# but images 1 and 4 of 5 was removed from the # ANALYZE files, 
# (e.g., due to low signal-to-noise), then IMAGE.INDEX.DARK=c(2,3,5)
# the number of items in this list has to match the number of slices in the
# respective ANALYZE file.
# 
############
############

.DEBUG.STEP <- "startup"
dbg <- function(step, ...) {
  .DEBUG.STEP <<- step
  cat(paste0("DEBUG [", step, "] ", paste(..., collapse=" "), "\n"))
  flush.console()
}

stop.at.boundary <- function(step, ...) {
  dbg(step, ...)
  stop("Execution stopped at current translated boundary.", call.=FALSE)
}

format.number <- function(x) {
  sprintf("%.6f", as.numeric(x))
}

show.scalar.stats <- function(name, value) {
  cat(paste0("STAT ", name, ": type=", class(value)[1], " value=", paste(value, collapse=","), "\n"))
}

show.vector.stats <- function(name, values) {
  cat(paste0(
    "STAT ", name, ": type=int_vector length=", length(values),
    " values=", paste(values, collapse=","),
    " min=", format.number(min(values)),
    " max=", format.number(max(values)),
    " mean=", format.number(mean(values)),
    " sum=", format.number(sum(values)),
    "\n"
  ))
}

show.array.stats <- function(name, value) {
  vals <- as.numeric(value)
  cat(paste0(
    "STAT ", name, ": type=array class=", class(value)[1],
    " dim=", paste(dim(value), collapse="x"),
    " min=", format.number(min(vals, na.rm=TRUE)),
    " max=", format.number(max(vals, na.rm=TRUE)),
    " mean=", format.number(mean(vals, na.rm=TRUE)),
    " sum=", format.number(sum(vals, na.rm=TRUE)),
    " na=", sum(is.na(value)),
    "\n"
  ))
}

options(error = function() {
  cat(paste0("ERROR: failure at step '", .DEBUG.STEP, "'\n"))
  traceback(3)
  stop("Execution stopped after debug traceback.")
})

dbg("startup", "Script started")


args <- commandArgs(trailingOnly=TRUE)

arg.or.env <- function(position, env.name, default="") {
  value <- ""
  if(length(args) >= position) value <- args[[position]]
  if(!nzchar(value)) value <- Sys.getenv(env.name, unset="")
  if(!nzchar(value)) value <- default
  value
}

strip.analyze.extension <- function(value) {
  sub("\\.(hdr|img)$", "", value, ignore.case=TRUE)
}

parse.index <- function(value) {
  if(!nzchar(value)) return(NULL)
  parsed <- as.integer(strsplit(value, ",", fixed=TRUE)[[1]])
  parsed <- parsed[!is.na(parsed)]
  if(!length(parsed)) return(NULL)
  parsed
}

INPUTDIR <- arg.or.env(1, "AIDAS_STEP3_INPUT_DIR", getwd())
OUTDIR <- arg.or.env(2, "AIDAS_STEP3_OUTPUT_DIR", INPUTDIR)
INPUTDIR <- normalizePath(INPUTDIR, mustWork=TRUE)
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)
OUTDIR <- normalizePath(OUTDIR, mustWork=TRUE)
setwd(INPUTDIR)

REFERENCE.DARK=strip.analyze.extension(arg.or.env(3, "AIDAS_REFERENCE_DARK", "DARK_MARKED"))
REFERENCE.LIGHT=strip.analyze.extension(arg.or.env(4, "AIDAS_REFERENCE_LIGHT", "LIGHT_MARKED"))
TO.PROCESS.DARK=strip.analyze.extension(arg.or.env(5, "AIDAS_TO_PROCESS_DARK", "DARK"))
TO.PROCESS.LIGHT=strip.analyze.extension(arg.or.env(6, "AIDAS_TO_PROCESS_LIGHT", "LIGHT"))
IMAGE.INDEX.LIGHT=parse.index(arg.or.env(7, "AIDAS_IMAGE_INDEX_LIGHT", ""))
IMAGE.INDEX.DARK=parse.index(arg.or.env(8, "AIDAS_IMAGE_INDEX_DARK", ""))
PIXEL.WIDTH=as.numeric(arg.or.env(9, "AIDAS_PIXEL_WIDTH", "3.89")) #<-- microns per pixel
PYEXPORTDIR <- file.path(OUTDIR, "step3_r_arrays")
dir.create(PYEXPORTDIR, showWarnings=FALSE, recursive=TRUE)

write.python.array <- function(name, value) {
  if(is.null(value)) return(invisible(FALSE))
  target <- file.path(PYEXPORTDIR, paste0(name, ".bin"))
  dims <- dim(value)
  if(is.null(dims)) dims <- length(value)
  con <- file(target, "wb")
  on.exit(close(con), add=TRUE)
  writeBin(as.numeric(value), con, size=8, endian="little")
  writeLines(paste(as.integer(dims), collapse=","), file.path(PYEXPORTDIR, paste0(name, ".shape")))
  invisible(TRUE)
}

export.python.core.arrays <- function() {
  dbg("python-export", "Writing core Python-readable R arrays")
  write.python.array("FLATTENED_DARK_RETINA_RRC", FLATTENED.DARK.RETINA.RRC)
  write.python.array("FLATTENED_LIGHT_RETINA_RRC", FLATTENED.LIGHT.RETINA.RRC)
  write.python.array("FLATTENED_MARKERS_RRC", FLATTENED.MARKERS.RRC)
  write.python.array("FIRST_GRAND_MEAN", FIRST.GRAND.MEAN)
  write.python.array("SECOND_GRAND_MEAN", SECOND.GRAND.MEAN)
  write.python.array("FINAL_GRAND_MEAN", FINAL.GRAND.MEAN)
  write.python.array("GRAND_PROFILE", GRAND.PROFILE)
  write.python.array("APPARENT_ANGLES_FOR_DARK", APPARENT.ANGLES.FOR.DARK)
  write.python.array("APPARENT_ANGLES_FOR_LIGHT", APPARENT.ANGLES.FOR.LIGHT)
  write.python.array("SHIFT_POSITION_DARK", SHIFT.POSITION.DARK)
  write.python.array("SHIFT_POSITION_LIGHT", SHIFT.POSITION.LIGHT)
  write.python.array("SHIFT_POSITION_DARK_REFINED", SHIFT.POSITION.DARK.REFINED)
  write.python.array("SHIFT_POSITION_LIGHT_REFINED", SHIFT.POSITION.LIGHT.REFINED)
  write.python.array("BEST_LAT_MOVE_DARK", BEST.LAT.MOVE.DARK)
  write.python.array("BEST_LAT_MOVE_LIGHT", BEST.LAT.MOVE.LIGHT)
  write.python.array("VERTEX", vertex)
}

export.python.final.arrays <- function() {
  dbg("python-export", "Writing final Python-readable R arrays")
  write.python.array("FLATTENED_DARK_RETINA_RRC_N", FLATTENED.DARK.RETINA.RRC.N)
  write.python.array("FLATTENED_LIGHT_RETINA_RRC_N", FLATTENED.LIGHT.RETINA.RRC.N)
  write.python.array("FLATTENED_DARK_RETINA_RRC_N_PROFILES", FLATTENED.DARK.RETINA.RRC.N.profiles)
  write.python.array("FLATTENED_LIGHT_RETINA_RRC_N_PROFILES", FLATTENED.LIGHT.RETINA.RRC.N.profiles)
  write.python.array("FLATTENED_DARK_RETINA_RRC_N_FOVEA_PROFILES", FLATTENED.DARK.RETINA.RRC.N.fovea.profiles)
  write.python.array("FLATTENED_LIGHT_RETINA_RRC_N_FOVEA_PROFILES", FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles)
}

save.tissue.border.plot <- function(filename, flattened, rpe, olm, onl.opl, inl.ipl, rnfl.gcl, vitreous.retina) {
  target <- file.path(OUTDIR, filename)
  x.axis <- seq(-100, 2750, 1)
  y.axis <- seq(-30, 430, 1)
  n <- min(length(x.axis), nrow(rpe), nrow(olm), nrow(onl.opl), nrow(inl.ipl), nrow(rnfl.gcl), nrow(vitreous.retina))

  png(filename=target, width=1600, height=520)
  on.exit(dev.off(), add=TRUE)
  image(
    as.vector(x.axis),
    as.vector(y.axis),
    as.matrix(flattened[, dim(flattened)[2]:1, 1]),
    xlab="Distance from Fovea (microns)",
    ylab="Distance from RPE (microns)",
    col=gray.colors(254)
  )
  matlines(x.axis[1:n], 431 - rpe[1:n, 1], col="red")
  matlines(x.axis[1:n], 431 - olm[1:n, 1], col="blue")
  matlines(x.axis[1:n], 431 - onl.opl[1:n, 1], col="red")
  matlines(x.axis[1:n], 431 - inl.ipl[1:n, 1], col="blue")
  matlines(x.axis[1:n], 431 - rnfl.gcl[1:n, 1], col="red")
  matlines(x.axis[1:n], 431 - vitreous.retina[1:n, 1], col="blue")
  invisible(target)
}


dbg("input-config", "Input directory:", INPUTDIR)
dbg("input-config", "Output directory:", OUTDIR)
dbg(
  "input-config",
  "length(IMAGE.INDEX.LIGHT)=",
  if(is.null(IMAGE.INDEX.LIGHT)) "auto" else length(IMAGE.INDEX.LIGHT),
  "length(IMAGE.INDEX.DARK)=",
  if(is.null(IMAGE.INDEX.DARK)) "auto" else length(IMAGE.INDEX.DARK)
)

########################################################################################
########################################################################################
########################################################################################
####################
### REQUIREMENTS ###
####################
#
# You'll need to install the AnalyzefMRI package and 
# the RNiftyReg package, if you haven't already.
# To see if you have the former, try running the this line of code:
library(AnalyzeFMRI)
# If there was no error, you're good to go.  
#
# Otherwise,
# to install AnalyzefMRI package, go to Packages>>Install Packages>>
# and pick a location, then scroll around to find "AnalyzefMRI"
# (you'll need to be connected to the internet to do this).
#
# ... same procedure for the other library:
library(RNiftyReg)
# 
############
############


########################################################################################
########################################################################################
########################################################################################
#
#
#######################################################
#
#
########### BELOW HERE IS THE REST OF THE CODE; 
########### HOPEFULLY NO NEED TO MODIFY ANYTHING BELOW.
#
#
#######################################################
#
#
########################################################################################
########################################################################################
########################################################################################

## this variable impacts how "bendy" the fit to the manually-defined RPE is allowed to be. 30 is nice. 10 is fine
DFonINITIALspline=10

## this variable impacts how "bendy" the next adjustment/flattening is after the first pass of linearization.
DFforSECONDfit=10


#
# load everything; sometimes it loads as four-dimensional instead of three-dimensional; correct this
dbg("load-images", "Loading Analyze volumes")
REF.DARK<-f.read.analyze.volume(paste(REFERENCE.DARK,".hdr",sep=""))
REF.LIGHT<-f.read.analyze.volume(paste(REFERENCE.LIGHT,".hdr",sep=""))
DARK<-f.read.analyze.volume(paste(TO.PROCESS.DARK,".hdr",sep=""))
LIGHT<-f.read.analyze.volume(paste(TO.PROCESS.LIGHT,".hdr",sep=""))
if(length(dim(REF.DARK))==4) REF.DARK=REF.DARK[,,,1]
if(length(dim(REF.LIGHT))==4) REF.LIGHT=REF.LIGHT[,,,1]
if(length(dim(DARK))==4) DARK=DARK[,,,1]
if(length(dim(LIGHT))==4) LIGHT=LIGHT[,,,1]
dbg("load-images", "REF.DARK dim:", paste(dim(REF.DARK), collapse="x"), "REF.LIGHT dim:", paste(dim(REF.LIGHT), collapse="x"))
dbg("load-images", "DARK dim:", paste(dim(DARK), collapse="x"), "LIGHT dim:", paste(dim(LIGHT), collapse="x"))
if(is.null(IMAGE.INDEX.LIGHT)) IMAGE.INDEX.LIGHT=seq(1, dim(LIGHT)[3], 1)
if(is.null(IMAGE.INDEX.DARK)) IMAGE.INDEX.DARK=seq(1, dim(DARK)[3], 1)
if(length(IMAGE.INDEX.LIGHT) != dim(LIGHT)[3]) {
  stop(paste0("IMAGE.INDEX.LIGHT length ", length(IMAGE.INDEX.LIGHT), " does not match LIGHT slices ", dim(LIGHT)[3], "."))
}
if(length(IMAGE.INDEX.DARK) != dim(DARK)[3]) {
  stop(paste0("IMAGE.INDEX.DARK length ", length(IMAGE.INDEX.DARK), " does not match DARK slices ", dim(DARK)[3], "."))
}
dbg("variable-stats", "Printing summary statistics for current variables")
show.scalar.stats("REFERENCE.DARK", REFERENCE.DARK)
show.scalar.stats("REFERENCE.LIGHT", REFERENCE.LIGHT)
show.scalar.stats("TO.PROCESS.DARK", TO.PROCESS.DARK)
show.scalar.stats("TO.PROCESS.LIGHT", TO.PROCESS.LIGHT)
show.scalar.stats("PIXEL.WIDTH", PIXEL.WIDTH)
show.scalar.stats("DFonINITIALspline", DFonINITIALspline)
show.scalar.stats("DFforSECONDfit", DFforSECONDfit)
show.vector.stats("IMAGE.INDEX.LIGHT", IMAGE.INDEX.LIGHT)
show.vector.stats("IMAGE.INDEX.DARK", IMAGE.INDEX.DARK)
show.array.stats("REF.DARK", REF.DARK)
show.array.stats("REF.LIGHT", REF.LIGHT)
show.array.stats("DARK", DARK)
show.array.stats("LIGHT", LIGHT)
########################################################
## right now, DARK[,2,] refers to y=2 of the first slice
## right now, DARK[2,,] refers to x=2 of the first slice
########################################################

#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

#
# next, we will work through each image and flatten it out.
# before we make a loop, let's do this in detail with
# the first dark slice; for which we'll also linearize the 
# fully marked-off retina (with each layer). 

#################################################
#################################################

#
# first, find the center of the fovea
R=REF.DARK[,,1]
Xs=R
insert=seq(1,nrow(R),1)
for(x in 1:ncol(Xs)) Xs[,x]=insert
Ys=R
insert=seq(1,ncol(R),1)
for(x in 1:nrow(Ys)) Ys[x,]=insert
rm(insert)

#
# NA-out unmarked space, and multiply
R[which(R<243)]=NA
R[which(R>243)]=NA
# set marked space to 1
R[which(R==243)]=1
#
Xcoords=Xs*R
Ycoords=Ys*R
fovea.line=cbind(Xcoords[which(!(is.na(Xcoords)))],Ycoords[which(!(is.na(Xcoords)))])
fovea.line=fovea.line[order(fovea.line[,1]),]
## just in case it's a perfectly vertical line....
if(length(unique(fovea.line[,1]))==1) fovea.line[1,1]=fovea.line[1,1]+0.1
dbg("fovea-center", "Translating the first R block for fovea center detection")
dbg("variable-stats", "Printing summary statistics for translated fovea variables")
show.array.stats("R", R)
show.array.stats("Xs", Xs)
show.array.stats("Ys", Ys)
show.array.stats("Xcoords", Xcoords)
show.array.stats("Ycoords", Ycoords)
show.array.stats("fovea.line", fovea.line)


#
# repeat that, but to find the RPE
R=REF.DARK[,,1]
R[which(R<255)]=NA
R[which(R>255)]=NA
R[which(R==255)]=1
Xcoords=Xs*R
Ycoords=Ys*R
RPE.line=cbind(Xcoords[which(!(is.na(Xcoords)))],Ycoords[which(!(is.na(Xcoords)))])
RPE.line=RPE.line[order(RPE.line[,1]),]
dbg("rpe-line", "Translating the next R block for RPE.line detection")
dbg("variable-stats", "Printing summary statistics for translated RPE variables")
show.array.stats("R", R)
show.array.stats("Xcoords", Xcoords)
show.array.stats("Ycoords", Ycoords)
show.array.stats("RPE.line", RPE.line)

#
# now, fit a smooth spline (which will let us collect derivatives) to the RPE.line
RPE.sp=smooth.spline(RPE.line[,1],RPE.line[,2],df=DFonINITIALspline);
RPE.spline=cbind(as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$x),
                 as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$y),
                 as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02),deriv = 1)$y))
#
# and calculate distance along the RPE line
RPE.spline.compare=rbind(RPE.spline[2:nrow(RPE.spline),],RPE.spline[nrow(RPE.spline),])
RPE.spline=cbind(RPE.spline[,1],RPE.spline)
RPE.spline[,1]<-sqrt( (RPE.spline[,2]-RPE.spline.compare[,1])^2 + (RPE.spline[,3]-RPE.spline.compare[,2])^2 )
RPE.spline=cbind(cumsum(RPE.spline[,1]),RPE.spline)

#
# and find out where the fovea line gets closest to the RPE line
fovea.curve=summary(lm(fovea.line[,2] ~ fovea.line[,1]))[4]$coef[,1]
compare.fovea.and.RPE=RPE.spline[,c(3,4,4,4)]
compare.fovea.and.RPE[,3]<-compare.fovea.and.RPE[,1]*fovea.curve[2]+fovea.curve[1]
compare.fovea.and.RPE[,4]<-abs(compare.fovea.and.RPE[,2]-compare.fovea.and.RPE[,3])
CENTER=which(compare.fovea.and.RPE[,4]==min(compare.fovea.and.RPE[,4]))[1]
CENTER.value=RPE.spline[CENTER,1]

#
# and make a reduced version of RPE.spline to capture equally-spaced lines on which we can sample
# the retina. 
RPE.info=RPE.spline[,c(3,4,5,1)]
#
# reminder that all units are in pixels, so mulitply by PIXEL.WIDTH
# the later analysis will be 100 microns centered on the fovea, and from 500 microns to 2750 microns
# but for now (since there will be a later registration step) accept eveything from -200 microns to 3000 microns
RPE.info[,4]<-round((RPE.info[,4]-CENTER.value)*PIXEL.WIDTH,0)
RPE.info=RPE.info[which( (RPE.info[,4]>(-200.9)) & (RPE.info[,4]<(3000.9)) ),]
RPE.info.2=unique(RPE.info[,4])
RPE.info.2=cbind(RPE.info.2,RPE.info.2,RPE.info.2,RPE.info.2)
for(x in 1:nrow(RPE.info.2)) RPE.info.2[x,1:3]=RPE.info[which(RPE.info[,4]==RPE.info.2[x,4])[1],1:3]

#
# can comment out the next two lines as needed:
image(y=seq(1,dim(DARK)[2],1),x=seq(1,dim(DARK)[1],1),DARK[,,1])
matlines(RPE.info.2[,1],RPE.info.2[,2])
RPE.info.2[,3]<-( (-1) / RPE.info.2[,3] )
colnames(RPE.info.2)<-c("x_pix","y_pix","perpendicular_slope_pix","dist.on.spline.microns")
dbg("rpe-spline", "Translating the spline/center/RPE.info block through line 305")
dbg("variable-stats", "Printing summary statistics for translated spline variables")
show.array.stats("RPE.spline.compare", RPE.spline.compare)
show.array.stats("RPE.spline", RPE.spline)
show.array.stats("fovea.curve", fovea.curve)
show.array.stats("compare.fovea.and.RPE", compare.fovea.and.RPE)
show.scalar.stats("CENTER", CENTER)
show.scalar.stats("CENTER.value", CENTER.value)
show.array.stats("RPE.info", RPE.info)
show.array.stats("RPE.info.2", RPE.info.2)

###@ 
###@ and let's extract the apparent angle from RPE.info.2
APPARENT.ANGLE=RPE.info.2[which( (RPE.info.2[,4]>500) & (RPE.info.2[,4]<2750) ),1:2]
SLOPEY=summary(lm(APPARENT.ANGLE[,2] ~ APPARENT.ANGLE[,1]))$coef[2,1]
SLOPEY.500.to.2750=SLOPEY
APPARENT.ANGLE.500.to.2750=APPARENT.ANGLE
## so the slope ("change in y given 1 unit change in x") gives us the angle via atan(SLOPE)
##... atan(1) (which would be 1 unit change in y for 1 unit change in x, so 45 deg) is 0.7853982... cuz it's in radians.
#DEGREES=atan(SLOPEY)*180/pi
##
## same for -100 to 100
###@ and let's extract the apparent angle from RPE.info.2
APPARENT.ANGLE=RPE.info.2[which( (RPE.info.2[,4]>-100) & (RPE.info.2[,4]<100) ),1:2]
SLOPEY=summary(lm(APPARENT.ANGLE[,2] ~ APPARENT.ANGLE[,1]))$coef[2,1]
SLOPEY.neg100.to.100=SLOPEY
APPARENT.ANGLE.neg100.to.100=APPARENT.ANGLE
dbg("apparent-angle", "Translating the APPARENT.ANGLE block through line 331")
dbg("variable-stats", "Printing summary statistics for translated APPARENT.ANGLE variables")
show.array.stats("APPARENT.ANGLE.500.to.2750", APPARENT.ANGLE.500.to.2750)
show.scalar.stats("SLOPEY.500.to.2750", SLOPEY.500.to.2750)
show.array.stats("APPARENT.ANGLE.neg100.to.100", APPARENT.ANGLE.neg100.to.100)
show.scalar.stats("SLOPEY.neg100.to.100", SLOPEY.neg100.to.100)
rm(SLOPEY,APPARENT.ANGLE)
###@ 
###@ 

#
# next, get the perpendiculars ready. 
## at each x,y point, calculate the intercept of the perpendicular ( -1 / splines slope)
INTERCEPTS=RPE.info.2[,1]
INTERCEPTS[]<-(RPE.info.2[,2]-(RPE.info.2[,1]*RPE.info.2[,3]))

#
## now, given the slope, we know how far we have to move in the x and y to go 500 microns total...
DELTAS=RPE.info.2[,c(1,2)]
#find the delta-x
DELTAS[,1]<-cos(atan(RPE.info.2[,3]))
## and the delta-y
DELTAS[,2]<-sin(atan(RPE.info.2[,3]))

#
## now, how many pixels do we need to move to go 500 microns?
pixel.move=500/PIXEL.WIDTH
pixel.move=round(pixel.move,1)

#
## multiply by the deltas (where are calculated for moving 1 pixel).
DELTAS=DELTAS*pixel.move

#
## get points close to (ADD) and away from ("SUB") the center of the eye.
ADD=DELTAS
FLIP=DELTAS*(-1)
ADD[,1]<-ifelse(FLIP[,2]>0,FLIP[,1],ADD[,1])
ADD[,2]<-ifelse(FLIP[,2]>0,FLIP[,2],ADD[,2])

#
## ADD is going 500 microns interior at the moment, 
## so subtract is that distance divided by 10, in the opposite direction
SUB=ADD/(-10)
## and we really only need ADD to go 450 microns into the eye
## from the RPE (by the time we hit RPE)
## so, convert ADD to 450 microns
ADD=ADD-(ADD/10)


Retina.Points=RPE.info.2[,c(4,1,2,3,3,3,3,3)]
Retina.Points[,5:6]=ADD+Retina.Points[,2:3]
Retina.Points[,7:8]=SUB+Retina.Points[,2:3]
colnames(Retina.Points)<-c("dist.on.spline.microns","x_pix","y_pix","perpendicular_slope_pix","end.x","end.y","start.x","start.y")
dbg("perpendiculars", "Translating the perpendicular-setup block through line 386")
dbg("variable-stats", "Printing summary statistics for translated perpendicular variables")
show.array.stats("INTERCEPTS", INTERCEPTS)
show.array.stats("DELTAS", DELTAS)
show.scalar.stats("pixel.move", pixel.move)
show.array.stats("ADD", ADD)
show.array.stats("FLIP", FLIP)
show.array.stats("SUB", SUB)
show.array.stats("Retina.Points", Retina.Points)

rm(RPE.spline.compare,compare.fovea.and.RPE,fovea.curve,RPE.sp,RPE.line,CENTER,CENTER.value,RPE.info)
rm(INTERCEPTS,DELTAS,RPE.info.2,SUB,ADD,FLIP,pixel.move)
rm(fovea.line)



#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
##
##
## now, set up a matrix for the outputs...
## we go in 1 micron steps.
## we have our x range dictated by the first column of Retina.Points
## and the y range will be -50 to 450

#
# let's do the flattened markers of layers. 
# we'll work back in a later loop, and fill in things like this:
###### FLATTENED.DARK.RETINA=array(data=NA, dim=c(nrow(Retina.Points),500,dim(DARK)[3]))
###### FLATTENED.LIGHT.RETINA=array(data=NA, dim=c(nrow(Retina.Points),500,dim(LIGHT)[3]))

#
##
FLATTENED.MARKERS=matrix(,nrow(Retina.Points),500)
dbg("flattened-markers", "FLATTENED.MARKERS dim:", paste(dim(FLATTENED.MARKERS), collapse="x"), "Retina.Points rows:", nrow(Retina.Points))
UpperX=dim(DARK)[2];
UpperY=dim(DARK)[1];

unwrapped.recon=as.vector(REF.DARK[,,1])
GETrecon=function(x) if((x[1]>=1)&(x[1]<=UpperX)&(x[2]>=1)&(x[2]<=UpperY)) unwrapped.recon[(((x[1]-1)*UpperY)+x[2])] else NA;

for(x in 1:nrow(Retina.Points))
{LINE=seq(Retina.Points[x,5],Retina.Points[x,7],(Retina.Points[x,7]-Retina.Points[x,5])/500);
 LINE=cbind(LINE,seq(Retina.Points[x,6],Retina.Points[x,8],(Retina.Points[x,8]-Retina.Points[x,6])/500));
 LINE=floor(LINE); ## because the matrix starts at 1,1, all coordinates calculated to-date would use anything between 1 and 1.999 to refer to 1.
 F=as.vector(tapply(LINE[,c(2,1)],as.factor(cbind(seq(1,(500+1),1),seq(1,(500+1),1))),GETrecon));
 FLATTENED.MARKERS[x,1:ncol(FLATTENED.MARKERS)]=F[2:length(F)]}
dbg("variable-stats", "Printing summary statistics for translated flattened-marker variables")
show.scalar.stats("UpperX", UpperX)
show.scalar.stats("UpperY", UpperY)
show.array.stats("unwrapped.recon", unwrapped.recon)
show.array.stats("FLATTENED.MARKERS", FLATTENED.MARKERS)
dbg("dark-loop", "Translating the dark-image loop through line 599")


#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

## make storage for the apparent angle data:
APPARENT.ANGLES.FOR.LIGHT=cbind(IMAGE.INDEX.LIGHT,IMAGE.INDEX.LIGHT,IMAGE.INDEX.LIGHT)
APPARENT.ANGLES.FOR.LIGHT[,2]<-NA
APPARENT.ANGLES.FOR.LIGHT[,3]<-NA
colnames(APPARENT.ANGLES.FOR.LIGHT)<-c("image","fovea_neg100_to_100","500_to_2750")
APPARENT.ANGLES.FOR.DARK=cbind(IMAGE.INDEX.DARK,IMAGE.INDEX.DARK,IMAGE.INDEX.DARK)
APPARENT.ANGLES.FOR.DARK[,2]<-NA
APPARENT.ANGLES.FOR.DARK[,3]<-NA
colnames(APPARENT.ANGLES.FOR.DARK)<-c("image","fovea_neg100_to_100","500_to_2750")
#
##
## NOW, USE LOOPS TO WORK THROUGH DARK AND LIGHT


# dark
FLATTENED.DARK.RETINA=array(data=NA, dim=c(nrow(Retina.Points),500,dim(DARK)[3]))
for(z in 1:length(IMAGE.INDEX.DARK))
 {dbg("dark-loop", "Processing z=", z, "of", length(IMAGE.INDEX.DARK), "REF.DARK slice dim:", paste(dim(REF.DARK[,,z]), collapse="x"))
  R=REF.DARK[,,z];
  R[which(R<243)]=NA;
  R[which(R>243)]=NA;
  R[which(R==243)]=1;
  #
  Xcoords=Xs*R;
  Ycoords=Ys*R;
  fovea.line=cbind(Xcoords[which(!(is.na(Xcoords)))],Ycoords[which(!(is.na(Xcoords)))]);
  fovea.line=fovea.line[order(fovea.line[,1]),];
  ## just in case it's a perfectly vertical line....
  if(length(unique(fovea.line[,1]))==1) fovea.line[1,1]=fovea.line[1,1]+0.1
  #
  # repeat that, but to find the RPE
  R=REF.DARK[,,z];
  R[which(R<255)]=NA;
  R[which(R>255)]=NA;
  R[which(R==255)]=1;
  Xcoords=Xs*R;
  Ycoords=Ys*R;
  RPE.line=cbind(Xcoords[which(!(is.na(Xcoords)))],Ycoords[which(!(is.na(Xcoords)))]);
  RPE.line=RPE.line[order(RPE.line[,1]),];
  #
  # now, fit a smooth spline (which will let us collect derivatives) to the RPE.line
  RPE.sp=smooth.spline(RPE.line[,1],RPE.line[,2],df=DFonINITIALspline);
  RPE.spline=cbind(as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$x),
                   as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$y),
                   as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02),deriv = 1)$y));
  #
  # and calculate distance along the RPE line
  RPE.spline.compare=rbind(RPE.spline[2:nrow(RPE.spline),],RPE.spline[nrow(RPE.spline),]);
  RPE.spline=cbind(RPE.spline[,1],RPE.spline);
  RPE.spline[,1]<-sqrt( (RPE.spline[,2]-RPE.spline.compare[,1])^2 + (RPE.spline[,3]-RPE.spline.compare[,2])^2 );
  RPE.spline=cbind(cumsum(RPE.spline[,1]),RPE.spline);
  #
  # and find out where the fovea line gets closest to the RPE line
  fovea.curve=summary(lm(fovea.line[,2] ~ fovea.line[,1]))[4]$coef[,1];
  compare.fovea.and.RPE=RPE.spline[,c(3,4,4,4)];
  compare.fovea.and.RPE[,3]<-compare.fovea.and.RPE[,1]*fovea.curve[2]+fovea.curve[1];
  compare.fovea.and.RPE[,4]<-abs(compare.fovea.and.RPE[,2]-compare.fovea.and.RPE[,3]);
  CENTER=which(compare.fovea.and.RPE[,4]==min(compare.fovea.and.RPE[,4]))[1];
  CENTER.value=RPE.spline[CENTER,1];
  #
  # and make a reduced version of RPE.spline to capture equally-spaced lines on which we can sample
  # the retina. 
  RPE.info=RPE.spline[,c(3,4,5,1)];
  #
  # reminder that all units are in pixels, so mulitply by PIXEL.WIDTH
  # the later analysis will be 100 microns centered on the fovea, and from 500 microns to 2750 microns
  # but for now (since there will be a later registration step) accept eveything from -200 microns to 3000 microns
  RPE.info[,4]<-round((RPE.info[,4]-CENTER.value)*PIXEL.WIDTH,0);
  RPE.info=RPE.info[which( (RPE.info[,4]>(-200.9)) & (RPE.info[,4]<(3000.9)) ),];
  RPE.info.2=unique(RPE.info[,4]);
  RPE.info.2=cbind(RPE.info.2,RPE.info.2,RPE.info.2,RPE.info.2);
  for(x in 1:nrow(RPE.info.2)) RPE.info.2[x,1:3]=RPE.info[which(RPE.info[,4]==RPE.info.2[x,4])[1],1:3];
  #
  # can comment out the next two lines as needed:
  #image(y=seq(1,dim(DARK)[2],1),x=seq(1,dim(DARK)[1],1),DARK[,,z])
  #matlines(RPE.info.2[,1],RPE.info.2[,2])
  RPE.info.2[,3]<-( (-1) / RPE.info.2[,3] );
  colnames(RPE.info.2)<-c("x_pix","y_pix","perpendicular_slope_pix","dist.on.spline.microns");
  ###@ 
  ###@ and let's extract the apparent angle from RPE.info.2
  APPARENT.ANGLE=RPE.info.2[which( (RPE.info.2[,4]>0) & (RPE.info.2[,4]<2750) ),1:2]
  SLOPEY=summary(lm(APPARENT.ANGLE[,2] ~ APPARENT.ANGLE[,1]))$coef[2,1]
  ## so the slope ("change in y given 1 unit change in x") gives us the angle via atan(SLOPE)
  ##... atan(1) (which would be 1 unit change in y for 1 unit change in x, so 45 deg) is 0.7853982... cuz it's in radians.
  APPARENT.ANGLES.FOR.DARK[z,3]=atan(SLOPEY)*180/pi
  rm(SLOPEY,APPARENT.ANGLE)
  ###@ and repeat for the fovea
  APPARENT.ANGLE=RPE.info.2[which( (RPE.info.2[,4]>-100) & (RPE.info.2[,4]<100) ),1:2]
  SLOPEY=summary(lm(APPARENT.ANGLE[,2] ~ APPARENT.ANGLE[,1]))$coef[2,1]
  APPARENT.ANGLES.FOR.DARK[z,2]=atan(SLOPEY)*180/pi
  rm(SLOPEY,APPARENT.ANGLE)
  ###@ 
  ###@ 
  #
  # next, get the perpendiculars ready. 
  ## at each x,y point, calculate the intercept of the perpendicular ( -1 / splines slope)
  INTERCEPTS=RPE.info.2[,1];
  INTERCEPTS[]<-(RPE.info.2[,2]-(RPE.info.2[,1]*RPE.info.2[,3]));
  #
  ## now, given the slope, we know how far we have to move in the x and y to go 500 microns total...
  DELTAS=RPE.info.2[,c(1,2)];
  #find the delta-x
  DELTAS[,1]<-cos(atan(RPE.info.2[,3]));
  ## and the delta-y
  DELTAS[,2]<-sin(atan(RPE.info.2[,3]));
  #
  ## now, how many pixels do we need to move to go 500 microns?
  pixel.move=500/PIXEL.WIDTH;
  pixel.move=round(pixel.move,1);
  #
  ## multiply by the deltas (where are calculated for moving 1 pixel).
  DELTAS=DELTAS*pixel.move;
  #
  ## get points close to (ADD) and away from ("SUB") the center of the eye.
  ADD=DELTAS;
  FLIP=DELTAS*(-1);
  ADD[,1]<-ifelse(FLIP[,2]>0,FLIP[,1],ADD[,1]);
  ADD[,2]<-ifelse(FLIP[,2]>0,FLIP[,2],ADD[,2]);
  #
  ## ADD is going 500 microns interior at the moment, 
  ## so subtract is that distance divided by 10, in the opposite direction
  SUB=ADD/(-10);
  ## and we really only need ADD to go 450 microns into the eye
  ## from the RPE (by the time we hit RPE)
  ## so, convert ADD to 450 microns
  ADD=ADD-(ADD/10);
  Retina.Points=RPE.info.2[,c(4,1,2,3,3,3,3,3)];
  Retina.Points[,5:6]=ADD+Retina.Points[,2:3];
  Retina.Points[,7:8]=SUB+Retina.Points[,2:3];
  colnames(Retina.Points)<-c("dist.on.spline.microns","x_pix","y_pix","perpendicular_slope_pix","end.x","end.y","start.x","start.y");
  rm(RPE.spline.compare,compare.fovea.and.RPE,fovea.curve,RPE.sp,RPE.line,CENTER,CENTER.value,RPE.info);
  rm(INTERCEPTS,DELTAS,RPE.info.2,SUB,ADD,FLIP,pixel.move);
  rm(fovea.line);
  #
  ##
  ## ##   ##   ##   ##   ##   ##   ## 
  ##
  #
  unwrapped.retina=as.vector(DARK[,,z])
  GETintensity=function(x) if((x[1]>=1)&(x[1]<=UpperX)&(x[2]>=1)&(x[2]<=UpperY)) unwrapped.retina[(((x[1]-1)*UpperY)+x[2])] else NA;
  for(x in 1:nrow(Retina.Points))
  {LINE=seq(Retina.Points[x,5],Retina.Points[x,7],(Retina.Points[x,7]-Retina.Points[x,5])/500);
   LINE=cbind(LINE,seq(Retina.Points[x,6],Retina.Points[x,8],(Retina.Points[x,8]-Retina.Points[x,6])/500));
   LINE=floor(LINE); ## because the matrix starts at 1,1, all coordinates calculated to-date would use anything between 1 and 1.999 to refer to 1.
   F=as.vector(tapply(LINE[,c(2,1)],as.factor(cbind(seq(1,(500+1),1),seq(1,(500+1),1))),GETintensity));
   FLATTENED.DARK.RETINA[x,1:ncol(FLATTENED.MARKERS),z]=F[2:length(F)]}}



dbg("dark-loop", "Translating the dark-image loop through line 599")


# light
FLATTENED.LIGHT.RETINA=array(data=NA, dim=c(nrow(Retina.Points),500,dim(LIGHT)[3]))
for(z in 1:length(IMAGE.INDEX.LIGHT))
 {dbg("light-loop", "Processing z=", z, "of", length(IMAGE.INDEX.LIGHT), "REF.LIGHT slice dim:", paste(dim(REF.LIGHT[,,z]), collapse="x"))
  R=REF.LIGHT[,,z];
  R[which(R<243)]=NA;
  R[which(R>243)]=NA;
  R[which(R==243)]=1;
  #
  Xcoords=Xs*R;
  Ycoords=Ys*R;
  fovea.line=cbind(Xcoords[which(!(is.na(Xcoords)))],Ycoords[which(!(is.na(Xcoords)))]);
  fovea.line=fovea.line[order(fovea.line[,1]),];
  ## just in case it's a perfectly vertical line....
  if(length(unique(fovea.line[,1]))==1) fovea.line[1,1]=fovea.line[1,1]+0.1
  #
  # repeat that, but to find the RPE
  R=REF.LIGHT[,,z];
  R[which(R<255)]=NA;
  R[which(R>255)]=NA;
  R[which(R==255)]=1;
  Xcoords=Xs*R;
  Ycoords=Ys*R;
  RPE.line=cbind(Xcoords[which(!(is.na(Xcoords)))],Ycoords[which(!(is.na(Xcoords)))]);
  RPE.line=RPE.line[order(RPE.line[,1]),];
  #
  # now, fit a smooth spline (which will let us collect derivatives) to the RPE.line
  RPE.sp=smooth.spline(RPE.line[,1],RPE.line[,2],df=DFonINITIALspline);
  RPE.spline=cbind(as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$x),
                   as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$y),
                   as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02),deriv = 1)$y));
  #
  # and calculate distance along the RPE line
  RPE.spline.compare=rbind(RPE.spline[2:nrow(RPE.spline),],RPE.spline[nrow(RPE.spline),]);
  RPE.spline=cbind(RPE.spline[,1],RPE.spline);
  RPE.spline[,1]<-sqrt( (RPE.spline[,2]-RPE.spline.compare[,1])^2 + (RPE.spline[,3]-RPE.spline.compare[,2])^2 );
  RPE.spline=cbind(cumsum(RPE.spline[,1]),RPE.spline);
  #
  # and find out where the fovea line gets closest to the RPE line
  fovea.curve=summary(lm(fovea.line[,2] ~ fovea.line[,1]))[4]$coef[,1];
  compare.fovea.and.RPE=RPE.spline[,c(3,4,4,4)];
  compare.fovea.and.RPE[,3]<-compare.fovea.and.RPE[,1]*fovea.curve[2]+fovea.curve[1];
  compare.fovea.and.RPE[,4]<-abs(compare.fovea.and.RPE[,2]-compare.fovea.and.RPE[,3]);
  CENTER=which(compare.fovea.and.RPE[,4]==min(compare.fovea.and.RPE[,4]))[1];
  CENTER.value=RPE.spline[CENTER,1];
  #
  # and make a reduced version of RPE.spline to capture equally-spaced lines on which we can sample
  # the retina. 
  RPE.info=RPE.spline[,c(3,4,5,1)];
  #
  # reminder that all units are in pixels, so mulitply by PIXEL.WIDTH
  # the later analysis will be 100 microns centered on the fovea, and from 500 microns to 2750 microns
  # but for now (since there will be a later registration step) accept eveything from -200 microns to 3000 microns
  RPE.info[,4]<-round((RPE.info[,4]-CENTER.value)*PIXEL.WIDTH,0);
  RPE.info=RPE.info[which( (RPE.info[,4]>(-200.9)) & (RPE.info[,4]<(3000.9)) ),];
  RPE.info.2=unique(RPE.info[,4]);
  RPE.info.2=cbind(RPE.info.2,RPE.info.2,RPE.info.2,RPE.info.2);
  for(x in 1:nrow(RPE.info.2)) RPE.info.2[x,1:3]=RPE.info[which(RPE.info[,4]==RPE.info.2[x,4])[1],1:3];
  #
  # can comment out the next two lines as needed:
  #image(y=seq(1,dim(LIGHT)[2],1),x=seq(1,dim(LIGHT)[1],1),LIGHT[,,z])
  #matlines(RPE.info.2[,1],RPE.info.2[,2])
  RPE.info.2[,3]<-( (-1) / RPE.info.2[,3] );
  colnames(RPE.info.2)<-c("x_pix","y_pix","perpendicular_slope_pix","dist.on.spline.microns");
  ###@ 
  ###@ and let's extract the apparent angle from RPE.info.2
  APPARENT.ANGLE=RPE.info.2[which( (RPE.info.2[,4]>0) & (RPE.info.2[,4]<2750) ),1:2]
  SLOPEY=summary(lm(APPARENT.ANGLE[,2] ~ APPARENT.ANGLE[,1]))$coef[2,1]
  ## so the slope ("change in y given 1 unit change in x") gives us the angle via atan(SLOPE)
  ##... atan(1) (which would be 1 unit change in y for 1 unit change in x, so 45 deg) is 0.7853982... cuz it's in radians.
  APPARENT.ANGLES.FOR.LIGHT[z,3]=atan(SLOPEY)*180/pi
  rm(SLOPEY,APPARENT.ANGLE)
  ###@ and repeat for the fovea
  APPARENT.ANGLE=RPE.info.2[which( (RPE.info.2[,4]>-100) & (RPE.info.2[,4]<100) ),1:2]
  SLOPEY=summary(lm(APPARENT.ANGLE[,2] ~ APPARENT.ANGLE[,1]))$coef[2,1]
  APPARENT.ANGLES.FOR.LIGHT[z,2]=atan(SLOPEY)*180/pi
  rm(SLOPEY,APPARENT.ANGLE)
  ###@ 
  ###@ 
  #
  # next, get the perpendiculars ready. 
  ## at each x,y point, calculate the intercept of the perpendicular ( -1 / splines slope)
  INTERCEPTS=RPE.info.2[,1];
  INTERCEPTS[]<-(RPE.info.2[,2]-(RPE.info.2[,1]*RPE.info.2[,3]));
  #
  ## now, given the slope, we know how far we have to move in the x and y to go 500 microns total...
  DELTAS=RPE.info.2[,c(1,2)];
  #find the delta-x
  DELTAS[,1]<-cos(atan(RPE.info.2[,3]));
  ## and the delta-y
  DELTAS[,2]<-sin(atan(RPE.info.2[,3]));
  #
  ## now, how many pixels do we need to move to go 500 microns?
  pixel.move=500/PIXEL.WIDTH;
  pixel.move=round(pixel.move,1);
  #
  ## multiply by the deltas (where are calculated for moving 1 pixel).
  DELTAS=DELTAS*pixel.move;
  #
  ## get points close to (ADD) and away from ("SUB") the center of the eye.
  ADD=DELTAS;
  FLIP=DELTAS*(-1);
  ADD[,1]<-ifelse(FLIP[,2]>0,FLIP[,1],ADD[,1]);
  ADD[,2]<-ifelse(FLIP[,2]>0,FLIP[,2],ADD[,2]);
  #
  ## ADD is going 500 microns interior at the moment, 
  ## so subtract is that distance divided by 10, in the opposite direction
  SUB=ADD/(-10);
  ## and we really only need ADD to go 450 microns into the eye
  ## from the RPE (by the time we hit RPE)
  ## so, convert ADD to 450 microns
  ADD=ADD-(ADD/10);
  Retina.Points=RPE.info.2[,c(4,1,2,3,3,3,3,3)];
  Retina.Points[,5:6]=ADD+Retina.Points[,2:3];
  Retina.Points[,7:8]=SUB+Retina.Points[,2:3];
  colnames(Retina.Points)<-c("dist.on.spline.microns","x_pix","y_pix","perpendicular_slope_pix","end.x","end.y","start.x","start.y");
  rm(RPE.spline.compare,compare.fovea.and.RPE,fovea.curve,RPE.sp,RPE.line,CENTER,CENTER.value,RPE.info);
  rm(INTERCEPTS,DELTAS,RPE.info.2,SUB,ADD,FLIP,pixel.move);
  rm(fovea.line);
  #
  ##
  ## ##   ##   ##   ##   ##   ##   ## 
  ##
  #
  unwrapped.retina=as.vector(LIGHT[,,z])
  GETintensity=function(x) if((x[1]>=1)&(x[1]<=UpperX)&(x[2]>=1)&(x[2]<=UpperY)) unwrapped.retina[(((x[1]-1)*UpperY)+x[2])] else NA;
  for(x in 1:nrow(Retina.Points))
  {LINE=seq(Retina.Points[x,5],Retina.Points[x,7],(Retina.Points[x,7]-Retina.Points[x,5])/500);
   LINE=cbind(LINE,seq(Retina.Points[x,6],Retina.Points[x,8],(Retina.Points[x,8]-Retina.Points[x,6])/500));
   LINE=floor(LINE); ## because the matrix starts at 1,1, all coordinates calculated to-date would use anything between 1 and 1.999 to refer to 1.
   F=as.vector(tapply(LINE[,c(2,1)],as.factor(cbind(seq(1,(500+1),1),seq(1,(500+1),1))),GETintensity));
   FLATTENED.LIGHT.RETINA[x,1:ncol(FLATTENED.MARKERS),z]=F[2:length(F)]}}

rm(UpperX,UpperY,x,z,F,GETrecon,GETintensity,REF.DARK,REF.LIGHT,Retina.Points,RPE.spline,unwrapped.retina,unwrapped.recon,LIGHT)
rm(DARK)
rm(Xcoords,Ycoords)
rm(LINE)
rm(R)

#a.Rdata

#EXPORT=FLATTENED.DARK.RETINA
#EXPORT[which(is.na(EXPORT))]=0
#f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste(TO.PROCESS.DARK,"_flat",sep=""),size="float")
#f.write.analyze(FLATTENED.LIGHT.RETINA[,dim(FLATTENED.LIGHT.RETINA)[2]:1,],paste(TO.PROCESS.LIGHT,"_flat",sep=""),size="float")




dbg("light-loop", "Translating the light-image loop through line 755")
dbg("variable-stats", "Printing summary statistics for translated dark/light loop variables")
show.array.stats("APPARENT.ANGLES.FOR.LIGHT", APPARENT.ANGLES.FOR.LIGHT)
show.array.stats("APPARENT.ANGLES.FOR.DARK", APPARENT.ANGLES.FOR.DARK)
show.array.stats("FLATTENED.DARK.RETINA", FLATTENED.DARK.RETINA)
show.array.stats("FLATTENED.LIGHT.RETINA", FLATTENED.LIGHT.RETINA)

#################################################
#################################################
#################################################
#################################################   ## now, we need to verify lock to the RPE, and after that spatially normalize across images.
#################################################
#################################################
#################################################


###
### at this point, let's convert the data OUT of the log-transformed values.

FLATTENED.DARK.RETINA[which(is.na(FLATTENED.DARK.RETINA))]=-32768
FLATTENED.DARK.RETINA=FLATTENED.DARK.RETINA+32768
##
## 34276 previously found to be the magic number; no values witnessed below that
## <new for 2022-JUN-19> replace this line, with the next
   #FLATTENED.DARK.RETINA[which(FLATTENED.DARK.RETINA<34376)]=34376
FLATTENED.DARK.RETINA[which(FLATTENED.DARK.RETINA<0)]=0
## </new for 2022-JUN-19> 

##
## this is proportional to log-transformed signal; easiest thin to do will be 2^(value/5000)
## then, if I ever want to convert back, use ln(new value) * 5000
FLATTENED.DARK.RETINA.RAW=2^(FLATTENED.DARK.RETINA/5000)

###
### same for light
###
### at this point, let's convert the data OUT of the log-transformed values.
FLATTENED.LIGHT.RETINA[which(is.na(FLATTENED.LIGHT.RETINA))]=-32768
FLATTENED.LIGHT.RETINA=FLATTENED.LIGHT.RETINA+32768
## <new for 2022-JUN-19> replace this line, with the next
    #FLATTENED.LIGHT.RETINA[which(FLATTENED.LIGHT.RETINA<34376)]=34376
FLATTENED.LIGHT.RETINA[which(FLATTENED.LIGHT.RETINA<0)]=0
## </new for 2022-JUN-19> 
FLATTENED.LIGHT.RETINA.RAW=2^(FLATTENED.LIGHT.RETINA/5000)
dbg("post-log-convert", "Converted DARK and LIGHT flattened arrays back to raw scale")
dbg("variable-stats", "Printing summary statistics for translated post-log conversion variables")
show.array.stats("FLATTENED.DARK.RETINA", FLATTENED.DARK.RETINA)
show.array.stats("FLATTENED.DARK.RETINA.RAW", FLATTENED.DARK.RETINA.RAW)
show.array.stats("FLATTENED.LIGHT.RETINA", FLATTENED.LIGHT.RETINA)
show.array.stats("FLATTENED.LIGHT.RETINA.RAW", FLATTENED.LIGHT.RETINA.RAW)
stop.at.boundary("exit-after-post-log-convert", "Reached current translated boundary after post-log conversion; stopping here.")


#################################################
#################################################
#################################################

###...after several tries to identify the RPE peak, or external border, or whatever, I gave up.
###...never met standards of just human tracing, which still isn't great.
### ...averaging all the images gives a fairly nice image (in terms of horizontal RPE)
### ...so maybe make an average, and then grab blocks of each individual image and see how far up/down I'd need to shift the whole thing 
### (or the outer 2/3rds of the retina, or whatever)

FIRST.GRAND.MEAN=FLATTENED.DARK.RETINA.RAW[,,1]
dbg("grand-mean", "Building FIRST.GRAND.MEAN from DARK and LIGHT volumes")
for(z in 2:dim(FLATTENED.DARK.RETINA.RAW)[3]) FIRST.GRAND.MEAN=FIRST.GRAND.MEAN+FLATTENED.DARK.RETINA.RAW[,,z]
for(z in 2:dim(FLATTENED.LIGHT.RETINA.RAW)[3]) FIRST.GRAND.MEAN=FIRST.GRAND.MEAN+FLATTENED.LIGHT.RETINA.RAW[,,z]
FIRST.GRAND.MEAN=FIRST.GRAND.MEAN/(dim(FLATTENED.DARK.RETINA.RAW)[3]+dim(FLATTENED.LIGHT.RETINA.RAW)[3])

#
## grap estimate sof retinal thickness based on hand-drawn location
ROUGH.VIT.RETINA.POSITION=cbind(seq(-200,3000,1),seq(-200,3000,1))
ROUGH.VIT.RETINA.POSITION[,2]<-NA
dbg("rough-vit-loop", "ROUGH.VIT.RETINA.POSITION rows:", nrow(ROUGH.VIT.RETINA.POSITION), "FLATTENED.MARKERS rows:", nrow(FLATTENED.MARKERS))
for(x in 1:nrow(ROUGH.VIT.RETINA.POSITION))
 {if(x > nrow(FLATTENED.MARKERS)) {
   stop(paste0("Index exceeds FLATTENED.MARKERS in rough-vit-loop: x=", x, ", rows=", nrow(FLATTENED.MARKERS), ". Adjust loop bounds to marker rows."))
  }
  A=which(FLATTENED.MARKERS[x,]==249);
  if(length(A)>0) ROUGH.VIT.RETINA.POSITION[x,2]=A[length(A)]}

#
## this will give us a marker of where to look when aligning outer retina (from 480, which is ~30 pix past the RPE signal peak, to 70% of the retinal thickness)
LOOK.TO=ROUGH.VIT.RETINA.POSITION
LOOK.TO[,2]<-round((250-(0.7*(250-LOOK.TO[,2]))),0)

####
####
#### now, grab blocks of data and see how to align them to the FIRST.GRAND.MEAN
##
## the moving window for this part will be 200 um wide
window.width.in.pixels=400
start.move=201
end.move=(nrow(FLATTENED.MARKERS)-start.move)-1
MEAN.x=function(x) mean(na.rm=TRUE,x)
window.factor=matrix(,ncol(FLATTENED.MARKERS),window.width.in.pixels);
for(y in 1:nrow(window.factor)) window.factor[y,]=y
window.factor=t(window.factor)
window.factor=as.factor(window.factor)

SHIFT.POSITION.DARK=matrix(,dim(FLATTENED.DARK.RETINA.RAW)[1],dim(FLATTENED.DARK.RETINA.RAW)[3]+1)
SHIFT.POSITION.DARK[,1]<-seq(-200,3000,1)
for(z in 1:dim(FLATTENED.DARK.RETINA)[3])
 {REVISE.DARK=FLATTENED.DARK.RETINA.RAW[,,z];
  for(x in seq(start.move,end.move,50))
   {profile=as.vector(tapply(REVISE.DARK[(x-199):(x+200),],window.factor,MEAN.x));
    comparison=as.vector(tapply(FIRST.GRAND.MEAN[(x-199):(x+200),],window.factor,MEAN.x));
    top.range=max(na.rm=T,LOOK.TO[(x-199):(x+200),2]);
    check=cbind(seq(480,top.range,-1),profile[480:top.range],comparison[480:top.range]);
    #plot(check[,1],check[,2],ylim=c(min(na.rm=T,check[,2:3]),max(na.rm=T,check[,2:3])));
    #matlines(check[,1],check[,3]);
    #
    #move +/-10 microns, here, positive ten means that whatever was at 450 would be placed at 440 (i.e. its a move towards the eye center) to match the group mean)
    slide=cbind(seq(-10,10,1),seq(-10,10,1))
    slide[11,2]=cor.test(check[,2],check[,3])$est;
    for(s in 1:10) slide[s,2]=cor.test(check[((slide[s,1]*-1)+1):nrow(check),2],check[1:(nrow(check)+slide[s,1]),3])$est;
    for(s in 12:21) slide[s,2]=cor.test(check[1:(nrow(check)-slide[s,1]),2],check[((slide[s,1])+1):nrow(check),3])$est;
    bestmove=slide[which(slide[,2]==max(na.rm=T,slide[,2]))[1],1];
    SHIFT.POSITION.DARK[x,(z+1)]=bestmove}}

## to harmonize with some code just next, here, restate SHIFT.POSITION.DARK as where 450 should go:
SHIFT.POSITION.DARK[,2:ncol(SHIFT.POSITION.DARK)]=450-SHIFT.POSITION.DARK[,2:ncol(SHIFT.POSITION.DARK)]

## now, refine estimates...
SHIFT.POSITION.DARK.REFINED=SHIFT.POSITION.DARK
for(y in 2:ncol(SHIFT.POSITION.DARK))
 {plot(SHIFT.POSITION.DARK[,1],SHIFT.POSITION.DARK[,y],ylim=c(430,470));
  SHIFT.POSITION.DARK.sp.maker=cbind(SHIFT.POSITION.DARK[,1],SHIFT.POSITION.DARK[,y]);
  SHIFT.POSITION.DARK.sp.maker=SHIFT.POSITION.DARK.sp.maker[which(!(is.na(SHIFT.POSITION.DARK.sp.maker[,2]))),];
  SHIFT.POSITION.DARK.sp=smooth.spline(SHIFT.POSITION.DARK.sp.maker[,1],SHIFT.POSITION.DARK.sp.maker[,2],df=DFforSECONDfit);
  SHIFT.POSITION.DARK.spline=cbind(as.numeric(predict(SHIFT.POSITION.DARK.sp,seq(-200,3000,1))$x),
                                 as.numeric(predict(SHIFT.POSITION.DARK.sp,seq(-200,3000,1))$y));
  matlines(SHIFT.POSITION.DARK.spline[,1],SHIFT.POSITION.DARK.spline[,2],col="red");
  SHIFT.POSITION.DARK.REFINED[,y]=SHIFT.POSITION.DARK.spline[,2]}

SHIFT.POSITION.DARK.REFINED[,2:ncol(SHIFT.POSITION.DARK.REFINED)]=round(SHIFT.POSITION.DARK.REFINED[,2:ncol(SHIFT.POSITION.DARK.REFINED)],0)
## for cosmetic purposes, whatever we have at 0 is carried over through -200
for(x in 1:199) SHIFT.POSITION.DARK.REFINED[x,2:ncol(SHIFT.POSITION.DARK.REFINED)]=SHIFT.POSITION.DARK.REFINED[200,2:ncol(SHIFT.POSITION.DARK.REFINED)]


###
### this block only for display purposes if needed (is using log-transformed values)
#FLATTENED.DARK.RETINA.REFINED=FLATTENED.DARK.RETINA
#FLATTENED.DARK.RETINA.REFINED[,,]<-0
#for(z in 1:dim(FLATTENED.DARK.RETINA)[3])
# {REFINE=FLATTENED.DARK.RETINA[,,z];
#  for(x in 1:nrow(REFINE))
#   {border=SHIFT.POSITION.DARK.REFINED[x,(z+1)];
#    if(border<450) FLATTENED.DARK.RETINA.REFINED[x,1:(500+(border-449)),z]=REFINE[x,(450-border):500];
#    if(border==450) FLATTENED.DARK.RETINA.REFINED[x,,z]=REFINE[x,];
#    if(border>450) FLATTENED.DARK.RETINA.REFINED[x,(1+(border-450)):500,z]=REFINE[x,1:(500-(border-450))]}}
#EXPORT=FLATTENED.DARK.RETINA.REFINED
#EXPORT[which(is.na(EXPORT))]=0
#f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste(TO.PROCESS.DARK,"_flat",sep=""),size="float")

##
### this actually shifts the real (non-log-transformed) data:
FLATTENED.DARK.RETINA.RAW.REFINED=FLATTENED.DARK.RETINA.RAW
FLATTENED.DARK.RETINA.RAW.REFINED[,,]<-0
for(z in 1:dim(FLATTENED.DARK.RETINA.RAW)[3])
 {REFINE=FLATTENED.DARK.RETINA.RAW[,,z];
  for(x in 1:nrow(REFINE))
   {border=SHIFT.POSITION.DARK.REFINED[x,(z+1)];
    if(border<450) FLATTENED.DARK.RETINA.RAW.REFINED[x,1:(500+(border-449)),z]=REFINE[x,(450-border):500];
    if(border==450) FLATTENED.DARK.RETINA.RAW.REFINED[x,,z]=REFINE[x,];
    if(border>450) FLATTENED.DARK.RETINA.RAW.REFINED[x,(1+(border-450)):500,z]=REFINE[x,1:(500-(border-450))]}}

## and don't forget to move the markers!
FLATTENED.MARKERS.REFINED=FLATTENED.MARKERS
FLATTENED.MARKERS.REFINED[,]<-0
REFINE=FLATTENED.MARKERS;
 for(x in 1:nrow(REFINE))
  {border=SHIFT.POSITION.DARK.REFINED[x,2];
   if(border<450) FLATTENED.MARKERS.REFINED[x,1:(500+(border-449))]=REFINE[x,(450-border):500];
   if(border==450) FLATTENED.MARKERS.REFINED[x,]=REFINE[x,];
   if(border>450) FLATTENED.MARKERS.REFINED[x,(1+(border-450)):500]=REFINE[x,1:(500-(border-450))]}


SHIFT.POSITION.LIGHT=matrix(,dim(FLATTENED.LIGHT.RETINA.RAW)[1],dim(FLATTENED.LIGHT.RETINA.RAW)[3]+1)
SHIFT.POSITION.LIGHT[,1]<-seq(-200,3000,1)
for(z in 1:dim(FLATTENED.LIGHT.RETINA)[3])
 {REVISE.LIGHT=FLATTENED.LIGHT.RETINA.RAW[,,z];
  for(x in seq(start.move,end.move,50))
   {profile=as.vector(tapply(REVISE.LIGHT[(x-199):(x+200),],window.factor,MEAN.x));
    comparison=as.vector(tapply(FIRST.GRAND.MEAN[(x-199):(x+200),],window.factor,MEAN.x));
    top.range=max(na.rm=T,LOOK.TO[(x-199):(x+200),2]);
    check=cbind(seq(480,top.range,-1),profile[480:top.range],comparison[480:top.range]);
    #plot(check[,1],check[,2],ylim=c(min(na.rm=T,check[,2:3]),max(na.rm=T,check[,2:3])));
    #matlines(check[,1],check[,3]);
    #
    #move +/-10 microns, here, positive ten means that whatever was at 450 would be placed at 440 (i.e. its a move towards the eye center) to match the group mean)
    slide=cbind(seq(-10,10,1),seq(-10,10,1))
    slide[11,2]=cor.test(check[,2],check[,3])$est;
    for(s in 1:10) slide[s,2]=cor.test(check[((slide[s,1]*-1)+1):nrow(check),2],check[1:(nrow(check)+slide[s,1]),3])$est;
    for(s in 12:21) slide[s,2]=cor.test(check[1:(nrow(check)-slide[s,1]),2],check[((slide[s,1])+1):nrow(check),3])$est;
    bestmove=slide[which(slide[,2]==max(na.rm=T,slide[,2]))[1],1];
    SHIFT.POSITION.LIGHT[x,(z+1)]=bestmove}}

## to harmonize with some code just next, here, restate SHIFT.POSITION.LIGHT as where 450 should go:
SHIFT.POSITION.LIGHT[,2:ncol(SHIFT.POSITION.LIGHT)]=450-SHIFT.POSITION.LIGHT[,2:ncol(SHIFT.POSITION.LIGHT)]

## now, refine estimates...
SHIFT.POSITION.LIGHT.REFINED=SHIFT.POSITION.LIGHT
for(y in 2:ncol(SHIFT.POSITION.LIGHT))
 {plot(SHIFT.POSITION.LIGHT[,1],SHIFT.POSITION.LIGHT[,y],ylim=c(430,470));
  SHIFT.POSITION.LIGHT.sp.maker=cbind(SHIFT.POSITION.LIGHT[,1],SHIFT.POSITION.LIGHT[,y]);
  SHIFT.POSITION.LIGHT.sp.maker=SHIFT.POSITION.LIGHT.sp.maker[which(!(is.na(SHIFT.POSITION.LIGHT.sp.maker[,2]))),];
  SHIFT.POSITION.LIGHT.sp=smooth.spline(SHIFT.POSITION.LIGHT.sp.maker[,1],SHIFT.POSITION.LIGHT.sp.maker[,2],df=DFforSECONDfit);
  SHIFT.POSITION.LIGHT.spline=cbind(as.numeric(predict(SHIFT.POSITION.LIGHT.sp,seq(-200,3000,1))$x),
                                 as.numeric(predict(SHIFT.POSITION.LIGHT.sp,seq(-200,3000,1))$y));
  matlines(SHIFT.POSITION.LIGHT.spline[,1],SHIFT.POSITION.LIGHT.spline[,2],col="red");
  SHIFT.POSITION.LIGHT.REFINED[,y]=SHIFT.POSITION.LIGHT.spline[,2]}

SHIFT.POSITION.LIGHT.REFINED[,2:ncol(SHIFT.POSITION.LIGHT.REFINED)]=round(SHIFT.POSITION.LIGHT.REFINED[,2:ncol(SHIFT.POSITION.LIGHT.REFINED)],0)
## for cosmetic purposes, whatever we have at 0 is carried over through -200
for(x in 1:199) SHIFT.POSITION.LIGHT.REFINED[x,2:ncol(SHIFT.POSITION.LIGHT.REFINED)]=SHIFT.POSITION.LIGHT.REFINED[200,2:ncol(SHIFT.POSITION.LIGHT.REFINED)]

###
### this block only for display purposes if needed (is using log-transformed values)
#FLATTENED.LIGHT.RETINA.REFINED=FLATTENED.LIGHT.RETINA
#FLATTENED.LIGHT.RETINA.REFINED[,,]<-0
#for(z in 1:dim(FLATTENED.LIGHT.RETINA)[3])
# {REFINE=FLATTENED.LIGHT.RETINA[,,z];
#  for(x in 1:nrow(REFINE))
#   {border=SHIFT.POSITION.LIGHT.REFINED[x,(z+1)];
#    if(border<450) FLATTENED.LIGHT.RETINA.REFINED[x,1:(500+(border-449)),z]=REFINE[x,(450-border):500];
#    if(border==450) FLATTENED.LIGHT.RETINA.REFINED[x,,z]=REFINE[x,];
#    if(border>450) FLATTENED.LIGHT.RETINA.REFINED[x,(1+(border-450)):500,z]=REFINE[x,1:(500-(border-450))]}}
#EXPORT=FLATTENED.LIGHT.RETINA.REFINED
#EXPORT[which(is.na(EXPORT))]=0
#f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste(TO.PROCESS.LIGHT,"_flat",sep=""),size="float")

##
### this actually shifts the real (non-log-transformed) data:
FLATTENED.LIGHT.RETINA.RAW.REFINED=FLATTENED.LIGHT.RETINA.RAW
FLATTENED.LIGHT.RETINA.RAW.REFINED[,,]<-0
for(z in 1:dim(FLATTENED.LIGHT.RETINA.RAW)[3])
 {REFINE=FLATTENED.LIGHT.RETINA.RAW[,,z];
  for(x in 1:nrow(REFINE))
   {border=SHIFT.POSITION.LIGHT.REFINED[x,(z+1)];
    if(border<450) FLATTENED.LIGHT.RETINA.RAW.REFINED[x,1:(500+(border-449)),z]=REFINE[x,(450-border):500];
    if(border==450) FLATTENED.LIGHT.RETINA.RAW.REFINED[x,,z]=REFINE[x,];
    if(border>450) FLATTENED.LIGHT.RETINA.RAW.REFINED[x,(1+(border-450)):500,z]=REFINE[x,1:(500-(border-450))]}}

#################################################
#################################################
#################################################

### now, see how much the flattened images would benefit from being shifted left or right (i.e., along the RPE, to/from the fovea)

SECOND.GRAND.MEAN=FLATTENED.DARK.RETINA.RAW.REFINED[,,1]
for(z in 2:dim(FLATTENED.DARK.RETINA.RAW)[3]) SECOND.GRAND.MEAN=SECOND.GRAND.MEAN+FLATTENED.DARK.RETINA.RAW.REFINED[,,z]
for(z in 2:dim(FLATTENED.LIGHT.RETINA.RAW)[3]) SECOND.GRAND.MEAN=SECOND.GRAND.MEAN+FLATTENED.LIGHT.RETINA.RAW.REFINED[,,z]
SECOND.GRAND.MEAN=SECOND.GRAND.MEAN/(dim(FLATTENED.DARK.RETINA.RAW.REFINED)[3]+dim(FLATTENED.LIGHT.RETINA.RAW.REFINED)[3])


### 
### allow movement of 39 microns (10 of the original pixels) either way. 

SGM=as.matrix(SECOND.GRAND.MEAN[40:(nrow(SECOND.GRAND.MEAN)-39),])
BEST.LAT.MOVE.DARK=cbind(seq(1,dim(FLATTENED.DARK.RETINA.RAW.REFINED)[3],1),seq(1,dim(FLATTENED.DARK.RETINA.RAW.REFINED)[3],1))
for(z in 1:dim(FLATTENED.DARK.RETINA.RAW.REFINED)[3])
 {REFINE=FLATTENED.DARK.RETINA.RAW.REFINED[,,z];
  slide=cbind(seq(-39,39,1),seq(-39,39,1));
  for(x in -39:39) slide[(x+40),2]=cor.test(as.vector(SGM),as.vector(REFINE[(40+x):((nrow(REFINE)-39)+x),]))$est;
  BEST.LAT.MOVE.DARK[z,2]=slide[which(slide[,2]==max(na.rm=T,slide[,2]))[1],1]}

SGM=as.matrix(SECOND.GRAND.MEAN[40:(nrow(SECOND.GRAND.MEAN)-39),])
BEST.LAT.MOVE.LIGHT=cbind(seq(1,dim(FLATTENED.LIGHT.RETINA.RAW.REFINED)[3],1),seq(1,dim(FLATTENED.LIGHT.RETINA.RAW.REFINED)[3],1))
for(z in 1:dim(FLATTENED.LIGHT.RETINA.RAW.REFINED)[3])
 {REFINE=FLATTENED.LIGHT.RETINA.RAW.REFINED[,,z];
  slide=cbind(seq(-39,39,1),seq(-39,39,1));
  for(x in -39:39) slide[(x+40),2]=cor.test(as.vector(SGM),as.vector(REFINE[(40+x):((nrow(REFINE)-39)+x),]))$est;
  BEST.LAT.MOVE.LIGHT[z,2]=slide[which(slide[,2]==max(na.rm=T,slide[,2]))[1],1]}

###
### negative means shift to the left...
### we are now about to crop to the final size of -100 to 2750:
#
# (don't forget to act on markers as well)
#
## "RRC" is RAW.REFINED.CROPPED
FLATTENED.LIGHT.RETINA.RRC=FLATTENED.LIGHT.RETINA.RAW.REFINED[1:2851,,]
FLATTENED.LIGHT.RETINA.RRC[,,]<-NA
FLATTENED.DARK.RETINA.RRC=FLATTENED.DARK.RETINA.RAW.REFINED[1:2851,,]
FLATTENED.DARK.RETINA.RRC[,,]<-NA
FLATTENED.MARKERS.RRC=FLATTENED.MARKERS.REFINED[1:2851,]
FLATTENED.MARKERS.RRC[,]<-NA

## negative 100 is at the position of 100 in the original.
## if we want to shift to the right by 30 (i.e., BEST.LAT.MOVE = +30)
## then we would want to start the reel sooner and sample from 100-BEST.LAT.MOVE to 2950-BEST.LAT.MOVE  ## (2950 because the original reel starts at -200)
## but,
## if we want to shift the reel to the left by 30(i.e., BEST.LAT.MOVE = -30)
## then we would want to start the reel sooner and sample from 100-BEST.LAT.MOVE to 2950-BEST.LAT.MOVE
##   which due to the double negative would be 130:2780
FLATTENED.MARKERS.RRC=FLATTENED.MARKERS.REFINED[(100-BEST.LAT.MOVE.DARK[1,2]):(2950-BEST.LAT.MOVE.DARK[1,2]),]
for(z in 1:nrow(BEST.LAT.MOVE.DARK)) FLATTENED.DARK.RETINA.RRC=FLATTENED.DARK.RETINA.RAW.REFINED[(100-BEST.LAT.MOVE.DARK[z,2]):(2950-BEST.LAT.MOVE.DARK[z,2]),,]
for(z in 1:nrow(BEST.LAT.MOVE.LIGHT)) FLATTENED.LIGHT.RETINA.RRC=FLATTENED.LIGHT.RETINA.RAW.REFINED[(100-BEST.LAT.MOVE.LIGHT[z,2]):(2950-BEST.LAT.MOVE.LIGHT[z,2]),,]


###
### finally, let's verify the position of the RPE.
### ...now that everything's aligned to the average image, we neglected the possibility that whoever marked the RPE was a bit "shy" and
###    was a bit exterior to (or interior to) the true RPE border.

FINAL.GRAND.MEAN=FLATTENED.DARK.RETINA.RRC[,,1]
for(z in 2:dim(FLATTENED.DARK.RETINA.RRC)[3]) FINAL.GRAND.MEAN=FINAL.GRAND.MEAN+FLATTENED.DARK.RETINA.RRC[,,z]
for(z in 2:dim(FLATTENED.LIGHT.RETINA.RRC)[3]) FINAL.GRAND.MEAN=FINAL.GRAND.MEAN+FLATTENED.LIGHT.RETINA.RRC[,,z]
FINAL.GRAND.MEAN=FINAL.GRAND.MEAN/(dim(FLATTENED.DARK.RETINA.RRC)[3]+dim(FLATTENED.LIGHT.RETINA.RRC)[3])

### c
### and we'll calculate the average curve from 500 microns to 2750 microns from the fovea:
GRAND.PROFILE=cbind(seq(1,500,1),seq(1,500,1))
for(x in 1:nrow(GRAND.PROFILE)) GRAND.PROFILE[x,2]=mean(na.rm=T,FINAL.GRAND.MEAN[,x])
## <new for 2023-SEP-05>
# need to make sure that RPE peak is extant and if it is captured
#plot(GRAND.PROFILE[,1],GRAND.PROFILE[,2])
#abline(v=450)
#abline(v=434)
#abline(v=466)
plot(GRAND.PROFILE[,1],GRAND.PROFILE[,2],type="l")
abline(v=450)
abline(v=434)
abline(v=466)
png(filename=file.path(OUTDIR, paste(REFERENCE.DARK,"_find_vertex.png",sep="")), width=1200, height=900)
plot(GRAND.PROFILE[,1],GRAND.PROFILE[,2],type="l")
abline(v=450)
abline(v=434)
abline(v=466)
dev.off()
## </new for 2023-SEP-05>
### isolate 450 +/- 16 pixels (the drawer should have been within 4 original-pixel-widths of the RPE).
GRAND.PROFILE=GRAND.PROFILE[434:466,]
plot(GRAND.PROFILE[,1],GRAND.PROFILE[,2])
check.sp=smooth.spline(GRAND.PROFILE[,1],GRAND.PROFILE[,2],df=10);
check.spline=cbind(as.numeric(predict(check.sp,seq(434,466,1))$x),
                   as.numeric(predict(check.sp,seq(434,466,1))$y),
                   as.numeric(predict(check.sp,seq(434,466,1),deriv = 1)$y));
matlines(check.spline[,1],check.spline[,2]);
##
## so, if there's a flip from negative to positive (i.e., the first peak as one comes into the retina from the choroicapilaris)
## take the first flip. 
### although this shouldn't happen in the lowest quartile of signal values:
threshold=quantile(check.spline[,2],0.25);
check.spline[,3]<-ifelse(check.spline[,2]<threshold,NA,check.spline[,3]);
#check.spline=check.spline[nrow(check.spline):1,]
vertex=check.spline[which(check.spline[,3]>0)[length(which(check.spline[,3]>0))],1]+1;
abline(v=vertex)
## 
## so, we're selecting the farthest-out positive value
png(filename=file.path(OUTDIR, paste(REFERENCE.DARK,"_vertex.png",sep="")), width=1200, height=900)
plot(GRAND.PROFILE[,1],GRAND.PROFILE[,2])
matlines(check.spline[,1],check.spline[,2])
abline(v=vertex)
dev.off()
##
##
## and, now, crop vertically so that vertex is at the same spot across subjects...
## our final images will span from -100 to 2750 from the fovea, and
##  from the RPE peak, interior 430 (i.e., RPE peak resides at column 431) and exterior 30 (total columns=461)

##
##
## <new for 2023-AUG-11>
#  so, a subject came through with unusually not-reflective RPE compared to the OStips, making it hard to find vertex, (length=0)
#  throwing off later cropping. 
if(length(vertex)==0)
 {## find the median slope over this short area... by subtracting that and re-running search
  ## we're NOT asking the most-exterior place where slope changes from pos to negative (or neg to positive, depending on if you're walking "in to" or "out of" the retina)
  ## but we're asking the most-exterior place where slope is-less-negative-than-typical-for-these-retinal-layers
  ## if you'd like, it's similar to subtracting a "wedge" of values from the graph (or rotating it counter-clockwise), and re-trying the prior analysis to find the local peak...
  ## ...but automatically subtracts a wedge (or makes a rotation) that makes half the slopes positive.
  check.spline[,3]=check.spline[,3]-median(na.rm=T,check.spline[,3]);
  vertex=check.spline[which(check.spline[,3]>=0)[length(which(check.spline[,3]>=0))],1]+1;
  abline(v=vertex,col="red");
  png(filename=file.path(OUTDIR, paste(REFERENCE.DARK,"_vertex.png",sep="")), width=1200, height=900)
  plot(GRAND.PROFILE[,1],GRAND.PROFILE[,2])
  matlines(check.spline[,1],check.spline[,2])
  abline(v=vertex,col="red")
  dev.off()}
## </new for 2023-AUG-11>
##
##


FLATTENED.MARKERS.RRC=FLATTENED.MARKERS.RRC[,(vertex-430):(vertex+30)]
FLATTENED.DARK.RETINA.RRC=FLATTENED.DARK.RETINA.RRC[,(vertex-430):(vertex+30),]
FLATTENED.LIGHT.RETINA.RRC=FLATTENED.LIGHT.RETINA.RRC[,(vertex-430):(vertex+30),]
##
## and make some images to help with visualization:


EXPORT=FLATTENED.DARK.RETINA.RRC
EXPORT[which(is.na(EXPORT))]=0
f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste("_flat_",TO.PROCESS.DARK,sep=""),size="float",path.out=OUTDIR)
EXPORT=FLATTENED.LIGHT.RETINA.RRC
EXPORT[which(is.na(EXPORT))]=0
f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste("_flat_",TO.PROCESS.LIGHT,sep=""),size="float",path.out=OUTDIR)

## !!here

#FLATTENED.DARK.RETINA.RRC

export.python.core.arrays()

### NOW, CLEAN UP A BIT
rm(A,BEST.LAT.MOVE.DARK,BEST.LAT.MOVE.LIGHT,bestmove,border)
rm(check,check.sp,check.spline,comparison,end.move,EXPORT,FINAL.GRAND.MEAN,FIRST.GRAND.MEAN)
rm(FLATTENED.DARK.RETINA,FLATTENED.DARK.RETINA.RAW,FLATTENED.DARK.RETINA.RAW.REFINED)
rm(FLATTENED.LIGHT.RETINA,FLATTENED.LIGHT.RETINA.RAW,FLATTENED.LIGHT.RETINA.RAW.REFINED)
rm(FLATTENED.MARKERS,FLATTENED.MARKERS.REFINED,GRAND.PROFILE,LOOK.TO)
rm(MEAN.x,profile,REFINE,REVISE.DARK,REVISE.LIGHT,ROUGH.VIT.RETINA.POSITION,s,SECOND.GRAND.MEAN,SGM)
rm(SHIFT.POSITION.DARK,SHIFT.POSITION.DARK.REFINED,SHIFT.POSITION.DARK.sp,SHIFT.POSITION.DARK.sp.maker,SHIFT.POSITION.DARK.spline)
rm(SHIFT.POSITION.LIGHT,SHIFT.POSITION.LIGHT.REFINED,SHIFT.POSITION.LIGHT.sp,SHIFT.POSITION.LIGHT.sp.maker,SHIFT.POSITION.LIGHT.spline,slide)
rm(start.move,threshold,top.range,vertex,window.factor,window.width.in.pixels,x,Xs,y,Ys,z)


# b.Rdata

save.image(file.path(OUTDIR, paste(TO.PROCESS.DARK,"__and__",TO.PROCESS.LIGHT,"__flat.RData",sep="")))


#################################################
#################################################
#################################################
#################################################   ## now, we need to identify each layer
#################################################
#################################################
#################################################


## first, we're going to strip the relevant info out of FLATTENED.MARKERS.RRC (grab the mean position)
dbg("layer-borders", "Reading hand-marked borders and identifying retinal layers")

## default RPE position is 431, can liberate slightly towards the end...

HAND.BORDERS=matrix(,nrow(FLATTENED.DARK.RETINA.RRC),6)
colnames(HAND.BORDERS)<-c("retina.vit","gcl.rnfl","inl.ipl","onl.opl","olm","rpe")
HAND.BORDERS[,6]<-431

for(x in 1:nrow(FLATTENED.MARKERS.RRC))
 {A=which(FLATTENED.MARKERS.RRC[x,]==254);
  if(length(A)>0) HAND.BORDERS[x,5]=mean(A);
  rm(A)
  A=which(FLATTENED.MARKERS.RRC[x,]==253);
  if(length(A)>0) HAND.BORDERS[x,4]=mean(A);
  rm(A)
  A=which(FLATTENED.MARKERS.RRC[x,]==252);
  if(length(A)>0) HAND.BORDERS[x,3]=mean(A);
  rm(A)
  A=which(FLATTENED.MARKERS.RRC[x,]==250);
  if(length(A)>0) HAND.BORDERS[x,2]=mean(A);
  rm(A)
  A=which(FLATTENED.MARKERS.RRC[x,]==249);
  if(length(A)>0) HAND.BORDERS[x,1]=mean(A)}
rm(A)


## now, we need to go through each image and do a search for the borders.
## dark first...


## the moving window for this part will be 40 um wide (10 original-sized voxel-widths)
window.width.in.pixels=40
start.move=21
end.move=(nrow(FLATTENED.DARK.RETINA.RRC)-start.move)-1
MEAN.x=function(x) mean(na.rm=TRUE,x)
window.factor=matrix(,ncol(FLATTENED.DARK.RETINA.RRC),window.width.in.pixels);
for(y in 1:nrow(window.factor)) window.factor[y,]=y
window.factor=t(window.factor)
window.factor=as.factor(window.factor)

HAND.BORDERS.factor=matrix(,6,window.width.in.pixels);
for(y in 1:nrow(HAND.BORDERS.factor)) HAND.BORDERS.factor[y,]=y
HAND.BORDERS.factor=t(HAND.BORDERS.factor)
HAND.BORDERS.factor=as.factor(HAND.BORDERS.factor)

##f.Rdata


TRUE.BORDERS.DARK=FLATTENED.DARK.RETINA.RRC[,1:6,]
TRUE.BORDERS.DARK[,,]<-NA
## would have column names as HAND.BORDERS
BLANK=TRUE.BORDERS.DARK[1,,1]
BLANK[6]=431
for(z in 1:dim(FLATTENED.DARK.RETINA.RRC)[3])
{REVIEW=FLATTENED.DARK.RETINA.RRC[,,z];
 for(x in start.move:end.move)
   {NEW.VALUES=BLANK;
    profile=cbind(seq(1,461,1),as.vector(tapply(REVIEW[(x-19):(x+20),],window.factor,MEAN.x)));
    SEGMENT=round(as.vector(tapply(HAND.BORDERS[(x-19):(x+20),],HAND.BORDERS.factor,MEAN.x)));
    #plot(profile[,1],profile[,2]);
    #abline(v=SEGMENT);
    if(!(is.na(SEGMENT[1])))
     {## for vit.retina, grab +/- 20 microns (to find local peak/trough); actual revision of hand-drawn border will just be +/- 10 pixels
      ## this uses half-height method
      check=profile[(SEGMENT[1]-20):(SEGMENT[1]+20),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)[11:31,];
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==1))>0) NEW.VALUES[1]=check[which(check[,4]==1)[1],1] else NEW.VALUES[1]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[2])))
     {## for gcl.rnfl, grab from NEW.VALUES[1] to the hand border +20 microns [OR LESS] (to find local peak/trough); actual revision of hand-drawn border will just be somehwere between the local peak (center of RNFL) and +10 pixels
      ## this uses half-height method
      ## 
      ## the [OR LESS] denotes the possibility that there's a SEGMENT[3] close by, in which case only move in half that distance.
      MOVEIN=20;
      if(!(is.na(SEGMENT[3]))) {MOVEINalt=ceiling(SEGMENT[3]-SEGMENT[2])/2; if(MOVEINalt<20) MOVEIN=MOVEINalt};
      check=profile[NEW.VALUES[1]:(SEGMENT[2]+MOVEIN),];
      if(NEW.VALUES[1]>SEGMENT[2]) SEGMENT[2]=NEW.VALUES[1];   ###<<<------ADDITION FOR JUL_20 onward
      ##
      ## odd thing happens here... sometimes the very surface of the retina is *extremely* reflective,
      ## and it throws off the "half-height" calculation a bit, so, log-transform these values first to blunt the effect...
      check[,2]<-log(check[,2]);
      ## find the peak between SEGMENT[2] and NEW.VALUES[1], and toss out anything interior to it.
      M=max(na.rm=T,check[1:which(check[,1]==SEGMENT[2]),2]);
      check=check[which(check[1:which(check[,1]==SEGMENT[2]),2]==M)[1]:nrow(check),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)
      if( (which(check[,1]==SEGMENT[2])+10)<nrow(check) ) check=check[1:(which(check[,1]==SEGMENT[2])+10),];
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==-1))>0) NEW.VALUES[2]=check[which(check[,4]==-1)[1],1] else NEW.VALUES[2]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[3])))
     {## for inl.ipl, grab the hand border +20 microns [OR LESS] (to find local peak/trough); actual revision of hand-drawn border will just be +/-10 pixels
      ## this uses half-height method
      ##
      ## the [OR LESS] denotes the possibility that there's a SEGMENT[3] close by, in which case only move in half that distance (in either direction; this is the sign of cramped quarters)
      MOVEIN=20;
      if(!(is.na(SEGMENT[4]))) {MOVEINalt=ceiling(SEGMENT[4]-SEGMENT[3])/2; if(MOVEINalt<20) MOVEIN=MOVEINalt};
      check=profile[(SEGMENT[3]-MOVEIN):(SEGMENT[3]+MOVEIN),];
      ## find the peak between the start and the hand-drawn border...
      ## we can safely toss things internal to this
      M=max(na.rm=T,check[1:(MOVEIN+1),2]);
      check=check[which(check[1:(MOVEIN+1),2]==M)[1]:nrow(check),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)
      if( (which(check[,1]==SEGMENT[3])+10)<nrow(check) ) check=check[1:(which(check[,1]==SEGMENT[3])+10),];
      if( (which(check[,1]==SEGMENT[3])-10)>1 ) check=check[(which(check[,1]==SEGMENT[3])-10):nrow(check),];
      if(length(check)==4) check=rbind(check,check);
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==-1))>0) NEW.VALUES[3]=check[which(check[,4]==-1)[1],1] else NEW.VALUES[3]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[4])))
     {## for onl.opl, grab the hand border +20 microns [OR LESS] (to find local peak/trough); actual revision of hand-drawn border will just be +/-10 pixels
      ## this uses half-height method
      ##
      ## the [OR LESS] denotes the possibility that there's a SEGMENT[3] close by, in which case only move in half that distance (in either direction; this is the sign of cramped quarters)
      MOVEIN=20;
      if(!(is.na(SEGMENT[5]))) {MOVEINalt=ceiling(SEGMENT[5]-SEGMENT[4])/2; if(MOVEINalt<20) MOVEIN=MOVEINalt};
      check=profile[(SEGMENT[4]-MOVEIN):(SEGMENT[4]+MOVEIN),];
      ## find the peak between the start and the hand-drawn border...
      ## we can safely toss things internal to this
      M=max(na.rm=T,check[1:(MOVEIN+1),2]);
      check=check[which(check[1:(MOVEIN+1),2]==M)[1]:nrow(check),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)
      if( (which(check[,1]==SEGMENT[4])+10)<nrow(check) ) check=check[1:(which(check[,1]==SEGMENT[4])+10),];
      if( (which(check[,1]==SEGMENT[4])-10)>1 ) check=check[(which(check[,1]==SEGMENT[4])-10):nrow(check),];
      if(length(check)==4) check=rbind(check,check);
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==-1))>0) NEW.VALUES[4]=check[which(check[,4]==-1)[1],1] else NEW.VALUES[4]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[5])))
     {## for olm, we "just" want the local peak
      check=profile[(SEGMENT[5]-5):(SEGMENT[5]+5),];
      ## find the peak within 5 microns of the hand-drawn border...
      M=max(na.rm=T,check[,2]);
      localpeak=round(mean(na.rm=T,check[which(check[,2]==M),1]));
      ## OK... so, if the localpeak is between rows 3 and 9, then grab +/- 2 microns, and fit a parabola to those five points...
      ## and accept the vertex as the best estimate ONLY IF it would fall within the same range (rows 3 and 9). 
      ## otherwise, just log localpeak in NEW.VALUES... ...so do that first, and overwrite if the contingency allows:
      NEW.VALUES[5]=localpeak;
      LOW=check[3,1];  
      HIGH=check[10,1];
      if( (localpeak>LOW)&(localpeak<HIGH) )
        {check=check[(which(check[,1]==localpeak)-2):(which(check[,1]==localpeak)+2),];
         c.sp=smooth.spline(check[,1],check[,2],df=2);
         cspline=cbind(seq(check[1,1],check[5,1],0.1),as.numeric(predict(c.sp,seq(check[1,1],check[5,1],0.1))$y));
         vertex=mean(cspline[which(cspline[,2]==max(cspline[,2])),1]);
         if( (vertex>LOW)&(vertex<HIGH) )  NEW.VALUES[5]=round(vertex,1)};
      rm(M,localpeak,LOW,HIGH,check)};
    TRUE.BORDERS.DARK[x,,z]=NEW.VALUES}}


###
###
### now do light....

TRUE.BORDERS.LIGHT=FLATTENED.LIGHT.RETINA.RRC[,1:6,]
TRUE.BORDERS.LIGHT[,,]<-NA
## would have column names as HAND.BORDERS
BLANK=TRUE.BORDERS.LIGHT[1,,1]
BLANK[6]=431
for(z in 1:dim(FLATTENED.LIGHT.RETINA.RRC)[3])
{REVIEW=FLATTENED.LIGHT.RETINA.RRC[,,z];
 for(x in start.move:end.move)
   {NEW.VALUES=BLANK;
    profile=cbind(seq(1,461,1),as.vector(tapply(REVIEW[(x-19):(x+20),],window.factor,MEAN.x)));
    SEGMENT=round(as.vector(tapply(HAND.BORDERS[(x-19):(x+20),],HAND.BORDERS.factor,MEAN.x)));
    #plot(profile[,1],profile[,2]);
    #abline(v=SEGMENT);
    if(!(is.na(SEGMENT[1])))
     {## for vit.retina, grab +/- 20 microns (to find local peak/trough); actual revision of hand-drawn border will just be +/- 10 pixels
      ## this uses half-height method
      check=profile[(SEGMENT[1]-20):(SEGMENT[1]+20),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)[11:31,];
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==1))>0) NEW.VALUES[1]=check[which(check[,4]==1)[1],1] else NEW.VALUES[1]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[2])))
     {## for gcl.rnfl, grab from NEW.VALUES[1] to the hand border +20 microns [OR LESS] (to find local peak/trough); actual revision of hand-drawn border will just be somehwere between the local peak (center of RNFL) and +10 pixels
      ## this uses half-height method
      ## 
      ## the [OR LESS] denotes the possibility that there's a SEGMENT[3] close by, in which case only move in half that distance.
      MOVEIN=20;
      if(!(is.na(SEGMENT[3]))) {MOVEINalt=ceiling(SEGMENT[3]-SEGMENT[2])/2; if(MOVEINalt<20) MOVEIN=MOVEINalt};
      check=profile[NEW.VALUES[1]:(SEGMENT[2]+MOVEIN),];
      if(NEW.VALUES[1]>SEGMENT[2]) SEGMENT[2]=NEW.VALUES[1];   ###<<<------ADDITION FOR JUL_20 onward
      ##
      ## odd thing happens here... sometimes the very surface of the retina is *extremely* reflective,
      ## and it throws off the "half-height" calculation a bit, so, log-transform these values first to blunt the effect...
      check[,2]<-log(check[,2]);
      ## find the peak between SEGMENT[2] and NEW.VALUES[1], and toss out anything interior to it.
      M=max(na.rm=T,check[1:which(check[,1]==SEGMENT[2]),2]);
      check=check[which(check[1:which(check[,1]==SEGMENT[2]),2]==M)[1]:nrow(check),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)
      if( (which(check[,1]==SEGMENT[2])+10)<nrow(check) ) check=check[1:(which(check[,1]==SEGMENT[2])+10),];
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==-1))>0) NEW.VALUES[2]=check[which(check[,4]==-1)[1],1] else NEW.VALUES[2]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[3])))
     {## for inl.ipl, grab the hand border +20 microns [OR LESS] (to find local peak/trough); actual revision of hand-drawn border will just be +/-10 pixels
      ## this uses half-height method
      ##
      ## the [OR LESS] denotes the possibility that there's a SEGMENT[3] close by, in which case only move in half that distance (in either direction; this is the sign of cramped quarters)
      MOVEIN=20;
      if(!(is.na(SEGMENT[4]))) {MOVEINalt=ceiling(SEGMENT[4]-SEGMENT[3])/2; if(MOVEINalt<20) MOVEIN=MOVEINalt};
      check=profile[(SEGMENT[3]-MOVEIN):(SEGMENT[3]+MOVEIN),];
      ## find the peak between the start and the hand-drawn border...
      ## we can safely toss things internal to this
      M=max(na.rm=T,check[1:(MOVEIN+1),2]);
      check=check[which(check[1:(MOVEIN+1),2]==M)[1]:nrow(check),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)
      if( (which(check[,1]==SEGMENT[3])+10)<nrow(check) ) check=check[1:(which(check[,1]==SEGMENT[3])+10),];
      if( (which(check[,1]==SEGMENT[3])-10)>1 ) check=check[(which(check[,1]==SEGMENT[3])-10):nrow(check),];
      if(length(check)==4) check=rbind(check,check);
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==-1))>0) NEW.VALUES[3]=check[which(check[,4]==-1)[1],1] else NEW.VALUES[3]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[4])))
     {## for onl.opl, grab the hand border +20 microns [OR LESS] (to find local peak/trough); actual revision of hand-drawn border will just be +/-10 pixels
      ## this uses half-height method
      ##
      ## the [OR LESS] denotes the possibility that there's a SEGMENT[3] close by, in which case only move in half that distance (in either direction; this is the sign of cramped quarters)
      MOVEIN=20;
      if(!(is.na(SEGMENT[5]))) {MOVEINalt=ceiling(SEGMENT[5]-SEGMENT[4])/2; if(MOVEINalt<20) MOVEIN=MOVEINalt};
      check=profile[(SEGMENT[4]-MOVEIN):(SEGMENT[4]+MOVEIN),];
      ## find the peak between the start and the hand-drawn border...
      ## we can safely toss things internal to this
      M=max(na.rm=T,check[1:(MOVEIN+1),2]);
      check=check[which(check[1:(MOVEIN+1),2]==M)[1]:nrow(check),];
      half=(min(na.rm=T,check[,2])+max(na.rm=T,check[,2]))/2;
      check=cbind(check,check)
      if( (which(check[,1]==SEGMENT[4])+10)<nrow(check) ) check=check[1:(which(check[,1]==SEGMENT[4])+10),];
      if( (which(check[,1]==SEGMENT[4])-10)>1 ) check=check[(which(check[,1]==SEGMENT[4])-10):nrow(check),];
      if(length(check)==4) check=rbind(check,check);
      check[,3]<-check[,2]-half;
      check[,4]<-sign(check[,3]);
      if(length(which(check[,4]==-1))>0) NEW.VALUES[4]=check[which(check[,4]==-1)[1],1] else NEW.VALUES[4]=check[which(abs(check[,3])==min(na.rm=T,abs(check[,3])))[1],1];
      rm(half,check)};
    if(!(is.na(SEGMENT[5])))
     {## for olm, we "just" want the local peak
      check=profile[(SEGMENT[5]-5):(SEGMENT[5]+5),];
      ## find the peak within 5 microns of the hand-drawn border...
      M=max(na.rm=T,check[,2]);
      localpeak=round(mean(na.rm=T,check[which(check[,2]==M),1]));
      ## OK... so, if the localpeak is between rows 3 and 9, then grab +/- 2 microns, and fit a parabola to those five points...
      ## and accept the vertex as the best estimate ONLY IF it would fall within the same range (rows 3 and 9). 
      ## otherwise, just log localpeak in NEW.VALUES... ...so do that first, and overwrite if the contingency allows:
      NEW.VALUES[5]=localpeak;
      LOW=check[3,1];  
      HIGH=check[10,1];
      if( (localpeak>LOW)&(localpeak<HIGH) )
        {check=check[(which(check[,1]==localpeak)-2):(which(check[,1]==localpeak)+2),];
         c.sp=smooth.spline(check[,1],check[,2],df=2);
         cspline=cbind(seq(check[1,1],check[5,1],0.1),as.numeric(predict(c.sp,seq(check[1,1],check[5,1],0.1))$y));
         vertex=mean(cspline[which(cspline[,2]==max(cspline[,2])),1]);
         if( (vertex>LOW)&(vertex<HIGH) )  NEW.VALUES[5]=round(vertex,1)};
      rm(M,localpeak,LOW,HIGH,check)};
    TRUE.BORDERS.LIGHT[x,,z]=NEW.VALUES}}


##
## now, we can go through and see if the RPE needs any revision. 
## this is very narrowly constrained: we're allowed to move it < 0.5 microns either way, based on the adjacent values AVERAGED OVER 500 to 2750
## basically, we average the strip, look at the values for 429:433 (columns in FLATTENED.LIGHT.RETINA.RRC and FLATTENED.DARK)
## fit a smooth spline with 3 df, and see where the peak is (between 

for(z in 1:dim(FLATTENED.DARK.RETINA.RRC)[3])
  {check=cbind(seq(1,461,1),seq(1,461,1));
   check[,2]<-NA;
   for(x in 429:433) check[x,2]=mean(na.rm=T,FLATTENED.DARK.RETINA.RRC[600:dim(FLATTENED.DARK.RETINA.RRC)[2],x,z]);
   check=check[429:433,];
   c.sp=smooth.spline(check[,1],check[,2],df=3);
   cspline=cbind(seq(430.6,431.4,0.1),as.numeric(predict(c.sp,seq(430.6,431.4,0.1))$y));
   peak=mean(cspline[which(cspline[,2]==max(cspline[,2])),1]);
   #TRUE.BORDERS.DARK[,6,z]<-peak}    ## by commenting-out this line instead of the next, we force RPE to not move even a little
   TRUE.BORDERS.DARK[,6,z]<-431}

for(z in 1:dim(FLATTENED.LIGHT.RETINA.RRC)[3])
  {check=cbind(seq(1,461,1),seq(1,461,1));
   check[,2]<-NA;
   for(x in 429:433) check[x,2]=mean(na.rm=T,FLATTENED.LIGHT.RETINA.RRC[600:dim(FLATTENED.LIGHT.RETINA.RRC)[2],x,z]);
   check=check[429:433,];
   c.sp=smooth.spline(check[,1],check[,2],df=3);
   cspline=cbind(seq(430.6,431.4,0.1),as.numeric(predict(c.sp,seq(430.6,431.4,0.1))$y));
   peak=mean(cspline[which(cspline[,2]==max(cspline[,2])),1]);
   #TRUE.BORDERS.LIGHT[,6,z]<-peak}    ## by commenting-out this line instead of the next, we force RPE to not move even a little
   TRUE.BORDERS.LIGHT[,6,z]<-431}

##
##
## visualize borders so far:

VITREOUS.RETINA.POSITION.DARK=TRUE.BORDERS.DARK[,1,]
RNFL.GCL.POSITION.DARK=TRUE.BORDERS.DARK[,2,]
INL.IPL.POSITION.DARK=TRUE.BORDERS.DARK[,3,]
ONL.OPL.POSITION.DARK=TRUE.BORDERS.DARK[,4,]
OLM.POSITION.DARK=TRUE.BORDERS.DARK[,5,]
RPE.POSITION.DARK=TRUE.BORDERS.DARK[,6,]

VITREOUS.RETINA.POSITION.LIGHT=TRUE.BORDERS.LIGHT[,1,]
RNFL.GCL.POSITION.LIGHT=TRUE.BORDERS.LIGHT[,2,]
INL.IPL.POSITION.LIGHT=TRUE.BORDERS.LIGHT[,3,]
ONL.OPL.POSITION.LIGHT=TRUE.BORDERS.LIGHT[,4,]
OLM.POSITION.LIGHT=TRUE.BORDERS.LIGHT[,5,]
RPE.POSITION.LIGHT=TRUE.BORDERS.LIGHT[,6,]

x=1
plot(VITREOUS.RETINA.POSITION.DARK[,x],ylim=c(0,450),typ="l")
matlines(VITREOUS.RETINA.POSITION.LIGHT[,x],lwd=2)
matlines(RNFL.GCL.POSITION.DARK[,x],col="red")
matlines(RNFL.GCL.POSITION.LIGHT[,x],col="red",lwd=2)
matlines(INL.IPL.POSITION.DARK[,x],col="blue")
matlines(INL.IPL.POSITION.LIGHT[,x],col="blue",lwd=2)
matlines(ONL.OPL.POSITION.DARK[,x],col="red")
matlines(ONL.OPL.POSITION.LIGHT[,x],col="red",lwd=2)
matlines(OLM.POSITION.DARK[,x],col="blue")
matlines(OLM.POSITION.LIGHT[,x],col="blue",lwd=2)
matlines(RPE.POSITION.DARK[,x],col="red")
matlines(RPE.POSITION.LIGHT[,x],col="red",lwd=2)




##
## d.Rdata
## next, use splines to extend the borders out to the end of the retinal strip, and to find any errors....

## first, a reflection on the whole point of all of this:
##  we want good RPE, OLM, and (whatever the next one up is, usually RNFL/vitreous) for -50 to +50 form the fovea, a little 100 micron block mini-profile.
##  and we want each border to be present for 500 to 2750, and to look great, since that's how we'll resample the main retina.
##  ...to this end, we can cut off a lot of VITREOUS.RETINA.POSITION.DARK, etc., before we proceed:

VITREOUS.RETINA.POSITION.DARK=VITREOUS.RETINA.POSITION.DARK[600:nrow(VITREOUS.RETINA.POSITION.DARK),]
RNFL.GCL.POSITION.DARK=RNFL.GCL.POSITION.DARK[600:nrow(RNFL.GCL.POSITION.DARK),]
INL.IPL.POSITION.DARK=INL.IPL.POSITION.DARK[600:nrow(INL.IPL.POSITION.DARK),]
ONL.OPL.POSITION.DARK=ONL.OPL.POSITION.DARK[600:nrow(ONL.OPL.POSITION.DARK),]
OLM.POSITION.DARK=OLM.POSITION.DARK[600:nrow(OLM.POSITION.DARK),]
RPE.POSITION.DARK=RPE.POSITION.DARK[600:nrow(RPE.POSITION.DARK),]

VITREOUS.RETINA.POSITION.LIGHT=VITREOUS.RETINA.POSITION.LIGHT[600:nrow(VITREOUS.RETINA.POSITION.LIGHT),]
RNFL.GCL.POSITION.LIGHT=RNFL.GCL.POSITION.LIGHT[600:nrow(RNFL.GCL.POSITION.LIGHT),]
INL.IPL.POSITION.LIGHT=INL.IPL.POSITION.LIGHT[600:nrow(INL.IPL.POSITION.LIGHT),]
ONL.OPL.POSITION.LIGHT=ONL.OPL.POSITION.LIGHT[600:nrow(ONL.OPL.POSITION.LIGHT),]
OLM.POSITION.LIGHT=OLM.POSITION.LIGHT[600:nrow(OLM.POSITION.LIGHT),]
RPE.POSITION.LIGHT=RPE.POSITION.LIGHT[600:nrow(RPE.POSITION.LIGHT),]

##
## and borrow some code from the mouse script:



####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $   DARK FIRST
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $



z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.DARK)
for(x in 1:dim(VITREOUS.RETINA.POSITION.DARK)[2])
 {LEFT=cbind(RPE.POSITION.DARK[1:split,x],
             OLM.POSITION.DARK[1:split,x],
             VITREOUS.RETINA.POSITION.DARK[1:split,x],
             ONL.OPL.POSITION.DARK[1:split,x],
             INL.IPL.POSITION.DARK[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.DARK
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  INL.IPL.POSITION.DARK[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}

### now, repeat that process for the other borders



### now, repeat that process for the other borders
z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.DARK)
for(x in 1:dim(VITREOUS.RETINA.POSITION.DARK)[2])
 {LEFT=cbind(INL.IPL.POSITION.DARK[1:split,x],
             RPE.POSITION.DARK[1:split,x],
             OLM.POSITION.DARK[1:split,x],
             VITREOUS.RETINA.POSITION.DARK[1:split,x],
             ONL.OPL.POSITION.DARK[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.DARK
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  ONL.OPL.POSITION.DARK[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}



## next!
z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.DARK)
for(x in 1:dim(VITREOUS.RETINA.POSITION.DARK)[2])
 {LEFT=cbind(ONL.OPL.POSITION.DARK[1:split,x],
             INL.IPL.POSITION.DARK[1:split,x],
             RPE.POSITION.DARK[1:split,x],
             OLM.POSITION.DARK[1:split,x],
             VITREOUS.RETINA.POSITION.DARK[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.DARK
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  VITREOUS.RETINA.POSITION.DARK[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}





### for the OLM, set z-threshold to 3
z.threshold=3
split=nrow(VITREOUS.RETINA.POSITION.DARK)
for(x in 1:dim(VITREOUS.RETINA.POSITION.DARK)[2])
 {LEFT=cbind(VITREOUS.RETINA.POSITION.DARK[1:split,x],
             ONL.OPL.POSITION.DARK[1:split,x],
             INL.IPL.POSITION.DARK[1:split,x],
             RPE.POSITION.DARK[1:split,x],
             OLM.POSITION.DARK[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.DARK
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  OLM.POSITION.DARK[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}




### z.threshold back down to 2
z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.DARK)
for(x in 1:dim(VITREOUS.RETINA.POSITION.DARK)[2])
 {LEFT=cbind(VITREOUS.RETINA.POSITION.DARK[1:split,x],
             ONL.OPL.POSITION.DARK[1:split,x],
             INL.IPL.POSITION.DARK[1:split,x],
             RPE.POSITION.DARK[1:split,x],
             OLM.POSITION.DARK[1:split,x],
             RNFL.GCL.POSITION.DARK[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,5]<-LEFT[,5]-LEFT[,6];
  LEFT[,4]<-LEFT[,4]-LEFT[,6];
  LEFT[,3]<-LEFT[,3]-LEFT[,6];
  LEFT[,2]<-LEFT[,2]-LEFT[,6];
  LEFT[,1]<-LEFT[,1]-LEFT[,6];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,5]);
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  #
  FIVE=cbind(seq(1,split,1), LEFT[,5]);
  FIVE=FIVE[which(!(is.na(FIVE[,2]))),];
  MODEL.5=smooth.spline(FIVE[,1],FIVE[,2],df=5);
  RESULT.5=as.numeric(predict(MODEL.5,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.5,col="red");
  DEVIATION.5=LEFT[,5]-RESULT.5;
  z.DEVIATION.5=DEVIATION.5;
  z.DEVIATION.5=(z.DEVIATION.5-mean(na.rm=T,DEVIATION.5))/sd(na.rm=T,DEVIATION.5);
  #
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.5,z.DEVIATION.1);
  overall.z[,7]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5]+overall.z[,6])/5;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  FIVE=cbind(seq(1,split,1), LEFT[,5]);
  #FIVE=FIVE[which(!(is.na(FIVE[,2]))),];
  MODEL.5=smooth.spline(FIVE[which(overall.z[,ncol(overall.z)]==0),1],FIVE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.5=as.numeric(predict(MODEL.5,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,6]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.DARK
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,6]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,6]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,6]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  GIVEBACK.5=cbind(LEFT[,5],RESULT.5);
  GIVEBACK.5[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.5[,2],GIVEBACK.5[,1]);
  GIVEBACK.5=cbind(GIVEBACK.5,LEFT.ORIG[,5],LEFT[,6]);
  GIVEBACK.5[,4]<-(GIVEBACK.5[,3]-GIVEBACK.5[,1]);
  ###
  ### and give it back!
  RNFL.GCL.POSITION.DARK[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4]+GIVEBACK.5[,4])/5}




####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $  REPEAT FOR LIGHT
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $





z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.LIGHT)
for(x in 1:dim(VITREOUS.RETINA.POSITION.LIGHT)[2])
 {LEFT=cbind(RPE.POSITION.LIGHT[1:split,x],
             OLM.POSITION.LIGHT[1:split,x],
             VITREOUS.RETINA.POSITION.LIGHT[1:split,x],
             ONL.OPL.POSITION.LIGHT[1:split,x],
             INL.IPL.POSITION.LIGHT[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.LIGHT
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  INL.IPL.POSITION.LIGHT[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}

### now, repeat that process for the other borders



### now, repeat that process for the other borders
z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.LIGHT)
for(x in 1:dim(VITREOUS.RETINA.POSITION.LIGHT)[2])
 {LEFT=cbind(INL.IPL.POSITION.LIGHT[1:split,x],
             RPE.POSITION.LIGHT[1:split,x],
             OLM.POSITION.LIGHT[1:split,x],
             VITREOUS.RETINA.POSITION.LIGHT[1:split,x],
             ONL.OPL.POSITION.LIGHT[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.LIGHT
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  ONL.OPL.POSITION.LIGHT[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}



## next!
z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.LIGHT)
for(x in 1:dim(VITREOUS.RETINA.POSITION.LIGHT)[2])
 {LEFT=cbind(ONL.OPL.POSITION.LIGHT[1:split,x],
             INL.IPL.POSITION.LIGHT[1:split,x],
             RPE.POSITION.LIGHT[1:split,x],
             OLM.POSITION.LIGHT[1:split,x],
             VITREOUS.RETINA.POSITION.LIGHT[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.LIGHT
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  VITREOUS.RETINA.POSITION.LIGHT[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}





### for the OLM, set z-threshold to 3
z.threshold=3
split=nrow(VITREOUS.RETINA.POSITION.LIGHT)
for(x in 1:dim(VITREOUS.RETINA.POSITION.LIGHT)[2])
 {LEFT=cbind(VITREOUS.RETINA.POSITION.LIGHT[1:split,x],
             ONL.OPL.POSITION.LIGHT[1:split,x],
             INL.IPL.POSITION.LIGHT[1:split,x],
             RPE.POSITION.LIGHT[1:split,x],
             OLM.POSITION.LIGHT[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,4]<-LEFT[,4]-LEFT[,5];
  LEFT[,3]<-LEFT[,3]-LEFT[,5];
  LEFT[,2]<-LEFT[,2]-LEFT[,5];
  LEFT[,1]<-LEFT[,1]-LEFT[,5];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.1);
  overall.z[,6]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5])/4;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,5]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.LIGHT
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,5]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,5]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,5]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  ###
  ### and give it back!
  OLM.POSITION.LIGHT[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4])/4}




### z.threshold back down to 2
z.threshold=2
split=nrow(VITREOUS.RETINA.POSITION.LIGHT)
for(x in 1:dim(VITREOUS.RETINA.POSITION.LIGHT)[2])
 {LEFT=cbind(VITREOUS.RETINA.POSITION.LIGHT[1:split,x],
             ONL.OPL.POSITION.LIGHT[1:split,x],
             INL.IPL.POSITION.LIGHT[1:split,x],
             RPE.POSITION.LIGHT[1:split,x],
             OLM.POSITION.LIGHT[1:split,x],
             RNFL.GCL.POSITION.LIGHT[1:split,x]);
  LEFT.ORIG=LEFT;
  LEFT[,5]<-LEFT[,5]-LEFT[,6];
  LEFT[,4]<-LEFT[,4]-LEFT[,6];
  LEFT[,3]<-LEFT[,3]-LEFT[,6];
  LEFT[,2]<-LEFT[,2]-LEFT[,6];
  LEFT[,1]<-LEFT[,1]-LEFT[,6];
  plot(seq(1,split,1),LEFT[,4],ylim=c(min(na.rm=T,LEFT[,1:4]),max(na.rm=T,LEFT[,1:4])));
  matlines(seq(1,split,1),LEFT[,5]);
  matlines(seq(1,split,1),LEFT[,4]);
  matlines(seq(1,split,1),LEFT[,3]);
  matlines(seq(1,split,1),LEFT[,2]);
  matlines(seq(1,split,1),LEFT[,1]);
  #
  FIVE=cbind(seq(1,split,1), LEFT[,5]);
  FIVE=FIVE[which(!(is.na(FIVE[,2]))),];
  MODEL.5=smooth.spline(FIVE[,1],FIVE[,2],df=5);
  RESULT.5=as.numeric(predict(MODEL.5,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.5,col="red");
  DEVIATION.5=LEFT[,5]-RESULT.5;
  z.DEVIATION.5=DEVIATION.5;
  z.DEVIATION.5=(z.DEVIATION.5-mean(na.rm=T,DEVIATION.5))/sd(na.rm=T,DEVIATION.5);
  #
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[,1],FOUR[,2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.4,col="red");
  DEVIATION.4=LEFT[,4]-RESULT.4
  z.DEVIATION.4=DEVIATION.4;
  z.DEVIATION.4=(z.DEVIATION.4-mean(na.rm=T,DEVIATION.4))/sd(na.rm=T,DEVIATION.4);
  #
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[,1],THREE[,2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.3,col="red");
  DEVIATION.3=LEFT[,3]-RESULT.3
  z.DEVIATION.3=DEVIATION.3;
  z.DEVIATION.3=(z.DEVIATION.3-mean(na.rm=T,DEVIATION.3))/sd(na.rm=T,DEVIATION.3);
  #
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[,1],TWO[,2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.2,col="red");
  DEVIATION.2=LEFT[,2]-RESULT.2
  z.DEVIATION.2=DEVIATION.2;
  z.DEVIATION.2=(z.DEVIATION.2-mean(na.rm=T,DEVIATION.2))/sd(na.rm=T,DEVIATION.2);
  #
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[,1],ONE[,2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  matlines(seq(1,split,1),RESULT.1,col="red");
  DEVIATION.1=LEFT[,1]-RESULT.1
  z.DEVIATION.1=DEVIATION.1;
  z.DEVIATION.1=(z.DEVIATION.1-mean(na.rm=T,DEVIATION.1))/sd(na.rm=T,DEVIATION.1);
  # 
  # 
  overall.z=cbind(seq(1,split,1),z.DEVIATION.1,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.5,z.DEVIATION.1);
  overall.z[,7]<-abs(overall.z[,2]+overall.z[,3]+overall.z[,4]+overall.z[,5]+overall.z[,6])/5;
  ## 
  ## if abs(overall.z) > z.threshold, replace the misbehaving data in RPE.POSITION with whatever a z score of 0 would be. 
  overall.z=cbind(overall.z,overall.z[,1]);
  overall.z[,ncol(overall.z)]<-ifelse(overall.z[,(ncol(overall.z)-1)]>z.threshold,1,0);
  ##
  ##
  ## ### now, re-calculate RESULT without that above-threshold data to see what we should replace it with:
  ##
  ONE=cbind(seq(1,split,1), LEFT[,1]);
  #ONE=ONE[which(!(is.na(ONE[,2]))),];
  MODEL.1=smooth.spline(ONE[which(overall.z[,ncol(overall.z)]==0),1],ONE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.1=as.numeric(predict(MODEL.1,seq(1,split,1))$y);
  TWO=cbind(seq(1,split,1), LEFT[,2]);
  #TWO=TWO[which(!(is.na(TWO[,2]))),];
  MODEL.2=smooth.spline(TWO[which(overall.z[,ncol(overall.z)]==0),1],TWO[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.2=as.numeric(predict(MODEL.2,seq(1,split,1))$y);
  THREE=cbind(seq(1,split,1), LEFT[,3]);
  #THREE=THREE[which(!(is.na(THREE[,2]))),];
  MODEL.3=smooth.spline(THREE[which(overall.z[,ncol(overall.z)]==0),1],THREE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.3=as.numeric(predict(MODEL.3,seq(1,split,1))$y);
  FOUR=cbind(seq(1,split,1), LEFT[,4]);
  #FOUR=FOUR[which(!(is.na(FOUR[,2]))),];
  MODEL.4=smooth.spline(FOUR[which(overall.z[,ncol(overall.z)]==0),1],FOUR[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.4=as.numeric(predict(MODEL.4,seq(1,split,1))$y);
  FIVE=cbind(seq(1,split,1), LEFT[,5]);
  #FIVE=FIVE[which(!(is.na(FIVE[,2]))),];
  MODEL.5=smooth.spline(FIVE[which(overall.z[,ncol(overall.z)]==0),1],FIVE[which(overall.z[,ncol(overall.z)]==0),2],df=5);
  RESULT.5=as.numeric(predict(MODEL.5,seq(1,split,1))$y);
  ##
  ## ### 
  ##
  ##
  GIVEBACK.1=cbind(LEFT[,1],RESULT.1);
  GIVEBACK.1[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.1[,2],GIVEBACK.1[,1]);
  GIVEBACK.1=cbind(GIVEBACK.1,LEFT.ORIG[,1],LEFT[,6]);
  GIVEBACK.1[,4]<-(GIVEBACK.1[,3]-GIVEBACK.1[,1]);
  ## now, GIVEBACK.1[,4] is one of four estimates we can generate for replacing RPE.POSITION.LIGHT
  ## let's make the other three
  GIVEBACK.2=cbind(LEFT[,2],RESULT.2);
  GIVEBACK.2[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.2[,2],GIVEBACK.2[,1]);
  GIVEBACK.2=cbind(GIVEBACK.2,LEFT.ORIG[,2],LEFT[,6]);
  GIVEBACK.2[,4]<-(GIVEBACK.2[,3]-GIVEBACK.2[,1]);
  GIVEBACK.3=cbind(LEFT[,3],RESULT.3);
  GIVEBACK.3[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.3[,2],GIVEBACK.3[,1]);
  GIVEBACK.3=cbind(GIVEBACK.3,LEFT.ORIG[,3],LEFT[,6]);
  GIVEBACK.3[,4]<-(GIVEBACK.3[,3]-GIVEBACK.3[,1]);
  GIVEBACK.4=cbind(LEFT[,4],RESULT.4);
  GIVEBACK.4[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.4[,2],GIVEBACK.4[,1]);
  GIVEBACK.4=cbind(GIVEBACK.4,LEFT.ORIG[,4],LEFT[,6]);
  GIVEBACK.4[,4]<-(GIVEBACK.4[,3]-GIVEBACK.4[,1]);
  GIVEBACK.5=cbind(LEFT[,5],RESULT.5);
  GIVEBACK.5[,1]<-ifelse(overall.z[,ncol(overall.z)]==1,GIVEBACK.5[,2],GIVEBACK.5[,1]);
  GIVEBACK.5=cbind(GIVEBACK.5,LEFT.ORIG[,5],LEFT[,6]);
  GIVEBACK.5[,4]<-(GIVEBACK.5[,3]-GIVEBACK.5[,1]);
  ###
  ### and give it back!
  RNFL.GCL.POSITION.LIGHT[1:split,x]=(GIVEBACK.1[,4]+GIVEBACK.2[,4]+GIVEBACK.3[,4]+GIVEBACK.4[,4]+GIVEBACK.5[,4])/5}





####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $  DONE WITH THAT PART!
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $
####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $ $ $


x=1
plot(VITREOUS.RETINA.POSITION.DARK[,x],ylim=c(0,450),typ="l")
matlines(VITREOUS.RETINA.POSITION.LIGHT[,x],lwd=2)
matlines(RNFL.GCL.POSITION.DARK[,x],col="red")
matlines(RNFL.GCL.POSITION.LIGHT[,x],col="red",lwd=2)
matlines(INL.IPL.POSITION.DARK[,x],col="blue")
matlines(INL.IPL.POSITION.LIGHT[,x],col="blue",lwd=2)
matlines(ONL.OPL.POSITION.DARK[,x],col="red")
matlines(ONL.OPL.POSITION.LIGHT[,x],col="red",lwd=2)
matlines(OLM.POSITION.DARK[,x],col="blue")
matlines(OLM.POSITION.LIGHT[,x],col="blue",lwd=2)
matlines(RPE.POSITION.DARK[,x],col="red")
matlines(RPE.POSITION.LIGHT[,x],col="red",lwd=2)


#########################################################
#########################################################
#########################################################
#########  In the next step,
#########  we use a high-degree-of-freedom spline (df=11) to smooth out any jitter in the localization.
#########  we'll also go back and make something like this for the fovea.

##
## we'll also take tyhat opportunity to extend the lines to 2750 (there was 20 microns that got ignored before because of moving window size)
## as we're doing that, we'll extract retinal thicknesses.
##
## in fovea, we only care about OLM-RPE
## everywhere else, we want the mean across 500 to 2750 for OLM-RPE, RNFL thickness, and total retinal thickness.
##


## work through the 500 to 2750 interval first...
## "R." for revised
R.VITREOUS.RETINA.POSITION.DARK=VITREOUS.RETINA.POSITION.DARK
R.RNFL.GCL.POSITION.DARK=RNFL.GCL.POSITION.DARK
R.INL.IPL.POSITION.DARK=INL.IPL.POSITION.DARK
R.ONL.OPL.POSITION.DARK=ONL.OPL.POSITION.DARK
R.OLM.POSITION.DARK=OLM.POSITION.DARK
R.RPE.POSITION.DARK=RPE.POSITION.DARK
R.VITREOUS.RETINA.POSITION.LIGHT=VITREOUS.RETINA.POSITION.LIGHT
R.RNFL.GCL.POSITION.LIGHT=RNFL.GCL.POSITION.LIGHT
R.INL.IPL.POSITION.LIGHT=INL.IPL.POSITION.LIGHT
R.ONL.OPL.POSITION.LIGHT=ONL.OPL.POSITION.LIGHT
R.OLM.POSITION.LIGHT=OLM.POSITION.LIGHT
R.RPE.POSITION.LIGHT=RPE.POSITION.LIGHT

for(z in 1:ncol(R.VITREOUS.RETINA.POSITION.DARK))
 {REVISE=cbind(seq(499,2750,1),R.VITREOUS.RETINA.POSITION.DARK[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.VITREOUS.RETINA.POSITION.DARK[,z]=S.spline[,2];
  #
  REVISE=cbind(seq(499,2750,1),R.RNFL.GCL.POSITION.DARK[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.RNFL.GCL.POSITION.DARK[,z]=S.spline[,2];
  #
  #
  REVISE=cbind(seq(499,2750,1),R.INL.IPL.POSITION.DARK[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.INL.IPL.POSITION.DARK[,z]=S.spline[,2];
  #
  #
  REVISE=cbind(seq(499,2750,1),R.ONL.OPL.POSITION.DARK[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.ONL.OPL.POSITION.DARK[,z]=S.spline[,2];
  #
  #
  REVISE=cbind(seq(499,2750,1),R.OLM.POSITION.DARK[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.OLM.POSITION.DARK[,z]=S.spline[,2];
  #
  #
  REVISE=R.RPE.POSITION.DARK[,z];
  FILL=REVISE[which(!(is.na(REVISE)))];
  REVISE[which(is.na(REVISE))]=FILL;
  R.RPE.POSITION.DARK[,z]=REVISE}

for(z in 1:ncol(R.VITREOUS.RETINA.POSITION.LIGHT))
 {REVISE=cbind(seq(499,2750,1),R.VITREOUS.RETINA.POSITION.LIGHT[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.VITREOUS.RETINA.POSITION.LIGHT[,z]=S.spline[,2];
  #
  REVISE=cbind(seq(499,2750,1),R.RNFL.GCL.POSITION.LIGHT[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.RNFL.GCL.POSITION.LIGHT[,z]=S.spline[,2];
  #
  #
  REVISE=cbind(seq(499,2750,1),R.INL.IPL.POSITION.LIGHT[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.INL.IPL.POSITION.LIGHT[,z]=S.spline[,2];
  #
  #
  REVISE=cbind(seq(499,2750,1),R.ONL.OPL.POSITION.LIGHT[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.ONL.OPL.POSITION.LIGHT[,z]=S.spline[,2];
  #
  #
  REVISE=cbind(seq(499,2750,1),R.OLM.POSITION.LIGHT[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=11);
  S.spline=cbind(as.numeric(predict(S.sp,seq(499,2750,1))$x),
                   as.numeric(predict(S.sp,seq(499,2750,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.OLM.POSITION.LIGHT[,z]=S.spline[,2];
  #
  #
  REVISE=R.RPE.POSITION.LIGHT[,z];
  FILL=REVISE[which(!(is.na(REVISE)))];
  REVISE[which(is.na(REVISE))]=FILL;
  R.RPE.POSITION.LIGHT[,z]=REVISE}



##
## and, pad these to match FLATTENED.DARK.RETINA.RRC, and FLATTENED.LIGHT.RETINA.RRC,
PAD=matrix(,599,ncol(R.VITREOUS.RETINA.POSITION.DARK))
R.VITREOUS.RETINA.POSITION.DARK=rbind(PAD,R.VITREOUS.RETINA.POSITION.DARK)
R.RNFL.GCL.POSITION.DARK=rbind(PAD,R.RNFL.GCL.POSITION.DARK)
R.INL.IPL.POSITION.DARK=rbind(PAD,R.INL.IPL.POSITION.DARK)
R.ONL.OPL.POSITION.DARK=rbind(PAD,R.ONL.OPL.POSITION.DARK)
R.OLM.POSITION.DARK=rbind(PAD,R.OLM.POSITION.DARK)
R.RPE.POSITION.DARK=rbind(PAD,R.RPE.POSITION.DARK)
PAD=matrix(,600,ncol(R.VITREOUS.RETINA.POSITION.LIGHT))
R.VITREOUS.RETINA.POSITION.LIGHT=rbind(PAD,R.VITREOUS.RETINA.POSITION.LIGHT)
R.RNFL.GCL.POSITION.LIGHT=rbind(PAD,R.RNFL.GCL.POSITION.LIGHT)
R.INL.IPL.POSITION.LIGHT=rbind(PAD,R.INL.IPL.POSITION.LIGHT)
R.ONL.OPL.POSITION.LIGHT=rbind(PAD,R.ONL.OPL.POSITION.LIGHT)
R.OLM.POSITION.LIGHT=rbind(PAD,R.OLM.POSITION.LIGHT)
R.RPE.POSITION.LIGHT=rbind(PAD,R.RPE.POSITION.LIGHT)






##
## calculate some of the "single value" outputs we care about:
MAIN.DARK.OUTPUTS=cbind(IMAGE.INDEX.DARK,IMAGE.INDEX.DARK,IMAGE.INDEX.DARK,IMAGE.INDEX.DARK)
colnames(MAIN.DARK.OUTPUTS)<-c("IMAGE.INDEX","whole.retinal.thick","RPE.to.OLM.distance","RNFL.thickness")
for(x in 1:nrow(MAIN.DARK.OUTPUTS))
 {MAIN.DARK.OUTPUTS[x,2]=mean(na.rm=T,R.RPE.POSITION.DARK[,x]-R.VITREOUS.RETINA.POSITION.DARK[,x]);
  MAIN.DARK.OUTPUTS[x,3]=mean(na.rm=T,R.RPE.POSITION.DARK[,x]-R.OLM.POSITION.DARK[,x]);
  MAIN.DARK.OUTPUTS[x,4]=mean(na.rm=T,R.RNFL.GCL.POSITION.DARK[,x]-R.VITREOUS.RETINA.POSITION.DARK[,x])}
##
## calculate some of the "single value" outputs we care about:
MAIN.LIGHT.OUTPUTS=cbind(IMAGE.INDEX.LIGHT,IMAGE.INDEX.LIGHT,IMAGE.INDEX.LIGHT,IMAGE.INDEX.LIGHT)
colnames(MAIN.LIGHT.OUTPUTS)<-c("IMAGE.INDEX","whole.retinal.thick","RPE.to.OLM.distance","RNFL.thickness")
for(x in 1:nrow(MAIN.LIGHT.OUTPUTS))
 {MAIN.LIGHT.OUTPUTS[x,2]=mean(na.rm=T,R.RPE.POSITION.LIGHT[,x]-R.VITREOUS.RETINA.POSITION.LIGHT[,x]);
  MAIN.LIGHT.OUTPUTS[x,3]=mean(na.rm=T,R.RPE.POSITION.LIGHT[,x]-R.OLM.POSITION.LIGHT[,x]);
  MAIN.LIGHT.OUTPUTS[x,4]=mean(na.rm=T,R.RNFL.GCL.POSITION.LIGHT[,x]-R.VITREOUS.RETINA.POSITION.LIGHT[,x])}

##
##
## now, work through and make the flattened images... we'll work on the 500 to 2750 segment first, and later zero-fill and make space for the little fovea part.
##
## .N for normalized

##
## in terms of thickness, we'll do steps of 1.25%, so 80 pixels covers through 0 to 100%, which is typically ~320 microns, slightly oversampling the outer half of the retina
## (which has more internal structure than some of the othert spans), so that we're sampling around the native resolution.
## and then show 16 microns into the vitreous spread over five pixels (well, it'll look like four), and 24 microns out towards the choroid, spread over five pixels 
##  SINCE THE VITREOUS IS BORING, we'll define each of these markers as the back end of the layer of interesy... so 
##  if we have values for... -10,0,10,20,...80,90,100,110,... those values represent "from -20 to -10","from -10 to -0",etc. 
## so, we'll end up with a 90 pixel-thick strip of retina
dbg("main-normalization", "Building spatially normalized main retinal strips")
FLATTENED.DARK.RETINA.RRC.N=FLATTENED.DARK.RETINA.RRC[,1:90,]
FLATTENED.DARK.RETINA.RRC.N[,,]<-0

depthstrip=seq(1,461,1)
WHICH.INDEX=function(x) which(abs(depthstrip-x)==min(na.rm=T,abs(depthstrip-x)))[1]
position.of.RPE.light=as.factor(seq(1,nrow(R.RPE.POSITION.LIGHT),1))
position.of.RPE.dark=as.factor(seq(1,nrow(R.RPE.POSITION.DARK),1))

END=dim(FLATTENED.DARK.RETINA.RRC)[1]
for(z in 1:dim(FLATTENED.DARK.RETINA.RRC)[3])
  {## assign 5 points from -24 to 0
    ## assign 18 points to the RPE-OLM span
    ## assign 16 points to the OLM-ONL/OPL span
    ## assign 16 points to the ONL/OPL - INL/IPL span
    ## assign 22 points to the INL/IPL - RNFL/GCL span 
    ## assign 8 points to the RNFL/GCL - retina/vitreous span
   ## assign 5 points from retina/vitreous to 16microns into the vitreous.
   A=as.vector(tapply(R.RPE.POSITION.DARK[601:END,z]+24,position.of.RPE.dark[601:END],WHICH.INDEX));
   B=as.vector(tapply(R.RPE.POSITION.DARK[601:END,z]+20,position.of.RPE.dark[601:END],WHICH.INDEX));
   C=as.vector(tapply(R.RPE.POSITION.DARK[601:END,z]+16,position.of.RPE.dark[601:END],WHICH.INDEX));
   D=as.vector(tapply(R.RPE.POSITION.DARK[601:END,z]+12,position.of.RPE.dark[601:END],WHICH.INDEX));
   E=as.vector(tapply(R.RPE.POSITION.DARK[601:END,z]+8,position.of.RPE.dark[601:END],WHICH.INDEX));
   F=as.vector(tapply(R.RPE.POSITION.DARK[601:END,z]+4,position.of.RPE.dark[601:END],WHICH.INDEX));
   G=as.vector(tapply(R.VITREOUS.RETINA.POSITION.DARK[601:END,z],position.of.RPE.dark[601:END],WHICH.INDEX));
   H=as.vector(tapply(R.VITREOUS.RETINA.POSITION.DARK[601:END,z]-4,position.of.RPE.dark[601:END],WHICH.INDEX));
   I=as.vector(tapply(R.VITREOUS.RETINA.POSITION.DARK[601:END,z]-8,position.of.RPE.dark[601:END],WHICH.INDEX));
   J=as.vector(tapply(R.VITREOUS.RETINA.POSITION.DARK[601:END,z]-12,position.of.RPE.dark[601:END],WHICH.INDEX));
   HARVEST=FLATTENED.DARK.RETINA.RRC[,,z];
   for(x in 601:nrow(HARVEST))
    {FLATTENED.DARK.RETINA.RRC.N[(x),1,z]=HARVEST[x,A[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),2,z]=HARVEST[x,B[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),3,z]=HARVEST[x,C[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),4,z]=HARVEST[x,D[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),5,z]=HARVEST[x,E[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),6,z]=HARVEST[x,F[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),87,z]=HARVEST[x,G[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),88,z]=HARVEST[x,H[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),89,z]=HARVEST[x,I[x]];
     FLATTENED.DARK.RETINA.RRC.N[(x),90,z]=HARVEST[x,J[x]]};
   ##
   ## "position.of.RPE.dark" is just a strip of numbers; OK to use as factor here...
   STARTPOINT=R.RPE.POSITION.DARK[601:END,z];
   ENDPOINT=R.OLM.POSITION.DARK[601:END,z];
   list.index=as.factor(seq(1,18,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/18)[1:18];
      FLATTENED.DARK.RETINA.RRC.N[(x+600),7:24,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.OLM.POSITION.DARK[601:END,z];
   ENDPOINT=R.ONL.OPL.POSITION.DARK[601:END,z];
   list.index=as.factor(seq(1,16,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/16)[1:16];
      FLATTENED.DARK.RETINA.RRC.N[(x+600),25:40,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.ONL.OPL.POSITION.DARK[601:END,z];
   ENDPOINT=R.INL.IPL.POSITION.DARK[601:END,z];
   list.index=as.factor(seq(1,16,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/16)[1:16];
      FLATTENED.DARK.RETINA.RRC.N[(x+600),41:56,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.INL.IPL.POSITION.DARK[601:END,z];
   ENDPOINT=R.RNFL.GCL.POSITION.DARK[601:END,z];
   list.index=as.factor(seq(1,22,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/22)[1:22];
      FLATTENED.DARK.RETINA.RRC.N[(x+600),57:78,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.RNFL.GCL.POSITION.DARK[601:END,z];
   ENDPOINT=R.VITREOUS.RETINA.POSITION.DARK[601:END,z];
   list.index=as.factor(seq(1,8,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/8)[1:8];
      FLATTENED.DARK.RETINA.RRC.N[(x+600),79:86,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]}}

FLATTENED.DARK.RETINA.RRC.N[which(is.na(FLATTENED.DARK.RETINA.RRC.N))]=0

## flip to put into same orientation as FLATTENED.DARK.RETINA.RRC
FLATTENED.DARK.RETINA.RRC.N=FLATTENED.DARK.RETINA.RRC.N[,90:1,]

## and an appropriate sequence would be 
#seq(-3.75,107.5,1.25)
## where e.g., "25" means "the value that fills 23.5% to 25% depth in the retina"


## now we can make a profile for each slice!

FLATTENED.DARK.RETINA.RRC.N.profiles=cbind(seq(-3.75,107.5,1.25),seq(-3.75,107.5,1.25))
for(z in 2:dim(FLATTENED.DARK.RETINA.RRC.N)[3]) FLATTENED.DARK.RETINA.RRC.N.profiles=cbind(FLATTENED.DARK.RETINA.RRC.N.profiles,seq(-3.75,107.5,1.25))
colnames(FLATTENED.DARK.RETINA.RRC.N.profiles)<-c("perc.depth",paste("image",IMAGE.INDEX.DARK,sep=""))

for(z in 1:dim(FLATTENED.DARK.RETINA.RRC.N)[3]) 
  {for(y in 1:dim(FLATTENED.DARK.RETINA.RRC.N)[2])
    {FLATTENED.DARK.RETINA.RRC.N.profiles[y,(z+1)]=mean(na.rm=T,FLATTENED.DARK.RETINA.RRC.N[,y,z])}}

z=1
plot(FLATTENED.DARK.RETINA.RRC.N.profiles[,1],FLATTENED.DARK.RETINA.RRC.N.profiles[,(z+1)],typ="l")
abline(v=c(0,10,37.5,57.5,77.5,100))


### now, repeat for light
FLATTENED.LIGHT.RETINA.RRC.N=FLATTENED.LIGHT.RETINA.RRC[,1:90,]
FLATTENED.LIGHT.RETINA.RRC.N[,,]<-0

depthstrip=seq(1,461,1)
WHICH.INDEX=function(x) which(abs(depthstrip-x)==min(na.rm=T,abs(depthstrip-x)))[1]
position.of.RPE.light=as.factor(seq(1,nrow(R.RPE.POSITION.LIGHT),1))
position.of.RPE.light=as.factor(seq(1,nrow(R.RPE.POSITION.LIGHT),1))

END=dim(FLATTENED.LIGHT.RETINA.RRC)[1]
for(z in 1:dim(FLATTENED.LIGHT.RETINA.RRC)[3])
  {## assign 5 points from -24 to 0
    ## assign 18 points to the RPE-OLM span
    ## assign 16 points to the OLM-ONL/OPL span
    ## assign 16 points to the ONL/OPL - INL/IPL span
    ## assign 22 points to the INL/IPL - RNFL/GCL span 
    ## assign 8 points to the RNFL/GCL - retina/vitreous span
   ## assign 5 points from retina/vitreous to 16microns into the vitreous.
   A=as.vector(tapply(R.RPE.POSITION.LIGHT[601:END,z]+24,position.of.RPE.light[601:END],WHICH.INDEX));
   B=as.vector(tapply(R.RPE.POSITION.LIGHT[601:END,z]+20,position.of.RPE.light[601:END],WHICH.INDEX));
   C=as.vector(tapply(R.RPE.POSITION.LIGHT[601:END,z]+16,position.of.RPE.light[601:END],WHICH.INDEX));
   D=as.vector(tapply(R.RPE.POSITION.LIGHT[601:END,z]+12,position.of.RPE.light[601:END],WHICH.INDEX));
   E=as.vector(tapply(R.RPE.POSITION.LIGHT[601:END,z]+8,position.of.RPE.light[601:END],WHICH.INDEX));
   F=as.vector(tapply(R.RPE.POSITION.LIGHT[601:END,z]+4,position.of.RPE.light[601:END],WHICH.INDEX));
   G=as.vector(tapply(R.VITREOUS.RETINA.POSITION.LIGHT[601:END,z],position.of.RPE.light[601:END],WHICH.INDEX));
   H=as.vector(tapply(R.VITREOUS.RETINA.POSITION.LIGHT[601:END,z]-4,position.of.RPE.light[601:END],WHICH.INDEX));
   I=as.vector(tapply(R.VITREOUS.RETINA.POSITION.LIGHT[601:END,z]-8,position.of.RPE.light[601:END],WHICH.INDEX));
   J=as.vector(tapply(R.VITREOUS.RETINA.POSITION.LIGHT[601:END,z]-12,position.of.RPE.light[601:END],WHICH.INDEX));
   HARVEST=FLATTENED.LIGHT.RETINA.RRC[,,z];
   for(x in 601:nrow(HARVEST))
    {FLATTENED.LIGHT.RETINA.RRC.N[(x),1,z]=HARVEST[x,A[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),2,z]=HARVEST[x,B[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),3,z]=HARVEST[x,C[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),4,z]=HARVEST[x,D[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),5,z]=HARVEST[x,E[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),6,z]=HARVEST[x,F[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),87,z]=HARVEST[x,G[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),88,z]=HARVEST[x,H[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),89,z]=HARVEST[x,I[x]];
     FLATTENED.LIGHT.RETINA.RRC.N[(x),90,z]=HARVEST[x,J[x]]};
   ##
   ## "position.of.RPE.light" is just a strip of numbers; OK to use as factor here...
   STARTPOINT=R.RPE.POSITION.LIGHT[601:END,z];
   ENDPOINT=R.OLM.POSITION.LIGHT[601:END,z];
   list.index=as.factor(seq(1,18,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/18)[1:18];
      FLATTENED.LIGHT.RETINA.RRC.N[(x+600),7:24,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.OLM.POSITION.LIGHT[601:END,z];
   ENDPOINT=R.ONL.OPL.POSITION.LIGHT[601:END,z];
   list.index=as.factor(seq(1,16,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/16)[1:16];
      FLATTENED.LIGHT.RETINA.RRC.N[(x+600),25:40,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.ONL.OPL.POSITION.LIGHT[601:END,z];
   ENDPOINT=R.INL.IPL.POSITION.LIGHT[601:END,z];
   list.index=as.factor(seq(1,16,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/16)[1:16];
      FLATTENED.LIGHT.RETINA.RRC.N[(x+600),41:56,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.INL.IPL.POSITION.LIGHT[601:END,z];
   ENDPOINT=R.RNFL.GCL.POSITION.LIGHT[601:END,z];
   list.index=as.factor(seq(1,22,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/22)[1:22];
      FLATTENED.LIGHT.RETINA.RRC.N[(x+600),57:78,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]};
   STARTPOINT=R.RNFL.GCL.POSITION.LIGHT[601:END,z];
   ENDPOINT=R.VITREOUS.RETINA.POSITION.LIGHT[601:END,z];
   list.index=as.factor(seq(1,8,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/8)[1:8];
      FLATTENED.LIGHT.RETINA.RRC.N[(x+600),79:86,z]=HARVEST[(x+600),as.vector(tapply(list,list.index,WHICH.INDEX))]}}

FLATTENED.LIGHT.RETINA.RRC.N[which(is.na(FLATTENED.LIGHT.RETINA.RRC.N))]=0

## flip to put into same orientation as FLATTENED.LIGHT.RETINA.RRC
FLATTENED.LIGHT.RETINA.RRC.N=FLATTENED.LIGHT.RETINA.RRC.N[,90:1,]

## and an appropriate sequence would be 
#seq(-3.75,107.5,1.25)
## where e.g., "25" means "the value that fills 23.5% to 25% depth in the retina"


## now we can make a profile for each slice!

FLATTENED.LIGHT.RETINA.RRC.N.profiles=cbind(seq(-3.75,107.5,1.25),seq(-3.75,107.5,1.25))
for(z in 2:dim(FLATTENED.LIGHT.RETINA.RRC.N)[3]) FLATTENED.LIGHT.RETINA.RRC.N.profiles=cbind(FLATTENED.LIGHT.RETINA.RRC.N.profiles,seq(-3.75,107.5,1.25))
colnames(FLATTENED.LIGHT.RETINA.RRC.N.profiles)<-c("perc.depth",paste("image",IMAGE.INDEX.LIGHT,sep=""))

for(z in 1:dim(FLATTENED.LIGHT.RETINA.RRC.N)[3]) 
  {for(y in 1:dim(FLATTENED.LIGHT.RETINA.RRC.N)[2])
    {FLATTENED.LIGHT.RETINA.RRC.N.profiles[y,(z+1)]=mean(na.rm=T,FLATTENED.LIGHT.RETINA.RRC.N[,y,z])}}

z=1
plot(FLATTENED.LIGHT.RETINA.RRC.N.profiles[,1],FLATTENED.LIGHT.RETINA.RRC.N.profiles[,(z+1)],typ="l")
abline(v=c(0,10,37.5,57.5,77.5,100))




################################################# #so, I have my outputs (MAIN.LIGHT.OUTPUTS,MAIN.DARK.OUTPUTS), and the data on angles (e.g., APPARENT.ANGLES.FOR.LIGHT)
dbg("fovea-normalization", "Building fovea-normalized retinal strips")
################################################# # and my flattened images... so I'm almost done... but I want some info from the fovea 
#################################################
#################################################   ## now, we need to identify each layer
#################################################
#################################################
#################################################

## we'll ultimately look at -50 to 50 microns from the foveal center
## (avascular area)
## the space from the RPE to the OLM will be spatially warped
## we want 30 microns into the foveal ONL; and we'll project this as:
##
## assign 5 points from -24 to 0
## assign 18 points to the RPE-OLM span
## assign 10 points to the OLM-ONL/OPL span


##
## first step is to go back and find those border markers for the fovea...
##
## RPE is col6 of TRUE.BORDERS.DARK
## OLM is col5 of TRUE.BORDERS.DARK

## fovea center is at 101; look a bit wider than -50 nd +50 to help with refining the border...
R.RPE.POSITION.DARK.FOVEA=TRUE.BORDERS.DARK[21:181,6,]
R.OLM.POSITION.DARK.FOVEA=TRUE.BORDERS.DARK[21:181,5,]

## and we'll want to run a smooth spline over that OLM

for(z in 1:ncol(R.OLM.POSITION.DARK.FOVEA))
 {REVISE=cbind(seq(-80,80,1),R.OLM.POSITION.DARK.FOVEA[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=3);
  S.spline=cbind(as.numeric(predict(S.sp,seq(-80,80,1))$x),
                   as.numeric(predict(S.sp,seq(-80,80,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.OLM.POSITION.DARK.FOVEA[,z]=S.spline[,2]}

## and prune to just be -50 to 50
R.OLM.POSITION.DARK.FOVEA2=R.OLM.POSITION.DARK.FOVEA[31:131,]
R.RPE.POSITION.DARK.FOVEA2=R.RPE.POSITION.DARK.FOVEA[31:131,]
## but pad for later use
R.OLM.POSITION.DARK.FOVEA=R.OLM.POSITION.DARK
R.OLM.POSITION.DARK.FOVEA[51:151,]=R.OLM.POSITION.DARK.FOVEA2
R.RPE.POSITION.DARK.FOVEA=R.RPE.POSITION.DARK
R.RPE.POSITION.DARK.FOVEA[51:151,]=R.RPE.POSITION.DARK.FOVEA2


## and save the OLM-RPE distance
MAIN.DARK.OUTPUTS.fovea=MAIN.DARK.OUTPUTS[,c(1,3)]
MAIN.DARK.OUTPUTS.fovea[,2]<-NA
for(x in 1:nrow(MAIN.DARK.OUTPUTS.fovea)) MAIN.DARK.OUTPUTS.fovea[x,2]=mean(na.rm=T,R.RPE.POSITION.DARK.FOVEA[,x]-R.OLM.POSITION.DARK.FOVEA[,x])

##<new for 2022-DEC-29, correct the fovea thickness>
## this line:
## for(x in 1:nrow(MAIN.DARK.OUTPUTS.fovea)) MAIN.DARK.OUTPUTS.fovea[x,2]=mean(na.rm=T,R.RPE.POSITION.DARK.FOVEA[,x]-R.OLM.POSITION.DARK.FOVEA[,x])
## replaced with this:
for(x in 1:nrow(MAIN.DARK.OUTPUTS.fovea)) MAIN.DARK.OUTPUTS.fovea[x,2]=mean(na.rm=T,R.RPE.POSITION.DARK.FOVEA[51:151,x]-R.OLM.POSITION.DARK.FOVEA[51:151,x])


## and transfer the data into a flattened output image...
FLATTENED.DARK.RETINA.RRC.N.fovea=FLATTENED.DARK.RETINA.RRC[,1:90,]
FLATTENED.DARK.RETINA.RRC.N.fovea[,,]<-0

depthstrip=seq(1,461,1)
WHICH.INDEX=function(x) which(abs(depthstrip-x)==min(na.rm=T,abs(depthstrip-x)))[1]
position.of.RPE.light=as.factor(seq(1,nrow(R.RPE.POSITION.LIGHT),1))
position.of.RPE.dark=as.factor(seq(1,nrow(R.RPE.POSITION.DARK),1))

END=151
for(z in 1:dim(FLATTENED.DARK.RETINA.RRC)[3])
  {## assign 5 points from -24 to 0
    ## assign 18 points to the RPE-OLM span
    ## assign 9 points from the OLM to 36microns interior
   A=as.vector(tapply(R.RPE.POSITION.DARK.FOVEA[51:END,z]+24,position.of.RPE.dark[51:END],WHICH.INDEX));
   B=as.vector(tapply(R.RPE.POSITION.DARK.FOVEA[51:END,z]+20,position.of.RPE.dark[51:END],WHICH.INDEX));
   C=as.vector(tapply(R.RPE.POSITION.DARK.FOVEA[51:END,z]+16,position.of.RPE.dark[51:END],WHICH.INDEX));
   D=as.vector(tapply(R.RPE.POSITION.DARK.FOVEA[51:END,z]+12,position.of.RPE.dark[51:END],WHICH.INDEX));
   E=as.vector(tapply(R.RPE.POSITION.DARK.FOVEA[51:END,z]+8,position.of.RPE.dark[51:END],WHICH.INDEX));
   F=as.vector(tapply(R.RPE.POSITION.DARK.FOVEA[51:END,z]+4,position.of.RPE.dark[51:END],WHICH.INDEX));
   G=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z],position.of.RPE.dark[51:END],WHICH.INDEX));
   H=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-4,position.of.RPE.dark[51:END],WHICH.INDEX));
   I=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-8,position.of.RPE.dark[51:END],WHICH.INDEX));
   J=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-12,position.of.RPE.dark[51:END],WHICH.INDEX));
   K=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-16,position.of.RPE.dark[51:END],WHICH.INDEX));
   L=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-20,position.of.RPE.dark[51:END],WHICH.INDEX));
   M=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-24,position.of.RPE.dark[51:END],WHICH.INDEX));
   N=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-28,position.of.RPE.dark[51:END],WHICH.INDEX));
   O=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-32,position.of.RPE.dark[51:END],WHICH.INDEX));
   P=as.vector(tapply(R.OLM.POSITION.DARK.FOVEA[51:END,z]-36,position.of.RPE.dark[51:END],WHICH.INDEX));
   HARVEST=FLATTENED.DARK.RETINA.RRC[,,z];
   for(x in 51:END)
    {FLATTENED.DARK.RETINA.RRC.N.fovea[(x),1,z]=HARVEST[x,A[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),2,z]=HARVEST[x,B[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),3,z]=HARVEST[x,C[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),4,z]=HARVEST[x,D[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),5,z]=HARVEST[x,E[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),6,z]=HARVEST[x,F[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),25,z]=HARVEST[x,G[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),26,z]=HARVEST[x,H[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),27,z]=HARVEST[x,I[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),28,z]=HARVEST[x,J[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),29,z]=HARVEST[x,K[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),30,z]=HARVEST[x,L[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),31,z]=HARVEST[x,M[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),32,z]=HARVEST[x,N[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),33,z]=HARVEST[x,O[x]];
     FLATTENED.DARK.RETINA.RRC.N.fovea[(x),34,z]=HARVEST[x,P[x]]};
   ##
   ## "position.of.RPE.dark" is just a strip of numbers; OK to use as factor here...
   STARTPOINT=R.RPE.POSITION.DARK.FOVEA[51:END,z];
   ENDPOINT=R.OLM.POSITION.DARK.FOVEA[51:END,z];
   list.index=as.factor(seq(1,18,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/18)[1:18];
      FLATTENED.DARK.RETINA.RRC.N.fovea[(x+50),7:24,z]=HARVEST[(x+50),as.vector(tapply(list,list.index,WHICH.INDEX))]}}

## and flip to get into same orientation as non-spatially-normalized data
FLATTENED.DARK.RETINA.RRC.N.fovea=FLATTENED.DARK.RETINA.RRC.N.fovea[,90:1,]


FLATTENED.DARK.RETINA.RRC.N.fovea.profiles=cbind(seq(-3.75,107.5,1.25),seq(-3.75,107.5,1.25))
for(z in 2:dim(FLATTENED.DARK.RETINA.RRC.N)[3]) FLATTENED.DARK.RETINA.RRC.N.fovea.profiles=cbind(FLATTENED.DARK.RETINA.RRC.N.fovea.profiles,seq(-3.75,107.5,1.25))
#colnames(FLATTENED.DARK.RETINA.RRC.N.fovea.profiles)<-c("perc.depth",paste("image",IMAGE.INDEX.LIGHT,sep=""))

for(z in 1:dim(FLATTENED.DARK.RETINA.RRC.N.fovea)[3]) 
  {for(y in 1:dim(FLATTENED.DARK.RETINA.RRC.N.fovea)[2])
    {FLATTENED.DARK.RETINA.RRC.N.fovea.profiles[y,(z+1)]=mean(na.rm=T,FLATTENED.DARK.RETINA.RRC.N.fovea[51:151,y,z])}}

z=1
plot(FLATTENED.DARK.RETINA.RRC.N.fovea.profiles[,1],FLATTENED.DARK.RETINA.RRC.N.fovea.profiles[,(z+1)],typ="l")
abline(v=c(0,10,37.5,57.5,77.5,100))

## and crop off that stuff we don't need...
FLATTENED.DARK.RETINA.RRC.N.fovea.profiles=FLATTENED.DARK.RETINA.RRC.N.fovea.profiles[57:90,]

## and combine the normalized flattened data...

FLATTENED.DARK.RETINA.RRC.N[50:152,,]=FLATTENED.DARK.RETINA.RRC.N.fovea[50:152,,]


## and save!
## the pixel dimensions of this are 1 micron in the x, and 1.25% of the retinal thickness in the y
EXPORT=FLATTENED.DARK.RETINA.RRC.N
EXPORT[which(is.na(EXPORT))]=0
f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste("_flat-normed_",TO.PROCESS.DARK,sep=""),size="float",path.out=OUTDIR)


######################################################################################################################################################################################################################################
## and repeat all of that for light!

##
## first step is to go back and find those border markers for the fovea...
##
## RPE is col6 of TRUE.BORDERS.LIGHT
## OLM is col5 of TRUE.BORDERS.LIGHT

## fovea center is at 101; look a bit wider than -50 nd +50 to help with refining the border...
R.RPE.POSITION.LIGHT.FOVEA=TRUE.BORDERS.LIGHT[21:181,6,]
R.OLM.POSITION.LIGHT.FOVEA=TRUE.BORDERS.LIGHT[21:181,5,]

## and we'll want to run a smooth spline over that OLM

for(z in 1:ncol(R.OLM.POSITION.LIGHT.FOVEA))
 {REVISE=cbind(seq(-80,80,1),R.OLM.POSITION.LIGHT.FOVEA[,z]);
  REVISE=REVISE[which(!(is.na(REVISE[,2]))),];
  S.sp=smooth.spline(REVISE[,1],REVISE[,2],df=3);
  S.spline=cbind(as.numeric(predict(S.sp,seq(-80,80,1))$x),
                   as.numeric(predict(S.sp,seq(-80,80,1))$y));
  plot(REVISE[,1],REVISE[,2]);
  matlines(S.spline[,1],S.spline[,2]);
  R.OLM.POSITION.LIGHT.FOVEA[,z]=S.spline[,2]}

## and prune to just be -50 to 50
R.OLM.POSITION.LIGHT.FOVEA2=R.OLM.POSITION.LIGHT.FOVEA[31:131,]
R.RPE.POSITION.LIGHT.FOVEA2=R.RPE.POSITION.LIGHT.FOVEA[31:131,]
## but pad for later use
R.OLM.POSITION.LIGHT.FOVEA=R.OLM.POSITION.LIGHT
R.OLM.POSITION.LIGHT.FOVEA[51:151,]=R.OLM.POSITION.LIGHT.FOVEA2
R.RPE.POSITION.LIGHT.FOVEA=R.RPE.POSITION.LIGHT
R.RPE.POSITION.LIGHT.FOVEA[51:151,]=R.RPE.POSITION.LIGHT.FOVEA2



## and save the OLM-RPE distance
MAIN.LIGHT.OUTPUTS.fovea=MAIN.LIGHT.OUTPUTS[,c(1,3)]
MAIN.LIGHT.OUTPUTS.fovea[,2]<-NA
##<new for 2022-DEC-29, correct the fovea thickness>
## this line:
## for(x in 1:nrow(MAIN.LIGHT.OUTPUTS.fovea)) MAIN.LIGHT.OUTPUTS.fovea[x,2]=mean(na.rm=T,R.RPE.POSITION.LIGHT.FOVEA[,x]-R.OLM.POSITION.LIGHT.FOVEA[,x])
## replaced with this:
for(x in 1:nrow(MAIN.LIGHT.OUTPUTS.fovea)) MAIN.LIGHT.OUTPUTS.fovea[x,2]=mean(na.rm=T,R.RPE.POSITION.LIGHT.FOVEA[51:151,x]-R.OLM.POSITION.LIGHT.FOVEA[51:151,x])



## and transfer the data into a flattened output image...
FLATTENED.LIGHT.RETINA.RRC.N.fovea=FLATTENED.LIGHT.RETINA.RRC[,1:90,]
FLATTENED.LIGHT.RETINA.RRC.N.fovea[,,]<-0

depthstrip=seq(1,461,1)
WHICH.INDEX=function(x) which(abs(depthstrip-x)==min(na.rm=T,abs(depthstrip-x)))[1]
position.of.RPE.light=as.factor(seq(1,nrow(R.RPE.POSITION.LIGHT),1))
position.of.RPE.light=as.factor(seq(1,nrow(R.RPE.POSITION.LIGHT),1))

END=151
for(z in 1:dim(FLATTENED.LIGHT.RETINA.RRC)[3])
  {## assign 5 points from -24 to 0
    ## assign 18 points to the RPE-OLM span
    ## assign 9 points from the OLM to 36microns interior
   A=as.vector(tapply(R.RPE.POSITION.LIGHT.FOVEA[51:END,z]+24,position.of.RPE.light[51:END],WHICH.INDEX));
   B=as.vector(tapply(R.RPE.POSITION.LIGHT.FOVEA[51:END,z]+20,position.of.RPE.light[51:END],WHICH.INDEX));
   C=as.vector(tapply(R.RPE.POSITION.LIGHT.FOVEA[51:END,z]+16,position.of.RPE.light[51:END],WHICH.INDEX));
   D=as.vector(tapply(R.RPE.POSITION.LIGHT.FOVEA[51:END,z]+12,position.of.RPE.light[51:END],WHICH.INDEX));
   E=as.vector(tapply(R.RPE.POSITION.LIGHT.FOVEA[51:END,z]+8,position.of.RPE.light[51:END],WHICH.INDEX));
   F=as.vector(tapply(R.RPE.POSITION.LIGHT.FOVEA[51:END,z]+4,position.of.RPE.light[51:END],WHICH.INDEX));
   G=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z],position.of.RPE.light[51:END],WHICH.INDEX));
   H=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-4,position.of.RPE.light[51:END],WHICH.INDEX));
   I=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-8,position.of.RPE.light[51:END],WHICH.INDEX));
   J=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-12,position.of.RPE.light[51:END],WHICH.INDEX));
   K=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-16,position.of.RPE.light[51:END],WHICH.INDEX));
   L=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-20,position.of.RPE.light[51:END],WHICH.INDEX));
   M=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-24,position.of.RPE.light[51:END],WHICH.INDEX));
   N=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-28,position.of.RPE.light[51:END],WHICH.INDEX));
   O=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-32,position.of.RPE.light[51:END],WHICH.INDEX));
   P=as.vector(tapply(R.OLM.POSITION.LIGHT.FOVEA[51:END,z]-36,position.of.RPE.light[51:END],WHICH.INDEX));
   HARVEST=FLATTENED.LIGHT.RETINA.RRC[,,z];
   for(x in 51:END)
    {FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),1,z]=HARVEST[x,A[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),2,z]=HARVEST[x,B[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),3,z]=HARVEST[x,C[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),4,z]=HARVEST[x,D[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),5,z]=HARVEST[x,E[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),6,z]=HARVEST[x,F[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),25,z]=HARVEST[x,G[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),26,z]=HARVEST[x,H[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),27,z]=HARVEST[x,I[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),28,z]=HARVEST[x,J[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),29,z]=HARVEST[x,K[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),30,z]=HARVEST[x,L[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),31,z]=HARVEST[x,M[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),32,z]=HARVEST[x,N[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),33,z]=HARVEST[x,O[x]];
     FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x),34,z]=HARVEST[x,P[x]]};
   ##
   ## "position.of.RPE.light" is just a strip of numbers; OK to use as factor here...
   STARTPOINT=R.RPE.POSITION.LIGHT.FOVEA[51:END,z];
   ENDPOINT=R.OLM.POSITION.LIGHT.FOVEA[51:END,z];
   list.index=as.factor(seq(1,18,1));
   for(x in 1:length(STARTPOINT)) 
     {list=seq(STARTPOINT[x],ENDPOINT[x],(ENDPOINT[x]-STARTPOINT[x])/18)[1:18];
      FLATTENED.LIGHT.RETINA.RRC.N.fovea[(x+50),7:24,z]=HARVEST[(x+50),as.vector(tapply(list,list.index,WHICH.INDEX))]}}

## and flip to get into same orientation as non-spatially-normalized data
FLATTENED.LIGHT.RETINA.RRC.N.fovea=FLATTENED.LIGHT.RETINA.RRC.N.fovea[,90:1,]


FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles=cbind(seq(-3.75,107.5,1.25),seq(-3.75,107.5,1.25))
for(z in 2:dim(FLATTENED.LIGHT.RETINA.RRC.N)[3]) FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles=cbind(FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles,seq(-3.75,107.5,1.25))
#colnames(FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles)<-c("perc.depth",paste("image",IMAGE.INDEX.LIGHT,sep=""))

for(z in 1:dim(FLATTENED.LIGHT.RETINA.RRC.N.fovea)[3]) 
  {for(y in 1:dim(FLATTENED.LIGHT.RETINA.RRC.N.fovea)[2])
    {FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles[y,(z+1)]=mean(na.rm=T,FLATTENED.LIGHT.RETINA.RRC.N.fovea[51:151,y,z])}}

z=1
plot(FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles[,1],FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles[,(z+1)],typ="l")
abline(v=c(0,10,37.5,57.5,77.5,100))

## and crop off that stuff we don't need...
FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles=FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles[57:90,]

## and combine the normalized flattened data...

FLATTENED.LIGHT.RETINA.RRC.N[50:152,,]=FLATTENED.LIGHT.RETINA.RRC.N.fovea[50:152,,]


## and save!
## the pixel dimensions of this are 1 micron in the x, and 1.25% of the retinal thickness in the y
EXPORT=FLATTENED.LIGHT.RETINA.RRC.N
EXPORT[which(is.na(EXPORT))]=0
f.write.analyze(EXPORT[,dim(EXPORT)[2]:1,],paste("_flat-normed_",TO.PROCESS.LIGHT,sep=""),size="float",path.out=OUTDIR)





########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
## we still need to harmonze export of the angle data

#e.Rdata

## clean up a bit
rm(A,B,BLANK,C,c.sp,check,cspline,D,depthstrip,DEVIATION.1,DEVIATION.2,DEVIATION.3)
rm(DEVIATION.4,DEVIATION.5,E,END,end.move,ENDPOINT,EXPORT,F,FILL,FIVE,FLATTENED.DARK.RETINA.RRC.N.fovea)
rm(FLATTENED.LIGHT.RETINA.RRC.N.fovea,FOUR,G,GIVEBACK.1,GIVEBACK.2,GIVEBACK.3,GIVEBACK.4,GIVEBACK.5,H)
rm(HAND.BORDERS,HAND.BORDERS.factor,HARVEST,I,J,K,L,LEFT,LEFT.ORIG,list,list.index)
rm(M,MEAN.x,MODEL.1,MODEL.2,MODEL.3,MODEL.4,MODEL.5,MOVEIN,MOVEINalt,N,NEW.VALUES,O)
rm(ONE,overall.z,P,PAD,peak,position.of.RPE.dark,position.of.RPE.light,profile,R.OLM.POSITION.DARK.FOVEA2,R.OLM.POSITION.LIGHT.FOVEA2)
rm(R.RPE.POSITION.DARK.FOVEA2,R.RPE.POSITION.LIGHT.FOVEA2,RESULT.1,RESULT.2,RESULT.3,RESULT.4,RESULT.5)
rm(REVIEW,REVISE,RNFL.GCL.POSITION.DARK,RNFL.GCL.POSITION.LIGHT,RPE.POSITION.DARK,RPE.POSITION.LIGHT,S.sp,S.spline)
rm(SEGMENT,split)
rm(start.move,STARTPOINT,THREE)
rm(WHICH.INDEX,TRUE.BORDERS.LIGHT,TWO,vertex,TRUE.BORDERS.DARK)   
rm(window.factor,window.width.in.pixels)
rm(x,y,z.DEVIATION.2,z.DEVIATION.3,z.DEVIATION.4,z.DEVIATION.5,z.threshold,z.DEVIATION.1)                              

rm(INL.IPL.POSITION.DARK,INL.IPL.POSITION.LIGHT,OLM.POSITION.DARK,OLM.POSITION.LIGHT,ONL.OPL.POSITION.DARK,ONL.OPL.POSITION.LIGHT,PIXEL.WIDTH)
rm(z)


MAIN.DARK.OUTPUTS=cbind(MAIN.DARK.OUTPUTS[,1],APPARENT.ANGLES.FOR.DARK[,3],MAIN.DARK.OUTPUTS[,2:4])
MAIN.LIGHT.OUTPUTS=cbind(MAIN.LIGHT.OUTPUTS[,1],APPARENT.ANGLES.FOR.LIGHT[,3],MAIN.LIGHT.OUTPUTS[,2:4])
colnames(MAIN.DARK.OUTPUTS)<-c("INDEX","apparent.angle.retinal.strip..degrees","whole.retinal.thick","RPE.to.OLM.distance","RNFL.thickness")
colnames(MAIN.LIGHT.OUTPUTS)<-c("INDEX","apparent.angle.retinal.strip..degrees","whole.retinal.thick","RPE.to.OLM.distance","RNFL.thickness")

MAIN.DARK.OUTPUTS.fovea=cbind(MAIN.DARK.OUTPUTS.fovea[,1],APPARENT.ANGLES.FOR.DARK[,2],MAIN.DARK.OUTPUTS.fovea[,2])
colnames(MAIN.DARK.OUTPUTS.fovea)<-c("INDEX","apparent.angle.fovea..degrees","RPE.to.OLM.distance")
MAIN.LIGHT.OUTPUTS.fovea=cbind(MAIN.LIGHT.OUTPUTS.fovea[,1],APPARENT.ANGLES.FOR.LIGHT[,2],MAIN.LIGHT.OUTPUTS.fovea[,2])
colnames(MAIN.LIGHT.OUTPUTS.fovea)<-c("INDEX","apparent.angle.fovea..degrees","RPE.to.OLM.distance")

dbg("final-export", "Writing tissue-border preview PNGs")
save.tissue.border.plot(
  paste("_tissueBorders__", TO.PROCESS.DARK, ".png", sep=""),
  FLATTENED.DARK.RETINA.RRC,
  R.RPE.POSITION.DARK,
  R.OLM.POSITION.DARK,
  R.ONL.OPL.POSITION.DARK,
  R.INL.IPL.POSITION.DARK,
  R.RNFL.GCL.POSITION.DARK,
  R.VITREOUS.RETINA.POSITION.DARK
)
save.tissue.border.plot(
  paste("_tissueBorders__", TO.PROCESS.LIGHT, ".png", sep=""),
  FLATTENED.LIGHT.RETINA.RRC,
  R.RPE.POSITION.LIGHT,
  R.OLM.POSITION.LIGHT,
  R.ONL.OPL.POSITION.LIGHT,
  R.INL.IPL.POSITION.LIGHT,
  R.RNFL.GCL.POSITION.LIGHT,
  R.VITREOUS.RETINA.POSITION.LIGHT
)

rm(APPARENT.ANGLES.FOR.DARK,APPARENT.ANGLES.FOR.LIGHT)
rm(VITREOUS.RETINA.POSITION.DARK,VITREOUS.RETINA.POSITION.LIGHT)



#######
####### what's left?
##
## desciptions for "dark" variables (light are the same, but for light data):
#DFforSECONDfit = setting for how bendy some of the fits are. Save for records.
#DFonINITIALspline = setting for how bendy some of the fits are. Save for records.
#FLATTENED.DARK.RETINA.RRC = flattened images. length of the retinal strip (from 100 microns left of the fovea center to 2750 microns right)
#                            is encoded by row (one row = 1 micron). The RPE is at column 430. Lower column# is more interior to retina
#                            each slice in this 3D array is another image from the same subject/condition
#                            each column is 1 micron distance
#FLATTENED.DARK.RETINA.RRC.N = spatially normalized flattened images. length of the retinal strip (from 100 microns left of the fovea center to 2750 microns right)
#                            is encoded by row (one row = 1 micron). 
#                            the few values from non-retina (vitreous or choroid) are in 4 micron steps from the border.
#                            otherwise, each column represents 1.25% of the total retinal thickness
#                            The RPE is at column 85. Lower column# is more interior to retina
#                            each slice in this 3D array is another image from the same subject/condition
#                              note: above is for the span from +500 to +2750 microns from the fovea;
#                                    this variable also includes the small segment in fovea, (from -50 to 50 microns) wherein
#                                    RPE is at column 85 (100% of the thick), OLM is at coumn 66 (77.5% of the thick), and the additional
#                                    pixels interior to the OLM are sampled every 4 microns. Mapping that onto demensions of the rest, this
#                                    lands at column 57, (representing 66.25% depth). Basically, we want a fair comparison of fovea and non-fovea
#                            in the main stript of retina, the ONL/OPL border is at column 50 [57.5% depth]
#FLATTENED.DARK.RETINA.RRC.N.fovea.profiles = mean profile of the fovea data
#FLATTENED.DARK.RETINA.RRC.N.profiles = mean profile of the main retinal strip; note that this is in "raw" reflectivities, not log-transformed
#FLATTENED.MARKERS.RRC = linearized but not spatially normalized data showing the mand-made markers
#IMAGE.INDEX.DARK = manually set, lists the ordder, and # of each image in series
#MAIN.DARK.OUTPUTS = image# in first column, second colum is the apparent angle of the RPE in each image (linear estimate for 500 to 2750 microns right of fovea)
#                    ...which is needed per Lujan's 2011 paper "Revealing Henle�s Fiber Layer Using Spectral Domain Optical Coherence Tomography"
#                    the third through fifth coulmns are mean thickness from 500 to 2750 microns right of the fovea.
#MAIN.DARK.OUTPUTS.fovea = image# in first column, second colum is the apparent angle of the RPE in each image 
#                          (linear estimate for -100 to 100 microns from the fovea... a slightly wider sweep than -50 to 50 used just in the service of making sure there's a nice
#                           pool of values for the linear estimate, which regardless is centered on the fovea)
#                           ...this is needed per Lujan's 2011 paper "Revealing Henle�s Fiber Layer Using Spectral Domain Optical Coherence Tomography"
#                           the third coulmn is the mean thickness from -50 to 50 microns right of the fovea.
#R.RPE.POSITION.DARK = position of the RPE (after "revision", hence the "R." prefix; uses smooth splines to minimize jitter/errors in layer localization)
#                      with distance along the RPE in the flattened retina stored by row (one row is 1 micron, span is -100 to 2750 microns from the optic nerve)
#                      and ncol = number of slices
#                      the values (if a whole number) represents a column in FLATTENED.DARK.RETINA.RRC
#                       ...but since each column in FLATTENED.DARK.RETINA.RRC is 1 microns, the values in this matrix are actually
#                        positions (in microns) of each layer. 
#                      RPE is set to position 431 earlier, and cannot deviate
#                      
#    the remaining .POSITION.DARK variables are nearly identical to R.RPE.POSITION.DARK, but for
#    the INL/IPL border, the OLM, the ONL/OPL border, the RNFL/GCL border, and the vitreous/RNFL border.
#    the position data is stored such that R.RPE.POSITION.DARK[,1]-R.VITREOUS.RETINA.POSITION.DARK[,1] will give a strip of retinal thicknesses for the first dark image
#     in microns, spanning from 500 to 2750 microns right of the fovea.
#
#R.INL.IPL.POSITION.DARK = (see above)
#R.OLM.POSITION.DARK = (see above)
#R.ONL.OPL.POSITION.DARK = (see above)
#R.RNFL.GCL.POSITION.DARK = (see above)
#R.VITREOUS.RETINA.POSITION.DARK = (see above)
#
#R.RPE.POSITION.DARK.FOVEA = like the above, but for -50 to 50 from the fovea
#R.OLM.POSITION.DARK.FOVEA = like the above, but for -50 to 50 from the fovea
#
#REFERENCE.DARK=filename of the marked up file; saving just to ensure appropriate/redundant labeling of data within this file                             
#TO.PROCESS.DARK=filename of the RAW data file; saving just to ensure appropriate/redundant labeling of data within this file   

dbg("final-export", "Writing final Step 3 profile tables")
EXPORT=as.data.frame(cbind(MAIN.DARK.OUTPUTS,t(FLATTENED.DARK.RETINA.RRC.N.profiles[,2:ncol(FLATTENED.DARK.RETINA.RRC.N.profiles)])))
EXPORT=round(EXPORT,3)
EXPORT=rbind(EXPORT[1,],EXPORT)
EXPORT[1,]=t(as.data.frame(c(TO.PROCESS.DARK,"angle","whole","RPEtoOLM","RNFL",seq(-3.75,107.5,1.25))))
EXPORT=t(EXPORT)
write(t(EXPORT),ncol=ncol(EXPORT),file=file.path(OUTDIR, paste("_dark_profiles_",TO.PROCESS.DARK,".txt",sep="")))

EXPORT=as.data.frame(cbind(MAIN.LIGHT.OUTPUTS,t(FLATTENED.LIGHT.RETINA.RRC.N.profiles[,2:ncol(FLATTENED.LIGHT.RETINA.RRC.N.profiles)])))
EXPORT=round(EXPORT,3)
EXPORT=rbind(EXPORT[1,],EXPORT)
EXPORT[1,]=t(as.data.frame(c(TO.PROCESS.LIGHT,"angle","whole","RPEtoOLM","RNFL",seq(-3.75,107.5,1.25))))
EXPORT=t(EXPORT)
write(t(EXPORT),ncol=ncol(EXPORT),file=file.path(OUTDIR, paste("_light_profiles_",TO.PROCESS.LIGHT,".txt",sep="")))

EXPORT=MAIN.DARK.OUTPUTS.fovea[,c(1,2,2,3,3)]
EXPORT[,3]<-NA
EXPORT[,5]<-NA
EXPORT.2=FLATTENED.DARK.RETINA.RRC.N.fovea.profiles[,2:ncol(FLATTENED.DARK.RETINA.RRC.N.fovea.profiles)]
EXPORT.2.buff=FLATTENED.DARK.RETINA.RRC.N.profiles[1:56,2:ncol(FLATTENED.DARK.RETINA.RRC.N.profiles)]
EXPORT.2.buff[,]<-NA
EXPORT.2=rbind(EXPORT.2.buff,EXPORT.2)
EXPORT=as.data.frame(cbind(EXPORT,t(EXPORT.2)))
EXPORT=round(EXPORT,3)
EXPORT=rbind(EXPORT[1,],EXPORT)
EXPORT[1,]=t(as.data.frame(c(paste("fovea",TO.PROCESS.DARK,sep=""),"angle","whole","RPEtoOLM","RNFL",seq(-3.75,107.5,1.25))))
EXPORT=t(EXPORT)
write(t(EXPORT),ncol=ncol(EXPORT),file=file.path(OUTDIR, paste("_fovea_dark_profiles_",TO.PROCESS.DARK,".txt",sep="")))

EXPORT=MAIN.LIGHT.OUTPUTS.fovea[,c(1,2,2,3,3)]
EXPORT[,3]<-NA
EXPORT[,5]<-NA
EXPORT.2=FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles[,2:ncol(FLATTENED.LIGHT.RETINA.RRC.N.fovea.profiles)]
EXPORT.2.buff=FLATTENED.LIGHT.RETINA.RRC.N.profiles[1:56,2:ncol(FLATTENED.LIGHT.RETINA.RRC.N.profiles)]
EXPORT.2.buff[,]<-NA
EXPORT.2=rbind(EXPORT.2.buff,EXPORT.2)
EXPORT=as.data.frame(cbind(EXPORT,t(EXPORT.2)))
EXPORT=round(EXPORT,3)
EXPORT=rbind(EXPORT[1,],EXPORT)
EXPORT[1,]=t(as.data.frame(c(paste("fovea",TO.PROCESS.LIGHT,sep=""),"angle","whole","RPEtoOLM","RNFL",seq(-3.75,107.5,1.25))))
EXPORT=t(EXPORT)
write(t(EXPORT),ncol=ncol(EXPORT),file=file.path(OUTDIR, paste("_fovea_light_profiles_",TO.PROCESS.LIGHT,".txt",sep="")))

export.python.final.arrays()

## and cleanup before a final Save
rm(EXPORT.2.buff,EXPORT.2,EXPORT)

save.image(file.path(OUTDIR, paste("_done_",TO.PROCESS.DARK,"__and__",TO.PROCESS.LIGHT,".RData",sep="")))
dbg("done", "R processing complete")













