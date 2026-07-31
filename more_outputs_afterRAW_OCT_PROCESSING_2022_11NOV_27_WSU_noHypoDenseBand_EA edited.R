## use this either immediately after running "RAW_OCT_PROCESSING".R
## (like, in the same RGui window)
## or after opening the file saved by "RAW_OCT_PROCESSING".R (which should have the prefix "_done_"


### LIGHT is the app's acquisition channel. Recreate the legacy DARK slots when
### loading a LIGHT-only workspace so the original output formulas still run.
if(!exists("TO.PROCESS.DARK")) TO.PROCESS.DARK="DARK"
if(!exists("FLATTENED.DARK.RETINA.RRC")) FLATTENED.DARK.RETINA.RRC=FLATTENED.LIGHT.RETINA.RRC
if(!exists("R.RPE.POSITION.DARK")) R.RPE.POSITION.DARK=R.RPE.POSITION.LIGHT
if(!exists("R.OLM.POSITION.DARK")) R.OLM.POSITION.DARK=R.OLM.POSITION.LIGHT
if(!exists("R.ONL.OPL.POSITION.DARK")) R.ONL.OPL.POSITION.DARK=R.ONL.OPL.POSITION.LIGHT
if(!exists("R.INL.IPL.POSITION.DARK")) R.INL.IPL.POSITION.DARK=R.INL.IPL.POSITION.LIGHT
if(!exists("R.RNFL.GCL.POSITION.DARK")) R.RNFL.GCL.POSITION.DARK=R.RNFL.GCL.POSITION.LIGHT
if(!exists("R.VITREOUS.RETINA.POSITION.DARK")) R.VITREOUS.RETINA.POSITION.DARK=R.VITREOUS.RETINA.POSITION.LIGHT

### The first member of each stack is used for the plots.
### Use an explicit PNG device because AIDaS runs this script through
### non-interactive Rscript, where dev.new()/savePlot() selects a PDF device.
save.tissue.border.output <- function(filename, flattened, rpe, olm, onl.opl, inl.ipl, rnfl.gcl, vitreous.retina) {
  x.axis=seq(-100,2750,1)
  y.axis=seq(-30,430,1)
  target=file.path(getwd(),filename)

  png(filename=target,width=1600,height=520)
  on.exit(dev.off())
  image(as.vector(x.axis),as.vector(y.axis),
        as.matrix(flattened[,dim(flattened)[2]:1,1]),
        xlab="Distance from Fovea (microns)",
        ylab="Distance from RPE (microns)",col=gray.colors(254))
  matlines(x.axis,431-rpe[1:2851,1],col="red")
  matlines(x.axis,431-olm[1:2851,1],col="blue")
  matlines(x.axis,431-onl.opl[1:2851,1],col="red")
  matlines(x.axis,431-inl.ipl[1:2851,1],col="blue")
  matlines(x.axis,431-rnfl.gcl[1:2851,1],col="red")
  matlines(x.axis,431-vitreous.retina[1:2851,1],col="blue")
  invisible(target)
}

save.tissue.border.output(
  paste("_tissueBorders__",TO.PROCESS.DARK,".png",sep=""),
  FLATTENED.DARK.RETINA.RRC,
  R.RPE.POSITION.DARK,
  R.OLM.POSITION.DARK,
  R.ONL.OPL.POSITION.DARK,
  R.INL.IPL.POSITION.DARK,
  R.RNFL.GCL.POSITION.DARK,
  R.VITREOUS.RETINA.POSITION.DARK
)
save.tissue.border.output(
  paste("_tissueBorders__",TO.PROCESS.LIGHT,".png",sep=""),
  FLATTENED.LIGHT.RETINA.RRC,
  R.RPE.POSITION.LIGHT,
  R.OLM.POSITION.LIGHT,
  R.ONL.OPL.POSITION.LIGHT,
  R.INL.IPL.POSITION.LIGHT,
  R.RNFL.GCL.POSITION.LIGHT,
  R.VITREOUS.RETINA.POSITION.LIGHT
)


## now, to export thicknesses
THICKNESS.EXPORT=as.data.frame(matrix(,7,2852))
THICKNESS.EXPORT[1,]=c(NA,seq(-100,2750,1))
THICKNESS.EXPORT[2,]=c(NA,(R.RPE.POSITION.LIGHT[1:2851,1]-R.VITREOUS.RETINA.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[3,]=c(NA,(R.RPE.POSITION.LIGHT[1:2851,1]-R.OLM.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[4,]=c(NA,(R.OLM.POSITION.LIGHT[1:2851,1]-R.ONL.OPL.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[5,]=c(NA,(R.ONL.OPL.POSITION.LIGHT[1:2851,1]-R.INL.IPL.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[6,]=c(NA,(R.INL.IPL.POSITION.LIGHT[1:2851,1]-R.RNFL.GCL.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[7,]=c(NA,(R.RNFL.GCL.POSITION.LIGHT[1:2851,1]-R.VITREOUS.RETINA.POSITION.LIGHT[1:2851,1]))
# Add summed layers for LIGHT, EA
summed_light = as.numeric(THICKNESS.EXPORT[4, 2:2852]) +
              as.numeric(THICKNESS.EXPORT[5, 2:2852]) +
              as.numeric(THICKNESS.EXPORT[6, 2:2852])
THICKNESS.EXPORT[8, ] = c(NA, summed_light)

THICKNESS.EXPORT[,1]=c("Distance_from_Fundus_um",
                       "WholeRetina_um",
                       "RPE_to_OLM_um",
                       "OLM_to_ONL_OPLborder_um",
                       "ONL_OPLborder_to_INL_IPLborder_um",
                       "INL_IPLborder_to_RNFL_GCLborder_um",
                       "RNFL_GCLborder_to_vitreous_um",
		       "Summed_layers")
THICKNESS.EXPORT.LIGHT=THICKNESS.EXPORT

THICKNESS.EXPORT[,]<-NA
THICKNESS.EXPORT[1,]=c(NA,seq(-100,2750,1))
THICKNESS.EXPORT[2,]=c(NA,(R.RPE.POSITION.DARK[1:2851,1]-R.VITREOUS.RETINA.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[3,]=c(NA,(R.RPE.POSITION.DARK[1:2851,1]-R.OLM.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[4,]=c(NA,(R.OLM.POSITION.DARK[1:2851,1]-R.ONL.OPL.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[5,]=c(NA,(R.ONL.OPL.POSITION.DARK[1:2851,1]-R.INL.IPL.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[6,]=c(NA,(R.INL.IPL.POSITION.DARK[1:2851,1]-R.RNFL.GCL.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[7,]=c(NA,(R.RNFL.GCL.POSITION.DARK[1:2851,1]-R.VITREOUS.RETINA.POSITION.DARK[1:2851,1]))
# Add summed layers for the legacy DARK-compatible output, EA
summed_dark = as.numeric(THICKNESS.EXPORT[4, 2:2852]) +
               as.numeric(THICKNESS.EXPORT[5, 2:2852]) +
               as.numeric(THICKNESS.EXPORT[6, 2:2852])
THICKNESS.EXPORT[8, ] = c(NA, summed_dark)
THICKNESS.EXPORT[,1]=c("Distance_from_Fundus_um",
                       "WholeRetina_um",
                       "RPE_to_OLM_um",
                       "OLM_to_ONL_OPLborder_um",
                       "ONL_OPLborder_to_INL_IPLborder_um",
                       "INL_IPLborder_to_RNFL_GCLborder_um",
                       "RNFL_GCLborder_to_vitreous_um",
		       "Summed_layers")
THICKNESS.EXPORT.DARK=THICKNESS.EXPORT

rm(THICKNESS.EXPORT)

write(t(t(THICKNESS.EXPORT.DARK)),ncol=nrow(THICKNESS.EXPORT.DARK),file=paste("_thickness_vs_distance_from_fovea_",TO.PROCESS.DARK,".txt",sep=""),sep="\t")
write(t(t(THICKNESS.EXPORT.LIGHT)),ncol=nrow(THICKNESS.EXPORT.LIGHT),file=paste("_thickness_vs_distance_from_fovea_",TO.PROCESS.LIGHT,".txt",sep=""),sep="\t")


