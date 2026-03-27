## use this either immediately after running "RAW_OCT_PROCESSING".R
## (like, in the same RGui window)
## or after opening the file saved by "RAW_OCT_PROCESSING".R (which should have the prefix "_done_"


### this assumes the light and dark images are one-and-done (rather than each being unique) and so just takes the first
### member of each stack. 


dev.new(width=12,height=4)
# to use the pdf style, cut out the preding line, and comment out the save.image line at the end
# but uncomment pdf() and dev.off() lines
#pdf(paste("_tissueBorders__",TO.PROCESS.DARK,".pdf",sep=""),width=12.8,height=3.7)
image(as.vector(seq(-100,2750,1)),as.vector(seq(-30,430,1)),
      as.matrix(FLATTENED.DARK.RETINA.RRC[,dim(FLATTENED.DARK.RETINA.RRC)[2]:1,1]),
      xlab="Distance from Fovea (microns)",
      ylab="Distance from RPE (microns)", col = gray.colors(254))

matlines(seq(-100,2750,1),431-R.RPE.POSITION.DARK[1:2851,1],col="red")
matlines(seq(-100,2750,1),431-R.OLM.POSITION.DARK[1:2851,1],col="blue")
matlines(seq(-100,2750,1),431-R.ONL.OPL.POSITION.DARK[1:2851,1],col="red")
matlines(seq(-100,2750,1),431-R.INL.IPL.POSITION.DARK[1:2851,1],col="blue")
matlines(seq(-100,2750,1),431-R.RNFL.GCL.POSITION.DARK[1:2851,1],col="red")
matlines(seq(-100,2750,1),431-R.VITREOUS.RETINA.POSITION.DARK[1:2851,1],col="blue")
#matlines(seq(-100,50,1),431-R.RPE.POSITION.DARK.FOVEA[1:151,1],col="red")
#matlines(seq(-100,50,1),431-R.OLM.POSITION.DARK.FOVEA[1:151,1],col="blue")
#<end>
#dev.off()
savePlot(filename=paste("_tissueBorders__",TO.PROCESS.DARK,".png",sep=""),type="png")



dev.new(width=12,height=4)
# to use the pdf style, cut out the preding line, and comment out the save.image line at the end
# but uncomment pdf() and dev.off() lines
#pdf(paste("_tissueBorders__",TO.PROCESS.LIGHT,".pdf",sep=""),width=12.8,height=3.7)
image(as.vector(seq(-100,2750,1)),as.vector(seq(-30,430,1)),
      as.matrix(FLATTENED.LIGHT.RETINA.RRC[,dim(FLATTENED.LIGHT.RETINA.RRC)[2]:1,1]),
      xlab="Distance from Fovea (microns)",
      ylab="Distance from RPE (microns)", col = gray.colors(254))

matlines(seq(-100,2750,1),431-R.RPE.POSITION.LIGHT[1:2851,1],col="red")
matlines(seq(-100,2750,1),431-R.OLM.POSITION.LIGHT[1:2851,1],col="blue")
matlines(seq(-100,2750,1),431-R.ONL.OPL.POSITION.LIGHT[1:2851,1],col="red")
matlines(seq(-100,2750,1),431-R.INL.IPL.POSITION.LIGHT[1:2851,1],col="blue")
matlines(seq(-100,2750,1),431-R.RNFL.GCL.POSITION.LIGHT[1:2851,1],col="red")
matlines(seq(-100,2750,1),431-R.VITREOUS.RETINA.POSITION.LIGHT[1:2851,1],col="blue")
#matlines(seq(-100,50,1),431-R.RPE.POSITION.LIGHT.FOVEA[1:151,1],col="red")
#matlines(seq(-100,50,1),431-R.OLM.POSITION.LIGHT.FOVEA[1:151,1],col="blue")
#<end>
#dev.off()
savePlot(filename=paste("_tissueBorders__",TO.PROCESS.LIGHT,".png",sep=""),type="png")


## now, to export thicknesses
THICKNESS.EXPORT=as.data.frame(matrix(,7,2852))
THICKNESS.EXPORT[1,]=c(NA,seq(-100,2750,1))
THICKNESS.EXPORT[2,]=c(NA,(R.RPE.POSITION.LIGHT[1:2851,1]-R.VITREOUS.RETINA.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[3,]=c(NA,(R.RPE.POSITION.LIGHT[1:2851,1]-R.OLM.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[4,]=c(NA,(R.OLM.POSITION.LIGHT[1:2851,1]-R.ONL.OPL.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[5,]=c(NA,(R.ONL.OPL.POSITION.LIGHT[1:2851,1]-R.INL.IPL.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[6,]=c(NA,(R.INL.IPL.POSITION.LIGHT[1:2851,1]-R.RNFL.GCL.POSITION.LIGHT[1:2851,1]))
THICKNESS.EXPORT[7,]=c(NA,(R.RNFL.GCL.POSITION.LIGHT[1:2851,1]-R.VITREOUS.RETINA.POSITION.LIGHT[1:2851,1]))
# Add summed layers for DARK, EA
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
THICKNESS.EXPORT.LIGHT=THICKNESS.EXPORT

THICKNESS.EXPORT[,]<-NA
THICKNESS.EXPORT[1,]=c(NA,seq(-100,2750,1))
THICKNESS.EXPORT[2,]=c(NA,(R.RPE.POSITION.DARK[1:2851,1]-R.VITREOUS.RETINA.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[3,]=c(NA,(R.RPE.POSITION.DARK[1:2851,1]-R.OLM.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[4,]=c(NA,(R.OLM.POSITION.DARK[1:2851,1]-R.ONL.OPL.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[5,]=c(NA,(R.ONL.OPL.POSITION.DARK[1:2851,1]-R.INL.IPL.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[6,]=c(NA,(R.INL.IPL.POSITION.DARK[1:2851,1]-R.RNFL.GCL.POSITION.DARK[1:2851,1]))
THICKNESS.EXPORT[7,]=c(NA,(R.RNFL.GCL.POSITION.DARK[1:2851,1]-R.VITREOUS.RETINA.POSITION.DARK[1:2851,1]))
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
THICKNESS.EXPORT.DARK=THICKNESS.EXPORT

rm(THICKNESS.EXPORT)

write(t(t(THICKNESS.EXPORT.DARK)),ncol=nrow(THICKNESS.EXPORT.DARK),file=paste("_thickness_vs_distance_from_fovea_",TO.PROCESS.DARK,".txt",sep=""),sep="\t")
write(t(t(THICKNESS.EXPORT.LIGHT)),ncol=nrow(THICKNESS.EXPORT.LIGHT),file=paste("_thickness_vs_distance_from_fovea_",TO.PROCESS.LIGHT,".txt",sep=""),sep="\t")


