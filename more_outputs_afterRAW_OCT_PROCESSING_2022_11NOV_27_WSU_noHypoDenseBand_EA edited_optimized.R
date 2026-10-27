## Optimized companion for the original RAW_OCT_PROCESSING output script.
## Run immediately after processing, or after loading its _done_ workspace.
## Exports still use getwd(), slice 1, and the original filenames and labels.
## Default graphics preserve legacy rendering. Set AIDAS_STEP3_RASTER=1 only
## when faster raster rendering is preferred over byte-identical PNGs.

# Keep unexpected diagnostic graphics in memory; named PNG exports are unchanged.
options(device=function(...) grDevices::pdf(file=NULL))

### LIGHT is the app's acquisition channel. Recreate only missing DARK slots.
if(!exists("TO.PROCESS.DARK")) TO.PROCESS.DARK="DARK"
if(!exists("FLATTENED.DARK.RETINA.RRC")) FLATTENED.DARK.RETINA.RRC=FLATTENED.LIGHT.RETINA.RRC
if(!exists("R.RPE.POSITION.DARK")) R.RPE.POSITION.DARK=R.RPE.POSITION.LIGHT
if(!exists("R.OLM.POSITION.DARK")) R.OLM.POSITION.DARK=R.OLM.POSITION.LIGHT
if(!exists("R.ONL.OPL.POSITION.DARK")) R.ONL.OPL.POSITION.DARK=R.ONL.OPL.POSITION.LIGHT
if(!exists("R.INL.IPL.POSITION.DARK")) R.INL.IPL.POSITION.DARK=R.INL.IPL.POSITION.LIGHT
if(!exists("R.RNFL.GCL.POSITION.DARK")) R.RNFL.GCL.POSITION.DARK=R.RNFL.GCL.POSITION.LIGHT
if(!exists("R.VITREOUS.RETINA.POSITION.DARK")) R.VITREOUS.RETINA.POSITION.DARK=R.VITREOUS.RETINA.POSITION.LIGHT

make_thickness_export <- function(rpe, olm, onl_opl, inl_ipl, rnfl_gcl, vitreous_retina) {
  distance <- seq(-100, 2750, by=1)
  rows <- seq_along(distance)
  borders <- list(rpe=rpe, olm=olm, onl_opl=onl_opl, inl_ipl=inl_ipl,
                  rnfl_gcl=rnfl_gcl, vitreous_retina=vitreous_retina)

  for(name in names(borders)) {
    border <- borders[[name]]
    if(!is.matrix(border) || !is.numeric(border) || is.complex(border) ||
       nrow(border) < length(rows) || ncol(border) < 1L) {
      stop(sprintf("`%s` must be a numeric matrix with at least 2851 rows and 1 column.",
                   name), call.=FALSE)
    }
    borders[[name]] <- border[rows, 1L]
  }

  values <- matrix(NA_real_, nrow=8L, ncol=length(distance)+1L)
  value_cols <- seq_along(distance)+1L
  values[1L, value_cols] <- distance
  values[2L, value_cols] <- borders$rpe-borders$vitreous_retina
  values[3L, value_cols] <- borders$rpe-borders$olm
  values[4L, value_cols] <- borders$olm-borders$onl_opl
  values[5L, value_cols] <- borders$onl_opl-borders$inl_ipl
  values[6L, value_cols] <- borders$inl_ipl-borders$rnfl_gcl
  values[7L, value_cols] <- borders$rnfl_gcl-borders$vitreous_retina
  # Preserve left-to-right arithmetic, NA propagation, and floating-point order.
  values[8L, value_cols] <- (values[4L, value_cols]+values[5L, value_cols])+
                          values[6L, value_cols]

  result <- as.data.frame(values, stringsAsFactors=FALSE)
  result[[1L]] <- c("Distance_from_Fundus_um",
                   "WholeRetina_um",
                   "RPE_to_OLM_um",
                   "OLM_to_ONL_OPLborder_um",
                   "ONL_OPLborder_to_INL_IPLborder_um",
                   "INL_IPLborder_to_RNFL_GCLborder_um",
                   "RNFL_GCLborder_to_vitreous_um",
                   "Summed_layers")
  result
}

write_thickness_export <- function(thickness, channel_name, output_dir=getwd()) {
  target <- file.path(output_dir, paste0("_thickness_vs_distance_from_fovea_",
                                        channel_name, ".txt"))
  # The legacy mixed data-frame conversion preserves per-column formatting.
  # write() emits eight columns in column-major order, without two transposes.
  write(as.matrix(thickness), file=target, ncolumns=nrow(thickness), sep="\t")
  invisible(target)
}

same_export_borders <- function(light, dark) {
  rows <- seq_len(2851L)
  all(vapply(seq_along(light), function(i) {
    identical(light[[i]][rows, 1L], dark[[i]][rows, 1L])
  }, logical(1)))
}

save.tissue.border.output <- function(filename, flattened, rpe, olm, onl.opl, inl.ipl, rnfl.gcl, vitreous.retina) {
  x.axis=seq(-100,2750,1)
  y.axis=seq(-30,430,1)
  target=file.path(getwd(),filename)

  png(filename=target,width=1600,height=520)
  on.exit(dev.off())
  plot.image <- image
  if(identical(Sys.getenv("AIDAS_STEP3_RASTER", unset=""), "1")) {
    plot.image <- function(...) image(..., useRaster=TRUE)
  }
  # Omit useRaster in the default path to preserve preferRaster graphics options.
  plot.image(as.vector(x.axis),as.vector(y.axis),
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

.output.light.borders <- list(
  R.RPE.POSITION.LIGHT, R.OLM.POSITION.LIGHT, R.ONL.OPL.POSITION.LIGHT,
  R.INL.IPL.POSITION.LIGHT, R.RNFL.GCL.POSITION.LIGHT, R.VITREOUS.RETINA.POSITION.LIGHT
)
.output.dark.borders <- list(
  R.RPE.POSITION.DARK, R.OLM.POSITION.DARK, R.ONL.OPL.POSITION.DARK,
  R.INL.IPL.POSITION.DARK, R.RNFL.GCL.POSITION.DARK, R.VITREOUS.RETINA.POSITION.DARK
)
.output.same.borders <- same_export_borders(.output.light.borders, .output.dark.borders)

.output.dark.target <- save.tissue.border.output(
  paste("_tissueBorders__",TO.PROCESS.DARK,".png",sep=""),
  FLATTENED.DARK.RETINA.RRC,
  R.RPE.POSITION.DARK, R.OLM.POSITION.DARK, R.ONL.OPL.POSITION.DARK,
  R.INL.IPL.POSITION.DARK, R.RNFL.GCL.POSITION.DARK, R.VITREOUS.RETINA.POSITION.DARK
)
.output.light.filename <- paste("_tissueBorders__",TO.PROCESS.LIGHT,".png",sep="")
.output.light.target <- file.path(getwd(), .output.light.filename)
.output.same.plot <- .output.same.borders &&
  identical(FLATTENED.DARK.RETINA.RRC[,,1L], FLATTENED.LIGHT.RETINA.RRC[,,1L])
.output.plot.copied <- FALSE
if(.output.same.plot) {
  .output.dark.path <- normalizePath(.output.dark.target, winslash="/", mustWork=FALSE)
  .output.light.path <- normalizePath(.output.light.target, winslash="/", mustWork=FALSE)
  if(.Platform$OS.type == "windows") {
    .output.dark.path <- tolower(.output.dark.path)
    .output.light.path <- tolower(.output.light.path)
  }
  .output.plot.copied <- identical(.output.dark.path, .output.light.path) ||
    isTRUE(file.copy(.output.dark.target, .output.light.target, overwrite=TRUE))
}
if(!.output.plot.copied) {
  save.tissue.border.output(
    .output.light.filename,
    FLATTENED.LIGHT.RETINA.RRC,
    R.RPE.POSITION.LIGHT, R.OLM.POSITION.LIGHT, R.ONL.OPL.POSITION.LIGHT,
    R.INL.IPL.POSITION.LIGHT, R.RNFL.GCL.POSITION.LIGHT, R.VITREOUS.RETINA.POSITION.LIGHT
  )
}

THICKNESS.EXPORT.LIGHT <- do.call(make_thickness_export, .output.light.borders)
THICKNESS.EXPORT.DARK <- if(.output.same.borders) THICKNESS.EXPORT.LIGHT else
                       do.call(make_thickness_export, .output.dark.borders)

write_thickness_export(THICKNESS.EXPORT.DARK, TO.PROCESS.DARK)
write_thickness_export(THICKNESS.EXPORT.LIGHT, TO.PROCESS.LIGHT)

rm(list=intersect(c(".output.light.borders", ".output.dark.borders", ".output.same.borders",
                    ".output.dark.target", ".output.light.filename", ".output.light.target",
                    ".output.same.plot", ".output.plot.copied", ".output.dark.path",
                    ".output.light.path"), ls(all.names=TRUE)))
