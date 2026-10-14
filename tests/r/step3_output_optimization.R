# R 3.3.1 byte regression for the separate optimized output companion.
# Image creation is stubbed for adversarial tables, then checked with real PNGs.
main <- function() {
  args <- commandArgs(trailingOnly=TRUE)
  original <- if(length(args)) args[1] else
    "more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited.R"
  optimized <- if(length(args)>1L) args[2] else
    "more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited_optimized.R"
  original <- normalizePath(original,mustWork=TRUE)
  optimized <- normalizePath(optimized,mustWork=TRUE)
  original.expressions <- parse(original)
  optimized.expressions <- parse(optimized)
  previous.directory <- getwd()
  previous.raster <- Sys.getenv("AIDAS_STEP3_RASTER",unset=NA_character_)
  previous.raster.preference <- getOption("preferRaster")
  Sys.unsetenv("AIDAS_STEP3_RASTER")
  options(preferRaster=FALSE)
  sandbox <- tempfile("step3-output-regression-")
  dir.create(sandbox)
  on.exit({
    setwd(previous.directory)
    options(preferRaster=previous.raster.preference)
    if(is.na(previous.raster)) Sys.unsetenv("AIDAS_STEP3_RASTER") else
      Sys.setenv(AIDAS_STEP3_RASTER=previous.raster)
    unlink(sandbox,recursive=TRUE)
  },add=TRUE)
  checks <- 0L
  assert <- function(condition,label) {
    if(!isTRUE(condition)) stop(label)
    checks <<- checks+1L
  }
  raw.file <- function(path) readBin(path,"raw",n=file.info(path)$size)
  same.bytes <- function(expected,actual,label) {
    assert(identical(expected,actual),paste(label,"bytes differ"))
  }
  border.names <- c("R.RPE.POSITION","R.OLM.POSITION","R.ONL.OPL.POSITION",
                    "R.INL.IPL.POSITION","R.RNFL.GCL.POSITION","R.VITREOUS.RETINA.POSITION")
  fixture <- function() {
    value <- list(TO.PROCESS.LIGHT="LIGHT",TO.PROCESS.DARK="DARK")
    rows <- 2865L
    for(i in seq_along(border.names)) {
      border <- matrix(rep(430-(i-1)*35 + sin(seq_len(rows)/71)*3,3),nrow=rows)
      border[,2] <- border[,2]+17
      border[,3] <- border[,3]-9
      value[[paste0(border.names[i],".LIGHT")]] <- border
      value[[paste0(border.names[i],".DARK")]] <- border
    }
    image <- array(as.numeric(seq_len(2851*461*2) %% 254),dim=c(2851,461,2))
    value$FLATTENED.LIGHT.RETINA.RRC <- image
    value$FLATTENED.DARK.RETINA.RRC <- image
    value
  }
  baseline <- fixture()
  cases <- list(normal=baseline)
  distinct <- baseline
  distinct$R.RPE.POSITION.DARK[1:2851,1] <- distinct$R.RPE.POSITION.DARK[1:2851,1]+4.25
  distinct$R.ONL.OPL.POSITION.DARK[1:2851,1] <- distinct$R.ONL.OPL.POSITION.DARK[1:2851,1]-2.75
  distinct$FLATTENED.DARK.RETINA.RRC[1,1,1] <- 7
  cases$distinct.channels <- distinct
  exceptional <- baseline
  for(i in seq_along(border.names)) {
    name <- paste0(border.names[i],".LIGHT")
    exceptional[[name]][10:15,1] <- c(NA_real_,NaN,Inf,-Inf,1e300,-1e300)
    exceptional[[paste0(border.names[i],".DARK")]] <- exceptional[[name]]
  }
  cases$exceptional <- exceptional
  arithmetic <- baseline
  for(channel in c("LIGHT","DARK")) {
    arithmetic[[paste0("R.OLM.POSITION.",channel)]][100:101,1] <- c(1,1)
    arithmetic[[paste0("R.ONL.OPL.POSITION.",channel)]][100:101,1] <- c(1e16,-1e16)
    arithmetic[[paste0("R.INL.IPL.POSITION.",channel)]][100:101,1] <- c(-1e16,1)
    arithmetic[[paste0("R.RNFL.GCL.POSITION.",channel)]][100:101,1] <- c(0,0)
  }
  cases$floating.order <- arithmetic
  unused <- baseline
  for(name in border.names) {
    name <- paste0(name,".DARK")
    unused[[name]][2852:2865,1] <- NA_real_
    unused[[name]][,2:3] <- unused[[name]][,2:3]+99
  }
  unused$FLATTENED.DARK.RETINA.RRC[,,2] <- unused$FLATTENED.DARK.RETINA.RRC[,,2]+1
  cases$padding.and.later.slices <- unused
  fallback <- baseline
  fallback[grepl("DARK",names(fallback),fixed=TRUE)] <- NULL
  cases$light.only.fallback <- fallback
  changed.image <- baseline
  changed.image$FLATTENED.DARK.RETINA.RRC[1,1,1] <- 7
  cases$changed.dark.image <- changed.image
  changed.border <- baseline
  changed.border$R.RPE.POSITION.DARK[10,1] <- changed.border$R.RPE.POSITION.DARK[10,1]+1
  cases$changed.dark.border <- changed.border

  execute <- function(expressions,input,directory) {
    dir.create(directory,recursive=TRUE)
    setwd(directory)
    output.names <- c("_tissueBorders__DARK.png","_tissueBorders__LIGHT.png",
                     "_thickness_vs_distance_from_fovea_DARK.txt",
                     "_thickness_vs_distance_from_fovea_LIGHT.txt")
    for(name in output.names) writeBin(charToRaw("stale output"),name)
    environment <- list2env(input,parent=globalenv())
    plot.files <- character(0)
    real.plotter <- NULL
    stub.plotter <- function(filename,...) {
      plot.files <<- c(plot.files,filename)
      writeBin(charToRaw("fresh synthetic PNG"),filename)
      invisible(file.path(getwd(),filename))
    }
    for(expr in expressions) {
      plot.definition <- is.call(expr) && is.symbol(expr[[1]]) &&
        as.character(expr[[1]]) %in% c("<-","=") &&
        identical(expr[[2]],as.name("save.tissue.border.output"))
      eval(expr,environment)
      if(plot.definition) {
        real.plotter <- environment$save.tissue.border.output
        environment$save.tissue.border.output <- stub.plotter
      }
    }
    if(is.null(real.plotter)) stop("Output plot helper not found")
    files <- lapply(output.names,function(name) raw.file(file.path(directory,name)))
    names(files) <- output.names
    list(environment=environment,files=files,plot.files=plot.files,real.plotter=real.plotter)
  }
  table.values <- function(value) unlist(lapply(2:2852,function(j) as.numeric(value[,j])),use.names=FALSE)
  for(name in names(cases)) {
    legacy <- execute(original.expressions,cases[[name]],file.path(sandbox,name,"original"))
    improved <- execute(optimized.expressions,cases[[name]],file.path(sandbox,name,"optimized"))
    for(filename in names(legacy$files))
      same.bytes(legacy$files[[filename]],improved$files[[filename]],paste(name,filename))
    for(channel in c("LIGHT","DARK")) {
      object <- paste0("THICKNESS.EXPORT.",channel)
      expected <- get(object,legacy$environment)
      actual <- get(object,improved$environment)
      same.bytes(serialize(table.values(expected),NULL,version=2),
                 serialize(table.values(actual),NULL,version=2),paste(name,channel,"numeric columns"))
      assert(identical(as.character(expected[,1]),as.character(actual[,1])),
             paste(name,channel,"labels differ"))
    }
    expected.plots <- if(name %in% c("distinct.channels","changed.dark.image","changed.dark.border")) 2L else 1L
    assert(length(legacy$plot.files)==2L,paste(name,"legacy plot count"))
    assert(length(improved$plot.files)==expected.plots,paste(name,"optimized plot reuse"))
    if(name=="floating.order") {
      assert(identical(table.values(get("THICKNESS.EXPORT.LIGHT",improved$environment))[
        c((100-1)*8+8,(101-1)*8+8)],c(0,1)),"Summed layer arithmetic order changed")
    }
    cat("CASE PASS:",name,"\n")
    flush.console()
  }
  # New helper validation should fail clearly before producing malformed tables.
  builder <- get("make_thickness_export",improved$environment)
  border.args <- unname(baseline[paste0(border.names,".LIGHT")])
  invalid.borders <- list(seq_len(2851),as.data.frame(border.args[[1]]),
                          matrix(0,2850,1),matrix(0,2851,0),
                          matrix(TRUE,2851,1),matrix(1+1i,2851,1))
  for(invalid in invalid.borders) {
    invalid.args <- border.args
    invalid.args[[1]] <- invalid
    failure <- tryCatch({do.call(builder,invalid.args); NULL},
                        error=function(e) conditionMessage(e))
    assert(!is.null(failure) && grepl("rpe",failure,fixed=TRUE),
           "Invalid rpe matrix was not rejected clearly")
  }
  # Check the unchanged default renderer with actual PNG devices.
  plot.args <- c(list(filename="real-default.png",flattened=baseline$FLATTENED.LIGHT.RETINA.RRC),
                 unname(baseline[paste0(border.names,".LIGHT")]))
  original.png.dir <- file.path(sandbox,"real-original")
  optimized.png.dir <- file.path(sandbox,"real-optimized")
  legacy <- execute(original.expressions,baseline,original.png.dir)
  improved <- execute(optimized.expressions,baseline,optimized.png.dir)
  for(prefer.raster in c(FALSE,TRUE)) {
    options(preferRaster=prefer.raster)
    plot.args$filename <- if(prefer.raster) "real-prefer-raster.png" else "real-default.png"
    setwd(original.png.dir)
    do.call(legacy$real.plotter,plot.args)
    setwd(optimized.png.dir)
    do.call(improved$real.plotter,plot.args)
    same.bytes(raw.file(file.path(original.png.dir,plot.args$filename)),
               raw.file(file.path(optimized.png.dir,plot.args$filename)),
               paste("Actual PNG preferRaster",prefer.raster))
  }
  options(preferRaster=FALSE)
  Sys.setenv(AIDAS_STEP3_RASTER="1")
  plot.args$filename <- "real-opt-in-raster.png"
  setwd(optimized.png.dir)
  do.call(improved$real.plotter,plot.args)
  same.bytes(raw.file(file.path(original.png.dir,"real-prefer-raster.png")),
             raw.file(file.path(optimized.png.dir,plot.args$filename)),"Explicit raster mode")
  cat("PASS:",checks,"optimized companion output compatibility checks on",R.version.string,"\n")
}
main()
