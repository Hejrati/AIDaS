# Rscript --vanilla verify_step3_deep_layers.R baseline.RData candidate.RData full|deep
# Workspaces are the unrounded precision reference. No tolerance is accepted.
compare_step3_deep_layers <- function(original_file, candidate_file, mode,
                                     check_exports=TRUE) {
  if(!mode %in% c("full","deep")) stop("Mode must be full or deep.",call.=FALSE)
  if(!file.exists(original_file) || !file.exists(candidate_file))
    stop("Both result workspace files must exist.",call.=FALSE)
  original <- new.env(parent=baseenv())
  candidate <- new.env(parent=baseenv())
  load(original_file,envir=original)
  load(candidate_file,envir=candidate)
  checks <- 0L
  assert <- function(condition,label) {
    if(!isTRUE(condition)) stop(label,call.=FALSE)
    checks <<- checks+1L
  }
  read_object <- function(environment,name) {
    if(!exists(name,envir=environment,inherits=FALSE))
      stop(paste("Required result is missing:",name),call.=FALSE)
    get(name,envir=environment,inherits=FALSE)
  }
  same <- function(before,after,label)
    assert(identical(before,after),paste("Exact comparison failed:",label))
  compare_object <- function(name)
    same(read_object(original,name),read_object(candidate,name),name)
  same(read_object(candidate,"deep_layers_only"),identical(mode,"deep"),"saved processing mode")

  if(mode=="full") {
    # Compare every baseline scientific numeric/logical object, including model
    # tables and scalar settings. Paths, executable closures and mode bookkeeping
    # vary between runs and are deliberately not treated as measured results.
    scientific <- Filter(function(name) {
      value <- read_object(original,name)
      is.numeric(value) || is.logical(value) || is.matrix(value) || is.array(value) ||
        is.data.frame(value)
    },ls(envir=original,all.names=TRUE))
    assert(length(scientific)>0L,"The reference workspace contains no scientific results")
    for(name in scientific) compare_object(name)
  } else {
    compare_object("FLATTENED.MARKERS.RRC")
    core <- c("FIRST.GRAND.MEAN","SECOND.GRAND.MEAN","FINAL.GRAND.MEAN","GRAND.PROFILE",
              "SHIFT.POSITION.DARK","SHIFT.POSITION.LIGHT","SHIFT.POSITION.DARK.REFINED",
              "SHIFT.POSITION.LIGHT.REFINED","BEST.LAT.MOVE.DARK","BEST.LAT.MOVE.LIGHT",
              "VERTEX","vertex")
    for(name in core) if(exists(name,envir=original,inherits=FALSE)) compare_object(name)
    for(channel in c("DARK","LIGHT")) {
      compare_object(paste0("IMAGE.INDEX.",channel))
      for(boundary in c("RPE","OLM","ONL.OPL","INL.IPL","VITREOUS.RETINA"))
        compare_object(paste0("R.",boundary,".POSITION.",channel))
      for(boundary in c("RPE","OLM"))
        compare_object(paste0("R.",boundary,".POSITION.",channel,".FOVEA"))
      image_name <- paste0("FLATTENED.",channel,".RETINA.RRC")
      compare_object(image_name)
      normalized_name <- paste0(image_name,".N")
      before <- read_object(original,normalized_name)
      after <- read_object(candidate,normalized_name)
      same(dim(before),dim(after),paste(normalized_name,"dimensions"))
      assert(length(dim(before))==3L && dim(before)[2L]==90L,
             paste(normalized_name,"must preserve the 90-column depth grid"))
      same(dimnames(before),dimnames(after),paste(normalized_name,"dimension names"))
      for(slice in seq_len(dim(before)[3L])) {
        same(before[,35:90,slice,drop=FALSE],after[,35:90,slice,drop=FALSE],
             paste(normalized_name,"retained depths, slice",slice))
        assert(all(is.na(after[,1:34,slice,drop=FALSE])),
               paste("Suppressed normalized samples are not all missing:",channel,"slice",slice))
      }
      profile_name <- paste0(normalized_name,".profiles")
      before_profile <- read_object(original,profile_name)
      after_profile <- read_object(candidate,profile_name)
      same(dim(before_profile),dim(after_profile),paste(profile_name,"dimensions"))
      same(dimnames(before_profile),dimnames(after_profile),paste(profile_name,"dimension names"))
      same(before_profile[,1L],after_profile[,1L],paste(profile_name,"depth coordinates"))
      same(before_profile[35:90,,drop=FALSE],after_profile[35:90,,drop=FALSE],
           paste(profile_name,"retained measurements"))
      assert(all(is.na(after_profile[1:34,-1L,drop=FALSE])),
             paste("Suppressed profile measurements remain:",channel))
      compare_object(paste0(normalized_name,".fovea.profiles"))
      compare_object(paste0("MAIN.",channel,".OUTPUTS.fovea"))
      summary_name <- paste0("MAIN.",channel,".OUTPUTS")
      before_summary <- read_object(original,summary_name)
      after_summary <- read_object(candidate,summary_name)
      same(dim(before_summary),dim(after_summary),paste(summary_name,"dimensions"))
      same(dimnames(before_summary),dimnames(after_summary),paste(summary_name,"dimension names"))
      retained <- c("INDEX","apparent.angle.retinal.strip..degrees","RPE.to.OLM.distance")
      same(before_summary[,retained,drop=FALSE],after_summary[,retained,drop=FALSE],
           paste(summary_name,"retained summary columns"))
      assert(all(is.na(after_summary[,c("whole.retinal.thick","RNFL.thickness"),drop=FALSE])),
             paste("Suppressed summary measurements remain:",channel))
      rnfl_name <- paste0("R.RNFL.GCL.POSITION.",channel)
      before_rnfl <- read_object(original,rnfl_name)
      after_rnfl <- read_object(candidate,rnfl_name)
      same(dim(before_rnfl),dim(after_rnfl),paste(rnfl_name,"placeholder dimensions"))
      same(dimnames(before_rnfl),dimnames(after_rnfl),paste(rnfl_name,"placeholder names"))
      assert(all(is.na(after_rnfl)),paste("GCL/RNFL results unexpectedly retained:",channel))
    }
    same(read_object(candidate,"retained_main_depth_columns"),35:90,"retained main depth columns")
  }

  if(check_exports) {
    original_dir <- dirname(original_file)
    candidate_dir <- dirname(candidate_file)
    bytes <- function(path) {
      if(!file.exists(path)) stop(paste("Required export is missing:",path),call.=FALSE)
      readBin(path,"raw",n=file.info(path)$size)
    }
    for(channel in c("DARK","LIGHT")) {
      acquisition <- read_object(original,paste0("TO.PROCESS.",channel))
      same(acquisition,read_object(candidate,paste0("TO.PROCESS.",channel)),paste(channel,"acquisition name"))
      filename <- paste0("_thickness_vs_distance_from_fovea_",acquisition,".txt")
      original_table <- read.table(file.path(original_dir,filename),header=TRUE,sep="\t",
                                   colClasses="character",check.names=FALSE)
      candidate_table <- read.table(file.path(candidate_dir,filename),header=TRUE,sep="\t",
                                    colClasses="character",check.names=FALSE)
      assert(ncol(original_table)==8L && ncol(candidate_table)==8L,
             paste(channel,"thickness export must retain eight column positions"))
      same(names(original_table),names(candidate_table),paste(channel,"thickness column labels"))
      same(nrow(original_table),nrow(candidate_table),paste(channel,"thickness distance extent"))
      if(mode=="full") {
        same(bytes(file.path(original_dir,filename)),bytes(file.path(candidate_dir,filename)),
             paste(channel,"full thickness export bytes"))
      } else {
        # Masking other rows can legitimately change text precision. Recreate
        # the unchanged retained subtraction formulas from unrounded matrices,
        # then format the complete eight-row masked table before comparing.
        boundary <- function(name) read_object(original,paste0("R.",name,".POSITION.",channel))[1:2851,1L]
        expected <- as.data.frame(matrix(NA_real_,nrow=8L,ncol=2852L))
        expected[1L,] <- c(NA,seq(-100,2750,1))
        expected[3L,] <- c(NA,boundary("RPE")-boundary("OLM"))
        expected[4L,] <- c(NA,boundary("OLM")-boundary("ONL.OPL"))
        expected[5L,] <- c(NA,boundary("ONL.OPL")-boundary("INL.IPL"))
        expected[,1L] <- names(original_table)
        connection <- textConnection("expected_lines","w",local=TRUE)
        write(as.matrix(expected),file=connection,ncolumns=8L,sep="\t")
        close(connection)
        same(expected_lines,readLines(file.path(candidate_dir,filename),warn=FALSE),
             paste(channel,"masked eight-column thickness export"))
        assert(all(is.na(candidate_table[,c(2L,6L,7L,8L),drop=FALSE])),
               paste(channel,"omitted thickness slots contain results"))
      }
      profile_files <- if(channel=="DARK") c(paste0("_dark_profiles_",acquisition,".txt"),
        paste0("_fovea_dark_profiles_",acquisition,".txt")) else
        c(paste0("_light_profiles_",acquisition,".txt"),paste0("_fovea_light_profiles_",acquisition,".txt"))
      for(filename in profile_files) {
        actual <- bytes(file.path(candidate_dir,filename))
        if(mode=="full" || grepl("_fovea_",filename,fixed=TRUE))
          same(bytes(file.path(original_dir,filename)),actual,paste(filename,"profile export bytes"))
      }
    }
    if(mode=="deep") {
      note <- file.path(candidate_dir,"deep_only_output_notes.txt")
      assert(file.exists(note),"Deep-only output notes are missing")
      note_lines <- readLines(note,warn=FALSE)
      assert(any(grepl("35:90",note_lines,fixed=TRUE)) && any(grepl("1:34",note_lines,fixed=TRUE)),
             "Deep-only notes do not identify retained and suppressed depth columns")
    }
  }
  cat("PASS:",checks,"exact",mode,"Step 3 deep-layer comparison checks\n")
  invisible(checks)
}

if(sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if(length(args)!=3L) stop("Usage: baseline.RData candidate.RData full|deep",call.=FALSE)
  compare_step3_deep_layers(args[1L],args[2L],args[3L])
}
