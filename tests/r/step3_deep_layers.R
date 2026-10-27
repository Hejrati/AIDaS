# Functional numerical regression for the separately bundled deep-layer variant.
main <- function() {
  paths <- c(main="RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R",
    candidate="RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized_deep_layers.R",
    output="more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited_optimized.R",
    candidate.output="more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited_optimized_deep_layers.R")
  scripts <- lapply(paths,parse)
  sandbox <- tempfile("step3-deep-regression-")
  dir.create(sandbox)
  previous <- getwd()
  previous.mode <- Sys.getenv("AIDAS_STEP3_DEEP_ONLY",unset=NA_character_)
  on.exit({setwd(previous); unlink(sandbox,recursive=TRUE)
    if(is.na(previous.mode)) Sys.unsetenv("AIDAS_STEP3_DEEP_ONLY") else
      Sys.setenv(AIDAS_STEP3_DEEP_ONLY=previous.mode)},add=TRUE)
  checks <- 0L
  assert <- function(condition,label) {
    if(!isTRUE(condition)) stop(label,call.=FALSE)
    checks <<- checks+1L
  }
  same <- function(expected,actual,label) assert(identical(expected,actual),paste(label,"changed"))
  call.to <- function(expr,name) is.call(expr) && identical(expr[[1L]],as.name(name))
  assigned <- function(expr,name=NULL) {
    valid <- is.call(expr) && is.symbol(expr[[1L]]) &&
      as.character(expr[[1L]]) %in% c("<-","=") && length(expr)==3L
    valid && (is.null(name) || identical(expr[[2L]],as.name(name)))
  }
  collect <- function(expr,predicate) {
    if(missing(expr)) return(list())
    result <- if(predicate(expr)) list(expr) else list()
    if(is.call(expr) || is.expression(expr)) for(child in as.list(expr))
      result <- c(result,collect(child,predicate))
    result
  }
  all.calls <- function(expressions,name) collect(expressions,function(expr) call.to(expr,name))
  define.functions <- function(expressions,environment) {
    for(expr in expressions) if(assigned(expr) && call.to(expr[[3L]],"function")) eval(expr,environment)
  }
  environments <- lapply(scripts,function(expressions) {
    environment <- new.env(parent=globalenv())
    define.functions(expressions,environment)
    environment$deep_layers_only <- FALSE
    environment
  })
  raw.file <- function(path) readBin(path,"raw",n=file.info(path)$size)
  assignments <- function(expressions,name) which(vapply(expressions,assigned,logical(1),name=name))

  # The switch defaults to the unchanged path and accepts only the documented
  # literal values. Read only its real startup statements, not a test copy.
  mode.start <- assignments(scripts$candidate,"deep_mode_setting")[1L]
  mode.end <- which(vapply(scripts$candidate,function(expr) call.to(expr,"rm") &&
    identical(expr[[2L]],as.name("deep_mode_setting")),logical(1)))[1L]
  assert(is.finite(mode.start) && is.finite(mode.end),"Deep-mode startup configuration is missing")
  read.mode <- function(setting) {
    if(is.na(setting)) Sys.unsetenv("AIDAS_STEP3_DEEP_ONLY") else Sys.setenv(AIDAS_STEP3_DEEP_ONLY=setting)
    environment <- new.env(parent=baseenv())
    for(expr in scripts$candidate[mode.start:mode.end]) eval(expr,environment)
    environment$deep_layers_only
  }
  same(FALSE,read.mode(NA_character_),"default full mode")
  same(FALSE,read.mode("0"),"explicit full mode")
  same(TRUE,read.mode("1"),"deep mode")
  for(setting in c("TRUE","false","2"," ")) {
    failure <- tryCatch({read.mode(setting); NULL},error=conditionMessage)
    assert(!is.null(failure),paste("Invalid deep mode was accepted:",setting))
  }

  # Probe the actual GCL detection condition, including short circuit behavior.
  detections <- Filter(function(expr) "deep_layers_only" %in% all.names(expr[[2L]]) &&
    "SEGMENT" %in% all.names(expr[[2L]]),all.calls(scripts$candidate,"if"))
  assert(length(detections)==1L,"Expected one conditional GCL detection branch")
  original.detection <- Filter(function(expr)
    identical(expr[[2L]],quote(!(is.na(SEGMENT[2])))),all.calls(scripts$main,"if"))
  assert(length(original.detection)==1L,"Original GCL detection reference is ambiguous")
  same(original.detection[[1L]][[3L]],detections[[1L]][[3L]],"GCL detection body")
  same(FALSE,eval(detections[[1L]][[2L]],list2env(list(deep_layers_only=TRUE),parent=baseenv())),
       "deep detection short circuit")
  for(value in c(NA_real_,100)) {
    probe <- list2env(list(deep_layers_only=FALSE,SEGMENT=c(90,value,120,140,160)),parent=baseenv())
    same(eval(original.detection[[1L]][[2L]],probe),eval(detections[[1L]][[2L]],probe),
         "full-mode GCL detection condition")
  }
  # Shared and deeper assignments must not be rewritten while removing GCL work.
  retained <- as.vector(outer(c("RPE","OLM","ONL.OPL","INL.IPL","VITREOUS.RETINA"),
                              c("DARK","LIGHT"),function(x,y) paste0(x,".POSITION.",y)))
  retained <- c(retained,paste0("R.",retained))
  retained.assignments <- function(expressions) collect(expressions,function(expr)
    assigned(expr) && any(all.names(expr[[2L]]) %in% retained))
  same(retained.assignments(scripts$main),retained.assignments(scripts$candidate),
       "retained/shared boundary assignment expressions and ordering")
  guards <- Filter(function(expr) identical(expr[[2L]],quote(!deep_layers_only)),
                   all.calls(scripts$candidate,"if"))
  correction.guards <- Filter(function(expr) "MODEL.5" %in% all.names(expr[[3L]]),guards)
  smoothing.guards <- Filter(function(expr) "S.sp" %in% all.names(expr[[3L]]) &&
    any(c("R.RNFL.GCL.POSITION.DARK","R.RNFL.GCL.POSITION.LIGHT") %in% all.names(expr[[3L]])),guards)
  sampling.guards <- Filter(function(expr) "HARVEST" %in% all.names(expr[[3L]]) &&
    any(c("R.RNFL.GCL.POSITION.DARK","R.RNFL.GCL.POSITION.LIGHT") %in% all.names(expr[[3L]])),guards)
  same(2L,length(correction.guards),"two GCL correction guards")
  same(2L,length(smoothing.guards),"two GCL smoothing guards")
  same(2L,length(sampling.guards),"two GCL-dependent sampling guards")
  for(guard in c(correction.guards,smoothing.guards,sampling.guards)) {
    # No source arrays or spline helpers exist: accidental execution must fail.
    eval(guard,list2env(list(deep_layers_only=TRUE),parent=baseenv()))
    checks <- checks+1L
    probe <- list2env(list(deep_layers_only=FALSE,ran=FALSE),parent=baseenv())
    guard[[3L]] <- quote(ran <- TRUE)
    eval(guard,probe)
    assert(probe$ran,"A full-mode work branch was disabled")
  }
  cleanup.start <- assignments(scripts$candidate,"optional_rnfl_objects")[1L]
  assert(is.finite(cleanup.start),"Optional RNFL cleanup is missing")
  for(present in list(character(0),c("FIVE","MODEL.5"),
                      c("DEVIATION.5","FIVE","GIVEBACK.5","MODEL.5","RESULT.5","z.DEVIATION.5"))) {
    environment <- new.env(parent=baseenv())
    environment$preserved <- 99L
    for(name in present) assign(name,1L,envir=environment)
    warnings <- character(0)
    withCallingHandlers({
      for(expr in scripts$candidate[cleanup.start:(cleanup.start+2L)]) eval(expr,environment)
    },warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")})
    same("preserved",ls(envir=environment,all.names=TRUE),"optional RNFL cleanup scope")
    assert(!length(warnings),"Skipping RNFL temporaries generated cleanup warnings")
  }

  names.borders <- c("VITREOUS.RETINA","RNFL.GCL","INL.IPL","ONL.OPL","OLM","RPE")
  # Exercise the complete final smoothing loops on three distinct slices.
  smoothing.loops <- function(expressions) Filter(function(expr) call.to(expr,"for") &&
    "smooth.spline" %in% all.names(expr) && "R.RNFL.GCL.POSITION.DARK" %in% all.names(expr) ||
    call.to(expr,"for") && "smooth.spline" %in% all.names(expr) &&
      "R.RNFL.GCL.POSITION.LIGHT" %in% all.names(expr),as.list(expressions))
  smooth.input <- list()
  for(channel in c("DARK","LIGHT")) for(i in seq_along(names.borders))
    smooth.input[[paste0("R.",names.borders[i],".POSITION.",channel)]] <-
      outer(seq_len(2252),seq_len(3),function(x,z) 80+i*45+sin(x/113)*2+z*.31)
  run.smoothing <- function(expressions,deep) {
    environment <- list2env(smooth.input,parent=globalenv())
    environment$deep_layers_only <- deep
    if(deep) for(channel in c("DARK","LIGHT"))
      environment[[paste0("R.RNFL.GCL.POSITION.",channel)]][,] <- NA_real_
    loops <- smoothing.loops(expressions)
    assert(length(loops)==2L,"Expected two complete final smoothing loops")
    for(expr in loops) eval(expr,environment)
    mget(names(smooth.input),envir=environment,inherits=FALSE)
  }
  smoothed.reference <- run.smoothing(scripts$main,FALSE)
  smoothed.full <- run.smoothing(scripts$candidate,FALSE)
  smoothed.deep <- run.smoothing(scripts$candidate,TRUE)
  for(name in names(smooth.input)) {
    same(smoothed.reference[[name]],smoothed.full[[name]],paste("full smoothing",name))
    if(grepl("RNFL.GCL",name,fixed=TRUE)) assert(all(is.na(smoothed.deep[[name]])),"Deep GCL smoothing filled placeholders") else
      same(smoothed.reference[[name]],smoothed.deep[[name]],paste("deep retained smoothing",name))
  }

  # Execute real main normalization, main profiles, foveal splines, foveal
  # sampling, foveal insertion and summary calculations on compact 3-slice data.
  # Only Analyze image writing and logging are stubbed; numerical code is real.
  input <- list(IMAGE.INDEX.DARK=c(2L,4L,9L),IMAGE.INDEX.LIGHT=c(2L,4L,9L),
                TO.PROCESS.DARK="DARK",TO.PROCESS.LIGHT="LIGHT",OUTDIR=sandbox)
  n <- 612L
  for(channel in c("DARK","LIGHT")) {
    image <- array(sin(seq_len(n*461L*3L)/117)*20+seq_len(n*461L*3L)%%173,
                   dim=c(n,461L,3L))
    if(channel=="DARK") image <- image+.125
    input[[paste0("FLATTENED.",channel,".RETINA.RRC")]] <- image
    true <- array(NA_real_,dim=c(n,6L,3L))
    for(i in seq_along(names.borders)) {
      border <- outer(seq_len(n),seq_len(3),function(x,z) 70+i*50+sin(x/91)*2+z*.173)
      input[[paste0("R.",names.borders[i],".POSITION.",channel)]] <- border
      true[,i,] <- border
    }
    true[25L,5L,2L] <- NA_real_
    input[[paste0("TRUE.BORDERS.",channel)]] <- true
    input[[paste0("APPARENT.ANGLES.FOR.",channel)]] <- matrix(c(2,4,9,.12,-.42,.06,-2.4,.3,1.4),nrow=3L)
  }
  input$FLATTENED.MARKERS.RRC <- array(seq_len(24),dim=c(4L,3L,2L))
  input$FIRST.GRAND.MEAN <- matrix(seq_len(12),nrow=4L)
  numerical.expressions <- function(expressions) {
    start <- assignments(expressions,"MAIN.DARK.OUTPUTS")[1L]
    cleanup <- which(vapply(expressions,function(expr) call.to(expr,"rm") &&
      ".step3.column.means" %in% all.names(expr),logical(1)))
    cleanup <- cleanup[cleanup>start][1L]
    angle <- assignments(expressions,"MAIN.DARK.OUTPUTS")[2L]
    c(as.list(expressions[start:(cleanup-1L)]),as.list(expressions[angle:(angle+7L)]))
  }
  run.numerical <- function(expressions,deep) {
    environment <- list2env(input,parent=globalenv())
    environment$deep_layers_only <- deep
    environment$dbg <- function(...) invisible(NULL)
    environment$f.write.analyze <- function(...) invisible(NULL)
    if(deep) for(channel in c("DARK","LIGHT")) {
      environment[[paste0("R.RNFL.GCL.POSITION.",channel)]][,] <- NA_real_
      environment[[paste0("TRUE.BORDERS.",channel)]][,2L,] <- NA_real_
    }
    for(expr in numerical.expressions(expressions)) tryCatch(eval(expr,environment),error=function(e)
      stop(paste(conditionMessage(e),"while evaluating",paste(deparse(expr),collapse=" ")),call.=FALSE))
    if(deep) environment$retained_main_depth_columns <- 35:90
    environment
  }
  reference <- run.numerical(scripts$main,FALSE)
  full <- run.numerical(scripts$candidate,FALSE)
  deep <- run.numerical(scripts$candidate,TRUE)
  validator.env <- new.env(parent=globalenv())
  define.functions(parse("release_tools/verify_step3_deep_layers.R"),validator.env)
  compare <- validator.env$compare_step3_deep_layers
  snapshot <- function(environment,target) {
    keep <- Filter(function(name) {value <- get(name,envir=environment,inherits=FALSE)
      !is.function(value) && (is.numeric(value) || is.logical(value) || is.matrix(value) ||
        is.array(value) || is.data.frame(value) || name %in% c("TO.PROCESS.DARK","TO.PROCESS.LIGHT"))
    },ls(envir=environment,all.names=TRUE))
    save(list=keep,file=target,envir=environment,version=2,compress=FALSE)
  }
  baseline.path <- file.path(sandbox,"baseline.RData")
  full.path <- file.path(sandbox,"full.RData")
  deep.path <- file.path(sandbox,"deep.RData")
  snapshot(reference,baseline.path); snapshot(full,full.path); snapshot(deep,deep.path)
  assert(compare(baseline.path,full.path,"full",check_exports=FALSE)>0L,"Full numerical comparison did not run")
  assert(compare(baseline.path,deep.path,"deep",check_exports=FALSE)>0L,"Deep numerical comparison did not run")
  # Validate failure detection independently of the production path. Changes in
  # later slices, masks, labels or core arrays must never pass a tolerant check.
  corruptions <- list(
    later.boundary=function(e) e$R.INL.IPL.POSITION.LIGHT[20L,3L] <- e$R.INL.IPL.POSITION.LIGHT[20L,3L]+1e-8,
    retained.sample=function(e) e$FLATTENED.DARK.RETINA.RRC.N[40L,50L,3L] <- 1e-8,
    omitted.sample=function(e) e$FLATTENED.LIGHT.RETINA.RRC.N[60L,1L,3L] <- 0,
    retained.profile=function(e) e$FLATTENED.LIGHT.RETINA.RRC.N.profiles[50L,3L] <- 0,
    omitted.profile=function(e) e$FLATTENED.DARK.RETINA.RRC.N.profiles[1L,2L] <- 0,
    depth.coordinates=function(e) e$FLATTENED.DARK.RETINA.RRC.N.profiles[1L,1L] <- 0,
    fovea.profile=function(e) e$FLATTENED.DARK.RETINA.RRC.N.fovea.profiles[1L,2L] <- 0,
    omitted.summary=function(e) e$MAIN.DARK.OUTPUTS[1L,"RNFL.thickness"] <- 0,
    gcl.placeholder=function(e) e$R.RNFL.GCL.POSITION.DARK[1L,3L] <- 0,
    marker.core=function(e) e$FLATTENED.MARKERS.RRC[1L] <- 999L,
    registration.core=function(e) e$FIRST.GRAND.MEAN[1L] <- 999L,
    mode=function(e) e$deep_layers_only <- FALSE,
    missing=function(e) rm(list="R.OLM.POSITION.LIGHT",envir=e))
  for(name in names(corruptions)) {
    changed <- new.env(parent=baseenv()); load(deep.path,envir=changed)
    corruptions[[name]](changed)
    target <- file.path(sandbox,"corrupted.RData")
    snapshot(changed,target)
    failure <- tryCatch({compare(baseline.path,target,"deep",check_exports=FALSE); NULL},error=conditionMessage)
    assert(!is.null(failure),paste("Validator accepted corruption:",name))
  }

  # Both output writers preserve all eight positions. Full mode is byte exact;
  # deep mode changes only the four documented measurement rows to missing.
  strip.writer <- function(writer) {body(writer)[[length(body(writer))]] <- quote(THICKNESS.EXPORT); writer}
  old.builder <- strip.writer(environments$main$write.thickness.table)
  new.builder <- strip.writer(environments$candidate$write.thickness.table)
  table.borders <- lapply(seq_len(6L),function(i)
    outer(seq_len(2851),seq_len(3),function(x,z) 430-(i-1)*43+sin(x/71)*3+z*.137))
  # Writer argument order is the reverse of boundary-array numbering.
  cases <- list(normal=table.borders)
  exceptional <- table.borders
  for(i in seq_along(exceptional)) exceptional[[i]][10:17,1L] <- c(NA,NaN,Inf,-Inf,1e300,-1e300,1e-100,-0)
  cases$exceptional <- exceptional
  for(case in names(cases)) for(channel in c("DARK","LIGHT")) for(mode in c(FALSE,TRUE)) {
    borders <- cases[[case]]
    expected <- do.call(old.builder,c(list(channel),borders))
    if(mode) expected[c(2L,6L,7L,8L),2:ncol(expected)] <- NA_real_
    environments$candidate$deep_layers_only <- mode
    actual <- do.call(new.builder,c(list(channel),borders))
    same(expected,actual,paste(case,channel,mode,"main eight-row table"))
    environments$candidate.output$deep_layers_only <- mode
    companion <- do.call(environments$candidate.output$make_thickness_export,borders)
    same(unname(as.matrix(expected)),unname(as.matrix(companion)),paste(case,channel,mode,"companion table"))
    expected.path <- file.path(sandbox,"expected-thickness.txt")
    write(as.matrix(expected),file=expected.path,ncolumns=8L,sep="\t")
    for(kind in c("candidate","candidate.output")) {
      directory <- file.path(sandbox,paste(case,channel,mode,kind,sep="-")); dir.create(directory)
      environment <- environments[[kind]]
      if(kind=="candidate") {environment$OUTDIR <- directory
        do.call(environment$write.thickness.table,c(list(channel),borders))
      } else environment$write_thickness_export(companion,channel,directory)
      target <- file.path(directory,paste0("_thickness_vs_distance_from_fovea_",channel,".txt"))
      same(raw.file(expected.path),raw.file(target),paste(case,channel,mode,kind,"thickness bytes"))
    }
  }
  # Four retained overlays in reduced mode; the original six in full mode.
  for(kind in c("candidate","candidate.output")) {
    environment <- environments[[kind]]
    plotter <- get(if(kind=="candidate") "save.tissue.border.plot" else "save.tissue.border.output",environment)
    environment$png <- function(...) invisible(NULL)
    environment$dev.off <- function(...) invisible(NULL)
    environment$image <- function(...) invisible(NULL)
    for(mode in c(FALSE,TRUE)) {
      environment$deep_layers_only <- mode
      lines <- list()
      environment$matlines <- function(x,y,col,...) lines[[length(lines)+1L]] <<- list(x=x,y=y,col=col)
      do.call(plotter,c(list("plot.png",array(0,dim=c(2L,2L,1L))),table.borders))
      same(if(mode) 4L else 6L,length(lines),paste(kind,mode,"overlay count"))
      for(i in seq_along(lines)) {
        same(seq(-100,2750,1),lines[[i]]$x,paste(kind,mode,i,"overlay distances"))
        same(431-table.borders[[i]][,1L],lines[[i]]$y,paste(kind,mode,i,"overlay coordinates"))
      }
    }
  }
  # Companion defaults must not inherit a stale mode from another environment.
  startup <- as.list(scripts$candidate.output)[1:2]
  parent <- list2env(list(deep_layers_only=TRUE),parent=baseenv())
  environment <- new.env(parent=parent)
  for(expr in startup) eval(expr,environment)
  same(FALSE,environment$deep_layers_only,"legacy companion mode default")
  for(value in list(NA,1,"1",logical(0),c(TRUE,FALSE))) {
    environment <- list2env(list(deep_layers_only=value),parent=baseenv())
    failure <- tryCatch({for(expr in startup) eval(expr,environment); NULL},error=conditionMessage)
    assert(!is.null(failure),"Companion accepted an invalid saved mode")
  }
  cat("PASS:",checks,"deep-layer functional regression checks on",R.version.string,"\n")
}
main()
