# Exercise actual startup and rendering code without running image registration.
# Rscript.exe --vanilla tests/r/step3_runtime_artifacts.R
main <- function() {
  mains <- c("RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R",
             "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized_deep_layers.R")
  companions <- c(
    "more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited_optimized.R",
    "more_outputs_afterRAW_OCT_PROCESSING_2022_11NOV_27_WSU_noHypoDenseBand_EA edited_optimized_deep_layers.R")
  mains <- normalizePath(mains, mustWork=TRUE)
  companions <- normalizePath(companions, mustWork=TRUE)
  previous.directory <- getwd()
  previous.device <- getOption("device")
  previous.environment <- Sys.getenv(c("AIDAS_STEP3_INPUT_DIR", "AIDAS_STEP3_OUTPUT_DIR",
    "AIDAS_STEP3_DEEP_ONLY", "AIDAS_STEP3_RASTER", "AIDAS_STEP3_DEFER_BORDER_PLOTS"),
    unset=NA_character_)
  sandbox <- tempfile("step3-runtime-artifacts-")
  dir.create(sandbox)
  on.exit({
    graphics.off()
    options(device=previous.device)
    setwd(previous.directory)
    for(name in names(previous.environment)) {
      value <- previous.environment[[name]]
      if(is.na(value)) Sys.unsetenv(name) else do.call(Sys.setenv, setNames(list(value), name))
    }
    unlink(sandbox, recursive=TRUE)
  }, add=TRUE)
  Sys.unsetenv(names(previous.environment))
  checks <- 0L
  assert <- function(value, label) {
    if(!isTRUE(value)) stop(label, call.=FALSE)
    checks <<- checks+1L
  }
  call.to <- function(expr, name) is.call(expr) && identical(expr[[1L]], as.name(name))
  assigned <- function(expr, name) is.call(expr) &&
    as.character(expr[[1L]]) %in% c("<-", "=") && length(expr)==3L &&
    identical(expr[[2L]], as.name(name))
  no.junk <- function(directory, label) assert(
    !any(tolower(list.files(directory, all.files=TRUE, recursive=TRUE,
                           include.dirs=TRUE)) %in% c("step3_r_arrays", "rplots.pdf", "r-plots.pdf")), label)
  border <- matrix(400, nrow=2851L)
  flattened <- array(0, dim=c(2851L, 461L, 1L))
  flattened[,200:300,] <- 200

  for(i in seq_along(mains)) {
    expressions <- parse(mains[[i]])
    workspace <- file.path(sandbox, paste0("main-", i))
    dir.create(workspace)
    Sys.setenv(AIDAS_STEP3_INPUT_DIR=workspace, AIDAS_STEP3_OUTPUT_DIR=workspace)
    environment <- new.env(parent=globalenv())
    first.library <- which(vapply(expressions, call.to, logical(1), name="library"))[[1L]]
    for(expr in expressions[seq_len(first.library-1L)]) eval(expr, environment)
    no.junk(workspace, "Main startup created redundant artifacts")
    assert(!exists("write.python.array", environment, inherits=FALSE), "Dead array sidecar writer remains")
    # A library or future diagnostic can request an implicit device; it must
    # never write Rplots.pdf, while explicit PNG exports must still work.
    plot(1:2, 1:2)
    graphics.off()
    no.junk(workspace, "Implicit graphics created a PDF")

    plotter <- environment$save.tissue.border.plot
    image.calls <- list()
    environment$image <- function(...) { image.calls[[length(image.calls)+1L]] <<- list(...); invisible(NULL) }
    environment$png <- function(...) invisible(NULL)
    environment$dev.off <- function(...) invisible(NULL)
    environment$matlines <- function(...) invisible(NULL)
    for(setting in c("", "1")) {
      Sys.setenv(AIDAS_STEP3_RASTER=setting)
      plotter("stub.png", flattened, border, border, border, border, border, border)
    }
    assert(is.null(image.calls[[1L]]$useRaster), "Default renderer changed")
    assert(identical(image.calls[[2L]]$useRaster, TRUE), "Explicit raster renderer was not selected")
    rm(list=c("image", "png", "dev.off", "matlines"), envir=environment)
    plotter("preview.png", flattened, border, border, border, border, border, border)
    assert(identical(readBin("preview.png", "raw", n=8L),
      as.raw(c(137,80,78,71,13,10,26,10))), "Explicit tissue-border PNG was not produced")
    no.junk(workspace, "Explicit PNG produced redundant artifacts")

    guards <- Filter(function(expr) call.to(expr, "if") &&
      "Sys.getenv" %in% all.names(expr[[2L]]) &&
      grepl("AIDAS_STEP3_DEFER_BORDER_PLOTS", paste(deparse(expr[[2L]]), collapse=""), fixed=TRUE),
      as.list(expressions))
    assert(length(guards)==1L, "Expected one main preview deferral guard")
    calls <- 0L
    environment$save.tissue.border.plot <- function(...) { calls <<- calls+1L; invisible(NULL) }
    # Every array is passed lazily, so a deferred guard must avoid touching
    # the entire scientific workspace, and standalone mode must render twice.
    for(setting in c("", "0", "1", "true")) {
      Sys.setenv(AIDAS_STEP3_DEFER_BORDER_PLOTS=setting)
      calls <- 0L
      before <- ls(environment, all.names=TRUE)
      eval(guards[[1L]], environment)
      assert(identical(calls, if(setting=="1") 0L else 2L), "Main preview deferral changed standalone behavior")
      assert(identical(before, ls(environment, all.names=TRUE)), "Preview guard mutated workspace objects")
    }
  }
  # Check each companion's actual graphics option independently of the main.
  for(i in seq_along(companions)) {
    expressions <- parse(companions[[i]])
    device.options <- Filter(function(expr) call.to(expr, "options") &&
      "device" %in% names(as.list(expr)), as.list(expressions))
    assert(length(device.options)==1L, "Companion lacks a safe default graphics device")
    workspace <- file.path(sandbox, paste0("companion-", i))
    dir.create(workspace)
    setwd(workspace)
    eval(device.options[[1L]])
    plot(1:2, 1:2)
    graphics.off()
    no.junk(workspace, "Companion implicit graphics created a PDF")
  }
  cat("PASS:", checks, "Step 3 runtime artifact and preview checks on", R.version.string, "\n")
}
main()
