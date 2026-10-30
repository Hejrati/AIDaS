# Exercise the real bundled main-script export paths with the app's LIGHT-only
# environment. Numerical processing is covered by the existing R regressions.
main <- function() {
  paths <- c(
    "RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R",
    "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R",
    "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized_deep_layers.R"
  )
  previous.mode <- Sys.getenv("AIDAS_LIGHT_ONLY", unset=NA_character_)
  sandbox <- tempfile("step3-light-only-main-")
  dir.create(sandbox)
  on.exit({
    unlink(sandbox, recursive=TRUE)
    if(is.na(previous.mode)) Sys.unsetenv("AIDAS_LIGHT_ONLY") else
      Sys.setenv(AIDAS_LIGHT_ONLY=previous.mode)
  }, add=TRUE)
  checks <- 0L
  assert <- function(condition, label) {
    if(!isTRUE(condition)) stop(label, call.=FALSE)
    checks <<- checks + 1L
  }
  call.to <- function(expr, name) {
    is.call(expr) && identical(expr[[1L]], as.name(name))
  }
  assigned <- function(expr, name=NULL) {
    valid <- is.call(expr) && is.symbol(expr[[1L]]) &&
      as.character(expr[[1L]]) %in% c("<-", "=") && length(expr)==3L
    valid && (is.null(name) || identical(expr[[2L]], as.name(name)))
  }
  collect <- function(expr, name) {
    if(missing(expr)) return(list())
    found <- if(call.to(expr, name)) list(expr) else list()
    if(is.call(expr) || is.expression(expr)) for(child in as.list(expr))
      found <- c(found, collect(child, name))
    found
  }
  library(AnalyzeFMRI)

  for(index in seq_along(paths)) {
    expressions <- parse(paths[index])
    environment <- new.env(parent=globalenv())
    for(expr in expressions) if(assigned(expr) && call.to(expr[[3L]], "function"))
      eval(expr, environment)
    mode <- Filter(function(expr) assigned(expr, ".LIGHT.ONLY"), as.list(expressions))
    assert(length(mode)==1L, paste(paths[index], "LIGHT-only startup setting missing"))
    Sys.unsetenv("AIDAS_LIGHT_ONLY")
    eval(mode[[1L]], environment)
    assert(identical(environment$.LIGHT.ONLY, FALSE), "Standalone default changed")
    Sys.setenv(AIDAS_LIGHT_ONLY="1")
    eval(mode[[1L]], environment)
    assert(identical(environment$.LIGHT.ONLY, TRUE), "App LIGHT-only mode was ignored")

    environment$OUTDIR <- file.path(sandbox, as.character(index))
    dir.create(environment$OUTDIR)
    environment$TO.PROCESS.DARK <- "DARK"
    environment$TO.PROCESS.LIGHT <- "subject_LIGHT"
    environment$REFERENCE.DARK <- "DARK_MARKED"
    environment$REFERENCE.LIGHT <- "subject_LIGHT_MARKED"
    environment$deep_layers_only <- index==3L
    # Rasterization keeps the full-sized real PNG test inexpensive; the plotting
    # and file-writing code under test still comes from the production script.
    environment$image <- function(..., useRaster=TRUE)
      graphics::image(..., useRaster=useRaster)
    environment$FLATTENED.LIGHT.RETINA.RRC <- array(0, dim=c(2851L,461L,2L))
    borders <- c("RPE", "OLM", "ONL.OPL", "INL.IPL", "RNFL.GCL", "VITREOUS.RETINA")
    for(border in borders)
      environment[[paste0("R.",border,".POSITION.LIGHT")]] <-
        matrix(100, nrow=2851L, ncol=2L)

    # DARK arrays are deliberately absent: suppressed exports must return before
    # evaluating array arguments or preparing a second Analyze volume.
    guards <- Filter(function(expr) call.to(expr, "if") &&
      ".LIGHT.ONLY" %in% all.names(expr) && "f.write.analyze" %in% all.names(expr),
      as.list(expressions))
    assert(length(guards)==2L, "Expected flat and normalized DARK export guards")
    for(expr in guards) eval(expr, environment)
    light.analyze <- Filter(function(expr) call.to(expr, "f.write.analyze"),
                            as.list(expressions))
    assert(length(light.analyze)==2L, "Expected two LIGHT Analyze exports")
    for(expr in light.analyze) {
      environment$EXPORT <- array(seq_len(12L), dim=c(2L,3L,2L))
      eval(expr, environment)
    }
    for(name in c("save.tissue.border.plot", "write.thickness.table"))
      for(expr in collect(expressions, name)) eval(expr, environment)
    # Evaluate all three real vertex-device paths, including the fallback path.
    for(expr in collect(expressions, "png")) {
      if(!"REFERENCE.DARK" %in% all.names(expr)) next
      eval(expr, environment)
      plot(1:3)
      dev.off()
    }
    for(expr in Filter(function(expr) call.to(expr, "save.image"),
                       as.list(expressions))) eval(expr, environment)

    optional.arrays <- exists("write.python.array", envir=environment, inherits=FALSE)
    if(optional.arrays) {
      environment$PYEXPORTDIR <- file.path(environment$OUTDIR, "arrays")
      dir.create(environment$PYEXPORTDIR)
      assert(identical(environment$write.python.array("FLATTENED_DARK_RETINA_RRC", 1:3), FALSE),
             "Optional DARK binary export was not suppressed")
      assert(identical(environment$write.python.array("FLATTENED_LIGHT_RETINA_RRC", 1:3), TRUE),
             "Optional LIGHT binary export was suppressed")
    }

    # Run the actual final-profile preparation and writer expressions as one
    # block, using distinct nonempty fixtures for each legacy paired channel.
    for(channel in c("DARK", "LIGHT")) {
      offset <- if(channel=="LIGHT") 1000 else 0
      environment[[paste0("MAIN.",channel,".OUTPUTS")]] <- matrix(1:10 + offset, nrow=2L)
      environment[[paste0("MAIN.",channel,".OUTPUTS.fovea")]] <- matrix(1:6 + offset, nrow=2L)
      environment[[paste0("FLATTENED.",channel,".RETINA.RRC.N.profiles")]] <-
        matrix(1:270 + offset, nrow=90L)
      environment[[paste0("FLATTENED.",channel,".RETINA.RRC.N.fovea.profiles")]] <-
        matrix(1:102 + offset, nrow=34L)
    }
    profile.start <- which(vapply(expressions, function(expr) assigned(expr, "EXPORT") &&
      "MAIN.DARK.OUTPUTS" %in% all.names(expr), logical(1)))[1L]
    profile.end <- which(vapply(expressions, function(expr)
      assigned(expr, "write.thickness.table"), logical(1)))[1L] - 1L
    assert(is.finite(profile.start) && is.finite(profile.end), "Profile export block missing")
    for(expr in expressions[profile.start:profile.end]) eval(expr, environment)

    expected <- c(
      "_flat_LIGHT.RData", "_done_LIGHT.RData",
      "LIGHT_MARKED_find_vertex.png", "LIGHT_MARKED_vertex.png",
      "_flat_subject_LIGHT.hdr", "_flat_subject_LIGHT.img",
      "_flat-normed_subject_LIGHT.hdr", "_flat-normed_subject_LIGHT.img",
      "_tissueBorders__subject_LIGHT.png",
      "_thickness_vs_distance_from_fovea_subject_LIGHT.txt",
      "_light_profiles_subject_LIGHT.txt", "_fovea_light_profiles_subject_LIGHT.txt"
    )
    if(optional.arrays) expected <- c(expected,
      "arrays/FLATTENED_LIGHT_RETINA_RRC.bin", "arrays/FLATTENED_LIGHT_RETINA_RRC.shape")
    files <- list.files(environment$OUTDIR, recursive=TRUE)
    assert(!any(grepl("dark", basename(files), ignore.case=TRUE)),
           paste(paths[index], "generated a DARK filename"))
    assert(identical(sort(files), sort(expected)),
           paste(paths[index], "LIGHT-only export inventory changed:", paste(files, collapse=", ")))
    assert(all(file.info(file.path(environment$OUTDIR, files))$size > 0),
           paste(paths[index], "generated an empty output"))
    for(filename in c("_light_profiles_subject_LIGHT.txt", "_fovea_light_profiles_subject_LIGHT.txt")) {
      profile <- read.table(file.path(environment$OUTDIR, filename), stringsAsFactors=FALSE)
      assert(identical(as.numeric(unlist(profile[1L,2:3], use.names=FALSE)), c(1001,1002)),
             paste(filename, "does not contain the LIGHT measurements"))
    }
    for(filename in c("_flat_LIGHT.RData", "_done_LIGHT.RData")) {
      restored <- new.env(parent=baseenv())
      assert(length(load(file.path(environment$OUTDIR, filename), envir=restored)) > 0L,
             paste(filename, "is not a readable R workspace"))
    }
    cat("PASS:", paths[index], length(files), "LIGHT-only artifacts; no DARK files\n")
  }
  cat("PASS:", checks, "LIGHT-only main export regression checks on", R.version.string, "\n")
}
main()
