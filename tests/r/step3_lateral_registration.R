# Byte-exact checks of the actual Step 3 lateral-registration implementation.
# Run in a fresh R process from the repository root:
# Rscript.exe tests/r/step3_lateral_registration.R
# An optional first argument selects another version of the processing script.
# The processing script is parsed; only .step3.lateral is evaluated. No Analyze
# package, image inputs, or files from debug_outputs are required.
(function() {
  args <- commandArgs(trailingOnly=TRUE)
  script <- if(length(args)) args[1] else "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R"
  script <- normalizePath(script, mustWork=TRUE)
  helpers <- new.env(parent=baseenv())
  for(expr in parse(script, keep.source=FALSE)) {
    if(is.call(expr) && as.character(expr[[1]]) %in% c("<-", "=") &&
       length(expr) == 3L && is.symbol(expr[[2]]) &&
       identical(as.character(expr[[2]]), ".step3.lateral")) {
      eval(expr, helpers)
    }
  }
  if(!exists(".step3.lateral", envir=helpers, inherits=FALSE)) {
    stop("Missing .step3.lateral in the processing script")
  }
  actual <- helpers$.step3.lateral
  actual.score <- get("score.slice", envir=environment(actual), inherits=FALSE)
  checks <- 0L
  same.bytes <- function(expected, observed, label) {
    if(!identical(serialize(expected, NULL, version=2),
                  serialize(observed, NULL, version=2))) {
      stop(paste(label, "serialized bytes differ"))
    }
    checks <<- checks + 1L
  }
  legacy.score <- function(image, reference) {
    slide <- cbind(seq(-39,39,1), seq(-39,39,1))
    for(x in -39:39) {
      slide[x+40,2] <- stats::cor.test(reference,
        as.vector(image[(40+x):((nrow(image)-39)+x),]))$est
    }
    best <- which(slide[,2] == max(na.rm=TRUE,slide[,2]))[1]
    list(move=slide[best,1], slide=slide)
  }
  legacy.register <- function(volume, reference) {
    n.slices <- dim(volume)[3]
    moves <- cbind(seq(1,n.slices,1), seq(1,n.slices,1))
    for(z in 1:n.slices) {
      result <- legacy.score(volume[,,z], reference)
      moves[z,2] <- result$move
    }
    list(moves=moves, slide=result$slide)
  }
  check.no.globals <- function(expr, label) {
    before <- ls(envir=.GlobalEnv, all.names=TRUE)
    result <- force(expr)
    after <- ls(envir=.GlobalEnv, all.names=TRUE)
    same.bytes(before, after, paste(label, "global variable names"))
    result
  }

  # Keep tests isolated from any existing user seed and restore it on exit.
  had.seed <- exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE)
  if(had.seed) initial.seed <- get(".Random.seed", envir=.GlobalEnv)
  on.exit({
    if(had.seed) {
      assign(".Random.seed", initial.seed, envir=.GlobalEnv)
    } else if(exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE)) {
      rm(list=".Random.seed", envir=.GlobalEnv)
    }
  }, add=TRUE)
  if(had.seed) rm(list=".Random.seed", envir=.GlobalEnv)
  if("parallel" %in% loadedNamespaces()) {
    stop("Run this test in a fresh Rscript process to check first-load RNG behavior")
  }

  # Registration must not emit the processing script's legacy image/plot
  # artifacts. No scratch folder or generated fixtures are needed by this test.
  artifact.pattern <- "^(Rplots[.]pdf|_flat-normed_|_done_|DARK__and__LIGHT__flat[.]RData)"
  before.artifacts <- list.files(pattern=artifact.pattern, all.files=TRUE)

  deterministic <- array(seq_len(130*10*2), c(130,10,2))
  reference <- as.vector(deterministic[40:91,,1])
  expected <- legacy.register(deterministic, reference)
  observed <- check.no.globals(actual(deterministic, reference, workers=1L),
                              "one-worker registration")
  same.bytes(expected, observed, "one-worker registration without a seed")
  stopifnot(!exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE))
  observed <- check.no.globals(actual(deterministic, reference, workers=2L),
                              "first parallel registration")
  same.bytes(expected, observed, "first parallel registration without a seed")
  stopifnot(!exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE))

  set.seed(331)
  volume <- array(rnorm(130*10*5), c(130,10,5))
  volume[70,5,2] <- NA_real_
  volume[80,4,3] <- NaN
  grand.mean <- volume[,,1] + volume[,,2]
  reference <- as.vector(as.matrix(grand.mean[40:91,]))
  for(z in 1:5) {
    same.bytes(legacy.score(volume[,,z], reference),
               actual.score(volume[,,z], reference),
               paste("all 79 scores and chosen shift for slice",z))
  }
  expected <- legacy.register(volume, reference)
  saved.seed <- .Random.seed
  sequential <- check.no.globals(actual(volume, reference, workers=1L),
                                "sequential registration")
  same.bytes(saved.seed, .Random.seed, "sequential RNG state")
  # Five slices exercise a final batch containing fewer slices than workers.
  parallel <- check.no.globals(actual(volume, reference, workers=2L),
                              "two-worker registration")
  same.bytes(saved.seed, .Random.seed, "parallel RNG state")
  same.bytes(expected, sequential, "legacy versus sequential complete result")
  same.bytes(expected, parallel, "legacy versus parallel complete result")
  same.bytes(sequential, parallel, "one versus two workers")

  tied.image <- matrix(rep(1:10, each=130), nrow=130)
  tied.reference <- as.vector(tied.image[40:91,])
  tied <- actual.score(tied.image, tied.reference)
  same.bytes(legacy.score(tied.image, tied.reference), tied, "all-score tie")
  stopifnot(identical(tied$move, -39))
  for(image in list(matrix(1,130,10), replace(volume[,,1], 1L, Inf))) {
    same.bytes(suppressWarnings(legacy.score(image, reference)),
               suppressWarnings(actual.score(image, reference)),
               "constant/infinite candidate scores")
  }

  invalid.workers <- list(0L, -1L, 1.5, NA_real_, Inf, numeric(0), c(1L,2L))
  for(workers in invalid.workers) {
    error <- tryCatch({ actual(volume, reference, workers); NULL },
                      error=function(e) conditionMessage(e))
    stopifnot(identical(error, "AIDAS_STEP3_WORKERS must be a positive integer"))
  }
  # Worker errors must also restore the existing saved seed.
  error <- tryCatch({ actual(volume, numeric(0), workers=2L); NULL },
                    error=function(e) conditionMessage(e))
  stopifnot(is.character(error), length(error) == 1L)
  same.bytes(saved.seed, .Random.seed, "failed parallel registration RNG state")

  same.bytes(before.artifacts,
             list.files(pattern=artifact.pattern, all.files=TRUE),
             "legacy processing artifact names")
  cat("PASS:", checks,
      "byte-exact lateral-registration checks; all 79 scores, ties, missing values, workers, and RNG preservation\n")
})()
