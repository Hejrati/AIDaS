# Byte-exact regression checks for the Step 3 performance helpers.
# Run with the same R runtime as the processing script, for example:
# Rscript.exe tests/r/step3_optimization_kernels.R
# Only helper/callback definitions are evaluated; image processing is not run.
args <- commandArgs(trailingOnly=TRUE)
script <- if(length(args)) args[1] else "RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R"
required <- c(".step3.sample.line", ".step3.column.means", ".step3.pearson",
              ".step3.nearest.depth", ".step3.depth.groups")
helpers <- new.env(parent=globalenv())
for(expr in parse(script)) {
  if(is.call(expr) && as.character(expr[[1]]) %in% c("<-", "=") &&
     length(expr) == 3L && is.symbol(expr[[2]]) &&
     as.character(expr[[2]]) %in% c(required,"WHICH.INDEX") && is.call(expr[[3]]) &&
     identical(expr[[3]][[1]], as.name("function"))) eval(expr, helpers)
}
missing <- required[!vapply(required, exists, logical(1), envir=helpers, inherits=FALSE)]
if(length(missing)) stop(paste("Missing performance helpers:", paste(missing, collapse=", ")))

checks <- 0L
capture <- function(expr) {
  warnings <- character(0)
  result <- tryCatch(withCallingHandlers(
    list(value=force(expr)),
    warning=function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }), error=function(e) list(error=conditionMessage(e)))
  result$warnings <- warnings
  result
}
assert.same <- function(expected, actual, label, compare.warnings=TRUE) {
  if(!identical(expected$error, actual$error))
    stop(paste(label, "error differs:", expected$error, "versus", actual$error))
  if(compare.warnings && !identical(expected$warnings, actual$warnings))
    stop(paste(label, "warnings differ:", paste(expected$warnings, collapse="; "),
               "versus", paste(actual$warnings, collapse="; ")))
  if(is.null(expected$error)) {
    expected.raw <- serialize(as.vector(expected$value), NULL, version=2)
    actual.raw <- serialize(as.vector(actual$value), NULL, version=2)
    if(!identical(expected.raw, actual.raw))
      stop(paste(label, "serialized bytes differ; types", typeof(expected$value),
                 "versus", typeof(actual$value)))
  }
  checks <<- checks + 1L
}

legacy.sample <- function(unwrapped, line, upper.x, upper.y) {
  get.intensity <- function(x)
    if((x[1]>=1) & (x[1]<=upper.x) & (x[2]>=1) & (x[2]<=upper.y))
      unwrapped[((x[1]-1)*upper.y)+x[2]] else NA
  as.vector(tapply(line[,c(2,1)],
    as.factor(cbind(seq(1,501,1), seq(1,501,1))), get.intensity))
}
set.seed(331)
upper.x <- 13L
upper.y <- 17L
sample.inputs <- list(
  as.vector(matrix(seq_len(upper.x*upper.y), nrow=upper.y)),
  rnorm(upper.x*upper.y),
  rep(NA_real_, upper.x*upper.y),
  rep(TRUE, upper.x*upper.y))
sample.lines <- list(
  cbind(rep(1,501), rep(1,501)),
  cbind(rep(upper.y,501), rep(upper.x,501)),
  cbind(sample(-2:(upper.y+2),501,replace=TRUE), sample(-2:(upper.x+2),501,replace=TRUE)),
  cbind(rep(0,501), rep(1,501)),
  cbind(rep(upper.y+1,501), rep(upper.x+1,501)),
  cbind(rep(Inf,501), rep(-Inf,501)))
for(i in seq_along(sample.inputs)) for(j in seq_along(sample.lines)) {
  expected <- capture(legacy.sample(sample.inputs[[i]], sample.lines[[j]], upper.x, upper.y))
  actual <- capture(helpers$.step3.sample.line(sample.inputs[[i]], sample.lines[[j]], upper.x, upper.y))
  assert.same(expected, actual, paste("sample input",i,"line",j))
  if(is.null(actual$error) && length(actual$value) != 501L) stop("Sample F must retain 501 entries")
}
for(coordinates in list(c(NA,1), c(1,NA), c(NaN,1), c(Inf,NA), c(NA,0), c(NA,NA))) {
  line <- cbind(rep(1,501), rep(1,501))
  line[250,] <- coordinates
  assert.same(capture(legacy.sample(sample.inputs[[2]],line,upper.x,upper.y)),
              capture(helpers$.step3.sample.line(sample.inputs[[2]],line,upper.x,upper.y)),
              paste("sample coordinates",paste(coordinates,collapse=",")))
}

legacy.means <- function(value) {
  index <- as.factor(matrix(rep(seq_len(ncol(value)), each=nrow(value)), nrow=nrow(value)))
  as.vector(tapply(value, index, function(x) mean(x, na.rm=TRUE)))
}
mean.inputs <- list(
  matrix(c(1,2,3,4,5,6),nrow=3),
  matrix(c(NA,NaN,NA,NaN,NA,NaN),nrow=3),
  matrix(c(Inf,1,2,-Inf,1,2,Inf,-Inf,1),nrow=3),
  matrix(c(1e300,1,-1e300,1e300,-1e300,1,0,-0,1),nrow=3),
  matrix(seq_len(40*461),nrow=40),
  matrix(rnorm(400*500),nrow=400),
  matrix(c(1,NA,NaN,Inf,-Inf,0,-0),ncol=1))
for(i in 1:20) {
  value <- matrix(rnorm(40*10),nrow=40)
  value[sample(length(value),30)] <- c(NA_real_,NaN,Inf,-Inf,NA_real_)
  mean.inputs[[length(mean.inputs)+1L]] <- value
}
for(i in seq_along(mean.inputs))
  assert.same(capture(legacy.means(mean.inputs[[i]])),
              capture(helpers$.step3.column.means(mean.inputs[[i]])),paste("means",i))

cor.inputs <- list(
  list(1:10,10:1),
  list(c(1,1,2,2,3,3),c(1,2,3,3,2,1)),
  list(c(1,NA,3,4,5),c(5,4,3,2,1)),
  list(c(1,2,NaN,4,NA,6),c(6,5,4,NA,2,1)),
  list(rep(1,5),1:5),
  list(rep(1,5),rep(1,5)),
  list(1:3,c(1,3,2)),
  list(1:2,2:1),
  list(c(NA,NA,1,2),c(1,2,3,4)),
  list(numeric(0),numeric(0)),
  list(c(1,Inf,3,4),c(1,2,3,4)),
  list(c(1,-Inf,3,4),c(1,2,3,4)),
  list(c(1,2,3),c(1,2)),
  list(c("1","2","3"),1:3),
  list(1:3,c("1","2","3")))
for(i in 1:50) cor.inputs[[length(cor.inputs)+1L]] <- list(rnorm(100),rnorm(100))
for(i in seq_along(cor.inputs)) {
  pair <- cor.inputs[[i]]
  assert.same(capture(as.vector(cor.test(pair[[1]],pair[[2]])$est)),
              capture(helpers$.step3.pearson(pair[[1]],pair[[2]])),paste("pearson",i))
}
# Verify the selected registration shift, including equal maxima and negative correlations.
for(i in 1:10) {
  reference <- rnorm(130)
  moving <- c(rep(0,i),reference,rep(0,20-i))
  candidates <- -10:10
  expected <- vapply(candidates,function(s) as.vector(cor.test(reference,moving[(11+s):(140+s)])$est),numeric(1))
  actual <- vapply(candidates,function(s) helpers$.step3.pearson(reference,moving[(11+s):(140+s)]),numeric(1))
  assert.same(capture(expected),capture(actual),paste("registration scores",i))
  assert.same(capture(candidates[which(expected==max(expected))[1]]),
              capture(candidates[which(actual==max(actual))[1]]),paste("registration shift",i))
}

depthstrip <- seq(1,461,1)
helpers$depthstrip <- depthstrip
legacy.depth <- function(x) which(abs(depthstrip-x)==min(na.rm=TRUE,abs(depthstrip-x)))[1]
depth.inputs <- c(seq(-10,480,0.25),1:461+0.5,1:461-0.5,
                  NA_real_,NaN,Inf,-Inf,1e308,-1e308,
                  2^52,2^53,2^54,-2^52,-2^53,-2^54,
                  1.5-.Machine$double.eps,1.5+.Machine$double.eps,
                  460.5-.Machine$double.eps*512,460.5+.Machine$double.eps*512)
expected <- capture(vapply(depth.inputs,legacy.depth,integer(1)))
actual <- capture(helpers$.step3.nearest.depth(depth.inputs))
assert.same(expected,actual,"nearest depth",compare.warnings=FALSE)

factor.inputs <- list(
  list(c(3.5,10.5,400.5),factor(c(4,5,6),levels=1:10)),
  list(c(3.5,10.5,400.5),factor(c("c","a","b"),levels=c("d","c","b","a","e"))),
  list(c(NA_real_,Inf,NaN),factor(c(4,5,6),levels=1:10)),
  list(c(3,10,400),factor(c(4,NA,6),levels=1:10)),
  list(c(3,10,400),factor(c(4,4,6),levels=1:10)),
  list(seq(1,461,length.out=18),as.factor(seq(1,18,1))),
  list(seq(1,461,length.out=16),as.factor(seq(1,16,1))))
for(i in seq_along(factor.inputs)) {
  input <- factor.inputs[[i]]
  assert.same(capture(as.vector(tapply(input[[1]],input[[2]],legacy.depth))),
              capture(helpers$.step3.depth.groups(input[[1]],input[[2]])),
              paste("depth factors",i),compare.warnings=FALSE)
}
# The cached spline prediction must preserve the entire matrix, attributes included.
spline.expression <- NULL
for(expr in parse(script)) {
  if(is.call(expr) && is.symbol(expr[[1]]) &&
     as.character(expr[[1]]) %in% c("<-","=") &&
     identical(expr[[2]],as.name("RPE.spline"))) {
    spline.expression <- expr
    break
  }
}
if(is.null(spline.expression)) stop("Initial RPE.spline assignment not found")
spline.environment <- new.env(parent=globalenv())
spline.environment$R <- matrix(0,nrow=37,ncol=25)
spline.x <- seq(1,37,1)
spline.environment$RPE.sp <- smooth.spline(spline.x,
  sin(spline.x/5)*3 + spline.x^2/50 + cos(spline.x/3),df=10)
spline.expected <- with(spline.environment,
  cbind(as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$x),
        as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02))$y),
        as.numeric(predict(RPE.sp,seq(0,dim(R)[1],0.02),deriv=1)$y)))
spline.actual <- eval(spline.expression,spline.environment)
if(!identical(serialize(spline.expected,NULL,version=2),
              serialize(spline.actual,NULL,version=2)))
  stop("Cached spline prediction differs in serialized bytes or attributes")
if(length(setdiff(ls(spline.environment),c("R","RPE.sp","RPE.spline"))))
  stop("Cached spline prediction leaked temporary variables")
checks <- checks + 1L
# Keep temporary optimization objects out of the legacy save.image outputs.
# Inspect top-level assignment/removal order without evaluating image processing.
active.helpers <- character(0)
workspace.saves <- 0L
for(expr in parse(script)) {
  if(!is.call(expr) || !is.symbol(expr[[1]])) next
  operation <- as.character(expr[[1]])
  if(operation %in% c("<-", "=") && is.symbol(expr[[2]])) {
    name <- as.character(expr[[2]])
    if(grepl("^\\.step3\\.",name)) active.helpers <- union(active.helpers,name)
  }
  if(operation == "rm") {
    removed <- vapply(as.list(expr)[-1],function(x) {
      if(is.symbol(x)) as.character(x) else ""
    },character(1))
    active.helpers <- setdiff(active.helpers,removed)
  }
  if(operation == "save.image") {
    if(length(active.helpers))
      stop(paste("Optimization objects remain at workspace save:",paste(active.helpers,collapse=", ")))
    workspace.saves <- workspace.saves + 1L
    checks <- checks + 1L
  }
}
if(workspace.saves != 2L) stop("Expected to check both legacy workspace saves")
cat("PASS:",checks-workspace.saves,"byte-exact Step 3 kernel comparisons and",
    workspace.saves,"workspace cleanup checks on",R.version.string,"\n")
