#' @title CDRobject Class
#'
#' @description S4 container for a TCR/BCR repertoire analysis, designed in the
#' spirit of \code{Seurat}. A single object carries clones, six-concept
#' features, the concept-bottleneck embedding, classification results and
#' biomarkers through the whole pipeline.
#'
#' @slot meta Sample-level metadata \code{data.frame} (sample_id, group, chain,
#'   reads, ...).
#' @slot clones Clone-level table \code{data.frame} (sample_id, cdr3_aa,
#'   v_gene, j_gene, chain, count, freq).
#' @slot features Sample-by-feature numeric \code{matrix} (six concept axes).
#' @slot feature_modules Named \code{list} mapping each concept module to its
#'   column indices in \code{features}.
#' @slot embedding Sample-by-dim numeric \code{matrix} (concept-bottleneck
#'   embedding).
#' @slot reduction Sample-by-2 numeric \code{matrix} (2-D projection).
#' @slot classification Named \code{list} with classifier fit, predictions and
#'   SHAP values.
#' @slot markers Named \code{list} with statistical-level and sequence-level
#'   markers.
#' @slot misc Named \code{list} for auxiliary data.
#'
#' @exportClass CDRobject
#' @export CDRobject
#' @import methods
CDRobject <- setClass(
  "CDRobject",
  slots = c(
    meta            = "data.frame",
    clones          = "data.frame",
    features        = "matrix",
    feature_modules = "list",
    embedding       = "matrix",
    reduction       = "matrix",
    classification  = "list",
    markers         = "list",
    misc            = "list"
  ),
  prototype = list(
    meta            = data.frame(),
    clones          = data.frame(),
    features        = matrix(nrow = 0, ncol = 0),
    feature_modules = list(),
    embedding       = matrix(nrow = 0, ncol = 0),
    reduction       = matrix(nrow = 0, ncol = 0),
    classification  = list(),
    markers         = list(),
    misc            = list()
  )
)

setValidity("CDRobject", function(object) {
  ok <- TRUE
  msg <- character()
  if (nrow(object@meta) > 0 && nrow(object@features) > 0) {
    if (nrow(object@meta) != nrow(object@features)) {
      ok <- FALSE
      msg <- "nrow(meta) must equal nrow(features)"
    }
  }
  if (ok) TRUE else msg
})

#' @export
setMethod("show", "CDRobject", function(object) {
  cat("An object of class CDRobject\n")
  cat("Samples:", nrow(object@meta), " | Clones:", nrow(object@clones), "\n")
  cat("Features:", ncol(object@features), " | Modules:",
      length(object@feature_modules), "\n")
  if (ncol(object@reduction) > 0)
    cat("Reduction:", paste(colnames(object@reduction), collapse = ", "), "\n")
  if (length(object@classification) > 0)
    cat("Classification: fitted (", object@classification$method, ")\n", sep = "")
  if (length(object@markers) > 0)
    cat("Markers: stat-level", nrow(object@markers$statistical),
        "| seq-level", nrow(object@markers$sequence), "\n")
  invisible(object)
})

#' @export
print.CDRobject <- function(x, ...) show(x)

setGeneric("dim")
setMethod("dim", "CDRobject", function(x)
  c(Samples = nrow(x@meta), Clones = nrow(x@clones)))
