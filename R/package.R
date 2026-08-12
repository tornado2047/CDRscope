#' CDRscope: Interpretable TCR/BCR Repertoire Analysis
#'
#' An interpretable analysis framework for TCR/BCR immune repertoires.
#' Integrates six research directions into six interpretable concept axes,
#' builds a concept-bottleneck embedding space, separates diseases with
#' explainable classifiers, and discovers biomarkers at statistical and
#' sequence levels.
#'
#' @section Workflow:
#' The analysis follows a five-layer pipeline mirroring the algorithm
#' architecture:
#' \enumerate{
#'   \item Input: \code{\link{ReadRepertoire}}, \code{\link{fetch_vdjdb}}
#'   \item QC: \code{\link{QCRepertoire}}, \code{\link{NormalizeRepertoire}}
#'   \item Features: \code{\link{ComputeFeatures}} (six concept modules)
#'   \item Embedding: \code{\link{ConceptBottleneckEmbed}}
#'   \item Classification: \code{\link{DiseaseClassify}} (with SHAP)
#'   \item Markers: \code{\link{FindMarkers}} (statistical + sequence)
#' }
#'
#' @docType package
#' @name CDRscope-package
#' @keywords internal
"_PACKAGE"
