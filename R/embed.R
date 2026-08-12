#' @title Concept-bottleneck embedding
#'
#' @description Builds the interpretable embedding space from the six-concept
#' features. The concept-bottleneck principle is enforced by projecting the
#' feature matrix through a linear bottleneck (PCA on scaled concept axes) so
#' that every embedding dimension is a signed combination of interpretable
#' concepts. The first two PCs populate \code{reduction} for plotting.
#'
#' @param object A \code{\link{CDRobject}} with features populated.
#' @param method Character; \code{"pca"} (default, always available) or
#'   \code{"umap"} (requires \code{umap} package).
#' @param ndim Integer; number of embedding dimensions. Default 10.
#'
#' @return A \code{CDRobject} with \code{embedding} and \code{reduction}
#'   populated.
#' @export
#' @importFrom stats prcomp
ConceptBottleneckEmbed <- function(object, method = c("pca", "umap"),
                                   ndim = 10) {
  stopifnot(inherits(object, "CDRobject"))
  method <- match.arg(method)
  feat <- object@features
  if (ncol(feat) == 0) stop("Run ComputeFeatures() first.")
  feat[is.na(feat)] <- 0
  if (method == "umap") {
    if (!requireNamespace("umap", quietly = TRUE)) {
      warning("Package 'umap' not available; falling back to PCA.")
      method <- "pca"
    }
  }
  if (method == "pca") {
    v <- apply(feat, 2, var, na.rm = TRUE)
    keep <- is.finite(v) & v > 0
    feat_keep <- feat[, keep, drop = FALSE]
    if (ncol(feat_keep) < 2)
      stop("Fewer than 2 non-constant features; cannot embed.")
    pc <- prcomp(feat_keep, scale. = TRUE,
                 rank. = min(ndim, ncol(feat_keep)))
    emb <- pc$x
    attr(emb, "rotation") <- pc$rotation
    attr(emb, "method") <- "pca"
    attr(emb, "used_features") <- colnames(feat_keep)
  } else {
    um <- umap::umap(feat)
    emb <- um$layout
    colnames(emb) <- paste0("UMAP_", seq_len(ncol(emb)))
    attr(emb, "method") <- "umap"
  }
  object@embedding <- emb
  object@reduction <- emb[, 1:min(2, ncol(emb)), drop = FALSE]
  colnames(object@reduction) <- c("CB1", "CB2")[seq_len(ncol(object@reduction))]
  object@misc$embed_method <- method
  object@misc$embed_rotation <- attr(emb, "rotation")
  object
}
