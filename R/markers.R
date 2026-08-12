#' @title Find biomarkers (statistical and sequence level)
#'
#' @description Two-layer marker discovery.
#' \itemize{
#'   \item \strong{Statistical level}: for each concept feature, tests
#'   disease-vs-rest with Wilcoxon and applies Benjamini-Hochberg FDR.
#'   \item \strong{Sequence level}: enriches CDR3 k-mers per disease group
#'   (Fisher's exact test) and finds public disease clonotypes shared across
#'   patients.
#' }
#'
#' @param object A \code{\link{CDRobject}} with features populated.
#' @param level Character; \code{"statistical"}, \code{"sequence"}, or
#'   \code{"both"}. Default \code{"both"}.
#' @param k Integer; k-mer size for sequence markers. Default 3.
#' @param min_patient_frac Numeric; minimum fraction of patients sharing a
#'   clonotype for it to be a public disease clonotype. Default 0.2.
#' @param fdr Numeric; FDR threshold. Default 0.05.
#'
#' @return A \code{CDRobject} with \code{markers} populated.
#' @export
#' @importFrom stats wilcox.test fisher.test p.adjust phyper
FindMarkers <- function(object, level = c("both", "statistical", "sequence"),
                        k = 3, min_patient_frac = 0.2, fdr = 0.05) {
  stopifnot(inherits(object, "CDRobject"))
  level <- match.arg(level)
  markers <- object@markers
  if (level %in% c("statistical", "both")) {
    feat <- object@features
    grp <- object@meta$group
    if (ncol(feat) == 0) stop("Run ComputeFeatures() first.")
    samples <- rownames(feat)
    groups <- unique(grp[!is.na(grp)])
    rows <- list()
    for (g in groups) {
      in_g <- which(grp == g)
      out_g <- which(grp != g & !is.na(grp))
      if (length(in_g) < 2 || length(out_g) < 2) next
      for (fn in colnames(feat)) {
        x <- feat[in_g, fn]; y <- feat[out_g, fn]
        p <- tryCatch(wilcox.test(x, y)$p.value, error = function(e) NA)
        rows[[length(rows) + 1]] <- data.frame(
          feature = fn, group = g,
          mean_in = mean(x, na.rm = TRUE),
          mean_out = mean(y, na.rm = TRUE),
          mean_diff = mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE),
          pval = p, stringsAsFactors = FALSE)
      }
    }
    stat_df <- do.call(rbind, rows)
    stat_df$padj <- p.adjust(stat_df$pval, method = "BH")
    stat_df <- stat_df[order(stat_df$padj), , drop = FALSE]
    rownames(stat_df) <- NULL
    markers$statistical <- stat_df
    markers$statistical_sig <- stat_df[stat_df$padj < fdr, , drop = FALSE]
  }
  if (level %in% c("sequence", "both")) {
    clones <- object@clones
    if (nrow(clones) == 0) stop("No clones available for sequence markers.")
    grp <- setNames(object@meta$group, object@meta$sample_id)
    groups <- unique(grp[!is.na(grp)])
    seq_rows <- list()
    public_rows <- list()
    for (g in groups) {
      g_samples <- names(which(grp == g))
      o_samples <- names(which(grp != g & !is.na(grp)))
      if (length(g_samples) < 2) next
      g_aa <- clones$cdr3_aa[clones$sample_id %in% g_samples]
      o_aa <- clones$cdr3_aa[clones$sample_id %in% o_samples]
      g_kmers <- unlist(lapply(g_aa, .kmers, k = k))
      o_kmers <- unlist(lapply(o_aa, .kmers, k = k))
      all_kmers <- unique(c(g_kmers, o_kmers))
      for (km in all_kmers) {
        a <- sum(g_kmers == km); b <- length(g_kmers) - a
        c <- sum(o_kmers == km); d <- length(o_kmers) - c
        if (a < 3) next
        p <- tryCatch(fisher.test(matrix(c(a, b, c, d), 2, 2))$p.value,
                      error = function(e) NA)
        or <- (a * d) / (b * c + 1e-9)
        seq_rows[[length(seq_rows) + 1]] <- data.frame(
          motif = km, group = g, count_in = a, count_out = c,
          odds_ratio = or, pval = p, stringsAsFactors = FALSE)
      }
      aa_by_sample <- split(clones$cdr3_aa[clones$sample_id %in% g_samples],
                            clones$sample_id[clones$sample_id %in% g_samples])
      shared <- Reduce(intersect, lapply(aa_by_sample, unique))
      n_pat <- length(g_samples)
      frac <- if (n_pat > 0) sapply(shared, function(s)
        mean(sapply(aa_by_sample, function(u) s %in% u))) else numeric(0)
      keep <- frac >= min_patient_frac
      if (any(keep)) {
        public_rows[[length(public_rows) + 1]] <- data.frame(
          cdr3_aa = shared[keep], group = g,
          patient_frac = frac[keep], stringsAsFactors = FALSE)
      }
    }
    seq_df <- do.call(rbind, seq_rows)
    if (!is.null(seq_df) && nrow(seq_df)) {
      seq_df$padj <- p.adjust(seq_df$pval, method = "BH")
      seq_df <- seq_df[order(-seq_df$odds_ratio), , drop = FALSE]
      rownames(seq_df) <- NULL
      markers$sequence <- seq_df
      markers$sequence_sig <- seq_df[seq_df$padj < fdr & seq_df$odds_ratio > 1, ,
                                     drop = FALSE]
    } else {
      markers$sequence <- data.frame()
      markers$sequence_sig <- data.frame()
    }
    pub_df <- do.call(rbind, public_rows)
    markers$public_clonotypes <- if (!is.null(pub_df)) {
      pub_df[order(-pub_df$patient_frac), , drop = FALSE]
    } else data.frame()
  }
  object@markers <- markers
  object
}
