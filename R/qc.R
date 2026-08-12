#' @title Quality control of a repertoire
#'
#' @description Filters non-productive CDR3 sequences: those containing
#' stop codons (\code{*}), non-amino-acid characters, or that fall outside a
#' length range. Optionally enforces the Cys-Phe/Trp framework. Updates the
#' \code{clones} slot and sample read counts in \code{meta}.
#'
#' @param object A \code{\link{CDRobject}}.
#' @param min_len Integer; minimum CDR3 amino-acid length. Default 6.
#' @param max_len Integer; maximum CDR3 amino-acid length. Default 30.
#' @param enforce_framework Logical; require leading C and trailing F/W.
#'   Default \code{TRUE}.
#'
#' @return A filtered \code{CDRobject}.
#' @export
QCRepertoire <- function(object, min_len = 6, max_len = 30,
                         enforce_framework = TRUE) {
  stopifnot(inherits(object, "CDRobject"))
  clones <- object@clones
  if (nrow(clones) == 0) return(object)
  aas <- "ACDEFGHIKLMNPQRSTVWY"
  bad_stop <- grepl("\\*", clones$cdr3_aa)
  bad_char <- !grepl(paste0("^[", aas, "]+$"), clones$cdr3_aa)
  bad_len  <- nchar(clones$cdr3_aa) < min_len | nchar(clones$cdr3_aa) > max_len
  keep <- !bad_stop & !bad_char & !bad_len
  if (enforce_framework) {
    fw <- grepl("^C.*[FW]$", clones$cdr3_aa)
    keep <- keep & fw
  }
  object@clones <- clones[keep, , drop = FALSE]
  object@meta$reads <- as.integer(tapply(object@clones$count,
                                         object@clones$sample_id, sum))[
    match(object@meta$sample_id,
          names(tapply(object@clones$count, object@clones$sample_id, sum)))]
  object@meta$reads[is.na(object@meta$reads)] <- 0L
  object@misc$qc <- list(
    removed = sum(!keep),
    kept = sum(keep),
    min_len = min_len, max_len = max_len,
    enforce_framework = enforce_framework
  )
  object
}

#' @title Normalize repertoire frequencies
#'
#' @description Recomputes per-sample clone frequencies and optionally
#' rarefies (down-samples) each sample to a fixed read count so that
#' diversity metrics are comparable across samples of unequal depth.
#'
#' @param object A \code{\link{CDRobject}}.
#' @param method Character; \code{"freq"} (default) recomputes frequencies;
#'   \code{"rarefy"} down-samples reads.
#' @param target_reads Integer; target read count per sample when
#'   \code{method = "rarefy"}. Samples below this are dropped. Default 1000.
#'
#' @return A normalized \code{CDRobject}.
#' @export
NormalizeRepertoire <- function(object, method = c("freq", "rarefy"),
                                target_reads = 1000) {
  stopifnot(inherits(object, "CDRobject"))
  method <- match.arg(method)
  clones <- object@clones
  if (nrow(clones) == 0) return(object)
  if (method == "freq") {
    clones$freq <- ave(clones$count, clones$sample_id,
                       FUN = function(x) x / sum(x))
    object@clones <- clones
    return(object)
  }
  keep_samples <- names(which(tapply(clones$count, clones$sample_id, sum) >=
                              target_reads))
  clones <- clones[clones$sample_id %in% keep_samples, , drop = FALSE]
  out <- do.call(rbind, lapply(split(clones, clones$sample_id), function(d) {
    pool <- rep(seq_len(nrow(d)), d$count)
    drawn <- sample(pool, target_reads, replace = FALSE)
    tab <- table(factor(drawed, levels = seq_len(nrow(d))))
    d$count <- as.integer(tab)
    d$freq <- d$count / target_reads
    d
  }))
  object@clones <- out
  object@meta <- object@meta[object@meta$sample_id %in% keep_samples, ,
                             drop = FALSE]
  object@meta$reads <- as.integer(tapply(object@clones$count,
                                         object@clones$sample_id, sum))[
    match(object@meta$sample_id,
          names(tapply(object@clones$count, object@clones$sample_id, sum)))]
  object@misc$normalize <- list(method = method, target_reads = target_reads)
  object
}
