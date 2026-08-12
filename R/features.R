# ---- amino-acid physico-chemical tables (internal) ------------------------
.AA_KD <- c(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5,
            G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
            P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
.AA_CHARGE <- c(K=+1, R=+1, H=+0.5, D=-1, E=-1)
.AA_SET <- names(.AA_KD)

.cdr3_charge <- function(seq) {
  aa <- strsplit(seq, "")[[1]]
  ch <- sum(.AA_CHARGE[aa[aa %in% names(.AA_CHARGE)]], na.rm = TRUE)
  ch / max(1, length(aa))
}
.cdr3_hydro <- function(seq) {
  aa <- strsplit(seq, "")[[1]]
  v <- .AA_KD[aa[aa %in% .AA_SET]]
  if (length(v) == 0) return(0)
  mean(v, na.rm = TRUE)
}
.gini <- function(x) {
  x <- sort(x[x > 0])
  n <- length(x)
  if (n == 0) return(NA)
  2 * sum(seq_len(n) * x) / (n * sum(x)) - (n + 1) / n
}
.kmers <- function(seq, k = 3) {
  L <- nchar(seq)
  if (L < k) return(character(0))
  vapply(seq_len(L - k + 1), function(i) substr(seq, i, i + k - 1),
         character(1))
}
.powerlaw_alpha <- function(counts) {
  counts <- sort(counts[counts > 0], decreasing = TRUE)
  if (length(counts) < 5) return(NA)
  r <- seq_along(counts)
  fit <- tryCatch(lm(log(counts) ~ log(r)), error = function(e) NULL)
  if (is.null(fit)) return(NA)
  -coef(fit)[2]
}
.shannon <- function(p) -sum(p * log(p[p > 0]))
.simpson <- function(p) 1 - sum(p^2)

# ---- module 1: antigen decoding ------------------------------------------
#' @title Concept module 1: antigen decoding features
#'
#' @description CDR3 3-mer spectrum (PCA-reduced), public-clonotype hit rate,
#' and epitope-prediction proxy. The 3-mer spectrum is summarised by its top
#' principal component to keep the concept axis compact and interpretable.
#'
#' @param object A \code{CDRobject}.
#' @param k Integer; k-mer size. Default 3.
#' @param public_ref Optional data.frame of known public CDR3s (column
#'   \code{cdr3_aa}). If \code{NULL}, uses a small built-in reference.
#' @return A \code{data.frame} of features per sample.
#' @export
ComputeMotifSpectrum <- function(object, k = 3, public_ref = NULL) {
  stopifnot(inherits(object, "CDRobject"))
  clones <- object@clones
  if (nrow(clones) == 0) return(data.frame(sample_id = object@meta$sample_id))
  if (is.null(public_ref)) {
    public_ref <- data.frame(cdr3_aa = c("CASSLGGQNTLYF", "CASSLGGNQDTQYF",
                                          "CASSDSYNEQFF", "CASSPSYNEQFF",
                                          "CSARTGYQNF"),
                             stringsAsFactors = FALSE)
  }
  samples <- unique(clones$sample_id)
  kmers_all <- unique(unlist(lapply(clones$cdr3_aa, .kmers, k = k)))
  M <- matrix(0, nrow = length(samples), ncol = length(kmers_all),
              dimnames = list(samples, kmers_all))
  for (i in seq_along(samples)) {
    s <- samples[i]
    ks <- unlist(lapply(clones$cdr3_aa[clones$sample_id == s], .kmers, k = k))
    t <- table(ks)
    M[i, names(t)] <- as.integer(t)
  }
  M <- M / rowSums(M)
  M[is.na(M)] <- 0
  motif_pc1 <- if (ncol(M) > 1) prcomp(M, scale. = TRUE)$x[, 1] else rep(0, nrow(M))
  pub <- vapply(samples, function(s) {
    aa <- clones$cdr3_aa[clones$sample_id == s]
    sum(aa %in% public_ref$cdr3_aa) / length(aa)
  }, numeric(1))
  data.frame(sample_id = samples,
             motif_pc1 = motif_pc1,
             public_hit_rate = pub,
             stringsAsFactors = FALSE)
}

# ---- module 2: repertoire statistics -------------------------------------
#' @title Concept module 2: repertoire statistics
#'
#' @description Diversity (Shannon entropy, Simpson, clonality), power-law
#' exponent, and CDR3 length distribution moments.
#' @param object A \code{CDRobject}.
#' @return A \code{data.frame} of features per sample.
#' @export
ComputeDiversity <- function(object) {
  stopifnot(inherits(object, "CDRobject"))
  clones <- object@clones
  if (nrow(clones) == 0) return(data.frame(sample_id = object@meta$sample_id))
  samples <- unique(clones$sample_id)
  res <- do.call(rbind, lapply(samples, function(s) {
    d <- clones[clones$sample_id == s, , drop = FALSE]
    p <- d$freq
    if (is.null(p) || all(is.na(p))) p <- d$count / sum(d$count)
    len <- nchar(d$cdr3_aa)
    data.frame(
      sample_id   = s,
      shannon     = .shannon(p),
      simpson     = .simpson(p),
      clonality   = 1 - .shannon(p) / log(length(p)),
      powerlaw_a  = .powerlaw_alpha(d$count),
      cdr3_len_mean = mean(len),
      cdr3_len_sd   = sd(len),
      stringsAsFactors = FALSE
    )
  }))
  res
}

# ---- module 3: selection imprint -----------------------------------------
#' @title Concept module 3: selection imprint
#'
#' @description Non-germline length proxy (insertion/deletion burden), CDR3
#' net charge and hydrophobicity distributions, and V/J gene usage bias
#' (KL divergence from uniform).
#' @param object A \code{CDRobject}.
#' @return A \code{data.frame} of features per sample.
#' @export
ComputeSelectionImprint <- function(object) {
  stopifnot(inherits(object, "CDRobject"))
  clones <- object@clones
  if (nrow(clones) == 0) return(data.frame(sample_id = object@meta$sample_id))
  samples <- unique(clones$sample_id)
  v_universe <- unique(clones$v_gene); j_universe <- unique(clones$j_gene)
  res <- do.call(rbind, lapply(samples, function(s) {
    d <- clones[clones$sample_id == s, , drop = FALSE]
    len <- nchar(d$cdr3_aa)
    charge <- vapply(d$cdr3_aa, .cdr3_charge, numeric(1))
    hydro  <- vapply(d$cdr3_aa, .cdr3_hydro,  numeric(1))
    vtab <- table(factor(d$v_gene, levels = v_universe)); vtab <- vtab / sum(vtab)
    jtab <- table(factor(d$j_gene, levels = j_universe)); jtab <- jtab / sum(jtab)
    kl <- function(p) { p <- p[p > 0]; q <- 1/length(p); sum(p * log(p / q)) }
    data.frame(
      sample_id     = s,
      n_insertion_proxy = mean(pmax(0, len - 10)),
      cdr3_charge_mean  = mean(charge),
      cdr3_charge_sd    = sd(charge),
      cdr3_hydro_mean   = mean(hydro),
      v_usage_kl        = kl(vtab),
      j_usage_kl        = kl(jtab),
      stringsAsFactors = FALSE
    )
  }))
  res
}

# ---- module 4: disease perturbation --------------------------------------
#' @title Concept module 4: disease perturbation
#'
#' @description Clonal expansion (Gini, top-clone fraction), convergence
#' index (cross-sample shared clonotypes), and diversity collapse relative to
#' a healthy baseline.
#' @param object A \code{CDRobject}.
#' @param healthy_label Character; label in \code{meta$group} used as baseline.
#'   Default \code{"healthy"}.
#' @return A \code{data.frame} of features per sample.
#' @export
ComputeConvergence <- function(object, healthy_label = "healthy") {
  stopifnot(inherits(object, "CDRobject"))
  clones <- object@clones
  if (nrow(clones) == 0) return(data.frame(sample_id = object@meta$sample_id))
  samples <- unique(clones$sample_id)
  grp <- setNames(object@meta$group, object@meta$sample_id)
  healthy_samples <- names(which(grp == healthy_label))
  baseline_shannon <- if (length(healthy_samples) > 0) {
    mean(vapply(healthy_samples, function(s) {
      d <- clones[clones$sample_id == s, , drop = FALSE]
      .shannon(d$freq)
    }, numeric(1)), na.rm = TRUE)
  } else NA
  per_sample_sets <- lapply(samples, function(s) {
    unique(clones$cdr3_aa[clones$sample_id == s])
  })
  names(per_sample_sets) <- samples
  res <- do.call(rbind, lapply(seq_along(samples), function(i) {
    s <- samples[i]
    d <- clones[clones$sample_id == s, , drop = FALSE]
    p <- d$freq
    sh <- .shannon(p)
    others <- setdiff(samples, s)
    conv <- if (length(others)) {
      mean(vapply(others, function(o)
        length(intersect(per_sample_sets[[s]], per_sample_sets[[o]])) /
        length(per_sample_sets[[s]]), numeric(1)))
    } else 0
    data.frame(
      sample_id       = s,
      gini            = .gini(d$count),
      top_clone_frac  = max(p),
      convergence_idx = conv,
      diversity_collapse = if (is.na(baseline_shannon)) NA else baseline_shannon - sh,
      stringsAsFactors = FALSE
    )
  }))
  res
}

# ---- module 5: history / lineage (BCR) -----------------------------------
#' @title Concept module 5: history & lineage
#'
#' @description Somatic hypermutation (SHM) burden proxy for BCR heavy chains.
#' For TCR-only repertoires this module returns NA placeholders (the axis is
#' retained so the concept space stays aligned across studies).
#' @param object A \code{CDRobject}.
#' @return A \code{data.frame} of features per sample.
#' @export
ComputeSHM <- function(object) {
  stopifnot(inherits(object, "CDRobject"))
  samples <- unique(object@clones$sample_id)
  if (length(samples) == 0) samples <- object@meta$sample_id
  is_bcr <- any(grepl("^IGH", object@clones$v_gene))
  if (is_bcr) {
    res <- do.call(rbind, lapply(samples, function(s) {
      d <- object@clones[object@clones$sample_id == s, , drop = FALSE]
      data.frame(sample_id = s,
                 shm_mean_len = mean(nchar(d$cdr3_aa)),
                 shm_diversity = length(unique(d$cdr3_aa)) / nrow(d),
                 stringsAsFactors = FALSE)
    }))
  } else {
    res <- data.frame(sample_id = samples,
                      shm_mean_len = NA_real_,
                      shm_diversity = NA_real_,
                      stringsAsFactors = FALSE)
  }
  object@misc$shm_available <- is_bcr
  res
}

# ---- module 6: chain pairing (single-cell) -------------------------------
#' @title Concept module 6: chain pairing
#'
#' @description Pairing diversity for single-cell VDJ data (TRA-TRB or
#' IGH-IGK/IGL). For bulk repertoires returns NA placeholders.
#' @param object A \code{CDRobject}.
#' @return A \code{data.frame} of features per sample.
#' @export
ComputePairing <- function(object) {
  stopifnot(inherits(object, "CDRobject"))
  samples <- unique(object@clones$sample_id)
  if (length(samples) == 0) samples <- object@meta$sample_id
  chains <- unique(object@clones$chain)
  paired <- length(intersect(chains, c("TRA", "TRG", "IGL", "IGK"))) > 0 &&
             length(intersect(chains, c("TRB", "TRD", "IGH"))) > 0
  if (paired && "pair_id" %in% colnames(object@clones)) {
    res <- do.call(rbind, lapply(samples, function(s) {
      d <- object@clones[object@clones$sample_id == s, , drop = FALSE]
      ptab <- table(d$pair_id)
      data.frame(sample_id = s,
                 pairing_diversity = .shannon(ptab / sum(ptab)),
                 pairing_n = length(ptab),
                 stringsAsFactors = FALSE)
    }))
  } else {
    res <- data.frame(sample_id = samples,
                      pairing_diversity = NA_real_,
                      pairing_n = NA_integer_,
                      stringsAsFactors = FALSE)
  }
  object@misc$pairing_available <- paired
  res
}

# ---- master feature builder ----------------------------------------------
#' @title Compute six-concept features
#'
#' @description Runs all six concept modules and assembles a sample-by-feature
#' matrix. Each module contributes one interpretable axis block. The mapping
#' from module to columns is stored in \code{feature_modules}.
#'
#' @param object A \code{\link{CDRobject}}.
#' @param modules Character vector; which modules to run. Default all six.
#' @param verbose Logical; print progress. Default \code{TRUE}.
#'
#' @return A \code{CDRobject} with \code{features} and \code{feature_modules}
#'   populated.
#'
#' @export
#' @seealso \code{\link{ComputeMotifSpectrum}}, \code{\link{ComputeDiversity}},
#'   \code{\link{ComputeSelectionImprint}}, \code{\link{ComputeConvergence}},
#'   \code{\link{ComputeSHM}}, \code{\link{ComputePairing}}
ComputeFeatures <- function(object,
                            modules = c("motif", "diversity", "selection",
                                        "convergence", "shm", "pairing"),
                            verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))
  runners <- list(
    motif       = ComputeMotifSpectrum,
    diversity   = ComputeDiversity,
    selection   = ComputeSelectionImprint,
    convergence = ComputeConvergence,
    shm         = ComputeSHM,
    pairing     = ComputePairing
  )
  modules <- match.arg(modules, names(runners), several.ok = TRUE)
  feats <- list()
  for (m in modules) {
    if (verbose) message("Computing concept module: ", m)
    feats[[m]] <- runners[[m]](object)
  }
  base <- lapply(feats, function(d) {
    rn <- d$sample_id; d$sample_id <- NULL; rownames(d) <- rn; d
  })
  ord <- object@meta$sample_id
  base <- lapply(base, function(d) d[ord, , drop = FALSE])
  feat_mat <- do.call(cbind, base)
  rownames(feat_mat) <- ord
  modules_map <- list()
  pos <- 0
  for (m in modules) {
    n <- ncol(base[[m]])
    modules_map[[m]] <- seq(pos + 1, pos + n)
    pos <- pos + n
  }
  object@features <- as.matrix(feat_mat)
  object@features[is.na(object@features)] <- 0
  object@feature_modules <- modules_map
  object@misc$feature_names <- colnames(feat_mat)
  if (verbose) message(sprintf("Assembled %d features across %d modules.",
                               ncol(feat_mat), length(modules)))
  object
}
