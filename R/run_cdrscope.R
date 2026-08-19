# ===========================================================================
# CDRscope v2.0 — Unified entry point with chain selection
# ===========================================================================

#' @title Run CDRscope analysis pipeline with chain selection
#'
#' @description Unified entry point for CDRscope v2.0. Automatically detects
#' available TCR/BCR chains from input data, or allows the user to specify
#' which chain combination to analyse. Supports three modes:
#' \itemize{
#'   \item \code{"single"} — Single-chain analysis (e.g. TRA or TRB alone)
#'   \item \code{"paired"} — TRA+TRB paired analysis (requires both chains)
#'   \item \code{"all"} — All available chains analysed jointly
#' }
#'
#' @param input Path to a directory of CSV files, a single CSV/TSV file, or a
#'   \code{CDRobject}. File names should follow the pattern
#'   \code{<sample_id>_r__<chain>.csv} or \code{<sample_id>__<chain>.csv}.
#' @param chain Character; chain selection mode. One of \code{"single"}
#'   (default: analyse the most abundant chain), \code{"paired"} (TRA+TRB),
#'   \code{"all"} (all chains jointly), or specific chain names like
#'   \code{"TRB"}, \code{"TRA"}, \code{"TRA+TRB"}, \code{"IGH"}, etc.
#' @param group Optional vector of group labels (e.g. \code{c("Control", "RA")}).
#'   If input is a directory, auto-detected from subdirectory names.
#' @param control_dir Path to control group directory (if using directory layout).
#' @param patient_dir Path to patient/disease group directory.
#' @param features Character; which feature set to use. \code{"basic"} (original
#'   6 modules, 20 features), \code{"enhanced"} (adds 5 RA-specific modules,
#'   ~65 features), or \code{"all"} (both). Default \code{"enhanced"}.
#' @param classifier Character; classifier method. One of \code{"rf"} (Random
#'   Forest), \code{"lasso"} (L1-regularised LR), \code{"xgb"} (XGBoost),
#'   \code{"ensemble"} (voting ensemble), \code{"glm"} (original logistic
#'   regression), or \code{"compare"} (run all and compare). Default \code{"rf"}.
#' @param cv_folds Integer; number of cross-validation folds. Default 5.
#' @param use_embedding Logical; whether to extract ESM-2 embeddings and run
#'   UMAP. Requires Python with \code{transformers} and \code{torch}.
#'   Default \code{FALSE} (fast mode).
#' @param verbose Logical; print progress. Default \code{TRUE}.
#'
#' @return A \code{list} with components:
#'   \itemize{
#'     \item \code{object} — The \code{CDRobject} with features and classification.
#'     \item \code{cv_results} — Cross-validation summary (accuracy, F1, per-fold).
#'     \item \code{feature_importance} — Feature importance scores.
#'     \item \code{chain_info} — Detected chains and their sample counts.
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Single chain (auto-detect)
#' result <- run_CDRscope("path/to/RA_data", chain = "single")
#'
#' # TRA+TRB paired
#' result <- run_CDRscope("path/to/RA_data", chain = "paired")
#'
#' # All chains
#' result <- run_CDRscope("path/to/RA_data", chain = "all")
#'
#' # Compare all classifiers
#' result <- run_CDRscope("path/to/RA_data", chain = "paired",
#'                         classifier = "compare")
#' }
run_CDRscope <- function(
    input,
    chain          = c("single", "paired", "all"),
    group          = NULL,
    control_dir    = NULL,
    patient_dir    = NULL,
    features       = c("enhanced", "basic", "all"),
    classifier     = c("rf", "lasso", "xgb", "ensemble", "glm", "compare"),
    cv_folds       = 5,
    use_embedding  = FALSE,
    verbose        = TRUE
) {
  chain     <- match.arg(chain)
  features  <- match.arg(features)
  classifier <- match.arg(classifier)

  if (verbose) {
    cat("============================================\n")
    cat("  CDRscope v2.0 — Unified Analysis Pipeline\n")
    cat("============================================\n")
    cat(sprintf("  Chain mode:   %s\n", chain))
    cat(sprintf("  Features:     %s\n", features))
    cat(sprintf("  Classifier:   %s\n", classifier))
    cat(sprintf("  CV folds:     %d\n", cv_folds))
    cat(sprintf("  ESM-2 embed:  %s\n", if (use_embedding) "yes" else "no"))
    cat("============================================\n\n")
  }

  # ---- Step 1: Load data & detect chains ----
  if (verbose) cat("Step 1: Loading data & detecting chains...\n")

  if (inherits(input, "CDRobject")) {
    obj <- input
    chain_info <- .detect_chains(obj)
  } else if (is.character(input) && dir.exists(input)) {
    # Directory layout: subdirectories for groups, CSV files per sample
    result <- .load_directory(input, control_dir, patient_dir, chain, verbose)
    obj        <- result$object
    chain_info <- result$chain_info
  } else if (is.character(input) && file.exists(input)) {
    # Single file
    obj <- ReadRepertoire(input)
    chain_info <- .detect_chains(obj)
  } else {
    stop("'input' must be a CDRobject, a directory path, or a file path.")
  }

  if (verbose) {
    cat(sprintf("  Samples: %d (%d groups)\n", nrow(obj@meta),
                length(unique(obj@meta$group))))
    cat(sprintf("  Clones:  %d\n", nrow(obj@clones)))
    cat(sprintf("  Chains:  %s\n", paste(chain_info$chains, collapse = ", ")))
    cat(sprintf("  Selected: %s\n", chain_info$selected))
  }

  # ---- Step 2: QC & Normalize ----
  if (verbose) cat("\nStep 2: QC & Normalize...\n")
  obj <- QCRepertoire(obj)
  obj <- NormalizeRepertoire(obj)
  if (verbose) cat(sprintf("  After QC: %d samples, %d clones\n",
                           nrow(obj@meta), nrow(obj@clones)))

  # ---- Step 3: Feature engineering ----
  if (verbose) cat("\nStep 3: Feature engineering...\n")

  if (chain_info$mode == "multi") {
    # Multi-chain: compute features per chain then merge
    feature_list <- .compute_multi_chain_features(obj, chain_info, features, verbose)
    obj <- feature_list$object
    feat_df <- feature_list$features
  } else {
    # Single chain: standard flow
    if (features %in% c("enhanced", "all")) {
      source(system.file("extdata", "features_ra.R", package = "CDRscope"),
             local = TRUE)
      obj <- ComputeFeaturesRA(obj, include_original = (features == "all"))
    } else {
      obj <- ComputeFeatures(obj)
    }
    feat_df <- as.data.frame(obj@features)
    feat_df$sample_id <- rownames(feat_df)
  }

  if (verbose) cat(sprintf("  Features: %d\n", ncol(feat_df) - 1))

  # ---- Step 4: Classification ----
  if (verbose) cat("\nStep 4: Classification (%s)...\n", classifier)

  cv_results <- .run_classification(obj, feat_df, classifier, cv_folds, verbose)
  importance <- cv_results$importance

  # ---- Step 5: ESM-2 embedding (optional) ----
  if (use_embedding) {
    if (verbose) cat("\nStep 5: ESM-2 embedding & UMAP...\n")
    embedding_result <- .run_embedding(obj, chain_info, verbose)
    obj@misc$embedding <- embedding_result
  }

  # ---- Step 6: Return results ----
  if (verbose) cat("\nDone.\n")

  list(
    object             = obj,
    cv_results         = cv_results$summary,
    cv_details         = cv_results$details,
    feature_importance = importance,
    chain_info         = chain_info,
    roc_data           = cv_results$roc_data,
    call               = match.call()
  )
}


# ===========================================================================
# Internal helpers
# ===========================================================================

#' Detect available chains from a CDRobject
#' @keywords internal
.detect_chains <- function(obj) {
  chains <- unique(obj@clones$chain)
  if (length(chains) == 0) chains <- "UNKNOWN"

  # Count samples per chain
  chain_counts <- table(obj@clones$chain)
  chain_counts <- sort(chain_counts, decreasing = TRUE)

  list(
    chains   = as.character(names(chain_counts)),
    counts   = as.integer(chain_counts),
    primary  = as.character(names(chain_counts)[1]),
    selected = "all",
    mode     = "single"
  )
}


#' Resolve chain selection
#' @keywords internal
.resolve_chain <- function(chain_info, chain_arg) {
  available <- chain_info$chains

  if (chain_arg == "single") {
    selected <- chain_info$primary
    mode <- "single"
  } else if (chain_arg == "paired") {
    if (all(c("TRA", "TRB") %in% available)) {
      selected <- "TRA+TRB"
      mode <- "multi"
    } else {
      warning("TRA+TRB not both available. Falling back to 'all'.")
      selected <- "all"
      mode <- "multi"
    }
  } else if (chain_arg == "all") {
    selected <- "all"
    mode <- if (length(available) > 1) "multi" else "single"
  } else if (grepl("\\+", chain_arg)) {
    # e.g. "TRA+TRB"
    parts <- strsplit(chain_arg, "\\+")[[1]]
    parts <- trimws(parts)
    if (all(parts %in% available)) {
      selected <- chain_arg
      mode <- "multi"
    } else {
      missing <- setdiff(parts, available)
      warning(sprintf("Chains not found: %s. Falling back to 'all'.",
                      paste(missing, collapse = ", ")))
      selected <- "all"
      mode <- "multi"
    }
  } else {
    # Specific single chain
    if (chain_arg %in% available) {
      selected <- chain_arg
      mode <- "single"
    } else {
      warning(sprintf("Chain '%s' not found. Using '%s'.",
                      chain_arg, chain_info$primary))
      selected <- chain_info$primary
      mode <- "single"
    }
  }

  list(selected = selected, mode = mode)
}


#' Load data from directory layout
#' @keywords internal
.load_directory <- function(data_dir, control_dir, patient_dir, chain_arg, verbose) {
  # Auto-detect subdirectories
  subdirs <- list.dirs(data_dir, full.names = TRUE, recursive = FALSE)
  if (length(subdirs) == 0) {
    stop("No subdirectories found in '", data_dir,
         "'. Expected Control/Patient directory layout.")
  }

  # Identify groups
  groups <- basename(subdirs)
  if (verbose) cat(sprintf("  Found %d groups: %s\n", length(groups),
                           paste(groups, collapse = ", ")))

  all_clones <- list()
  all_meta <- list()

  for (i in seq_along(subdirs)) {
    grp <- groups[i]
    pattern <- if (chain_arg %in% c("paired", "TRA+TRB")) {
      "__TR[AB]\\.csv$"
    } else if (chain_arg %in% c("single", "all")) {
      "\\.csv$"
    } else {
      sprintf("__%s\\.csv$", chain_arg)
    }

    files <- list.files(subdirs[i], pattern = pattern, full.names = TRUE)
    if (verbose) cat(sprintf("  %s: %d files\n", grp, length(files)))

    for (f in files) {
      bn <- basename(f)
      sid <- sub("__.*$", "", bn)
      chain_type <- if (grepl("__TRA", bn)) "TRA"
                    else if (grepl("__TRB", bn)) "TRB"
                    else if (grepl("__TRG", bn)) "TRG"
                    else if (grepl("__TRD", bn)) "TRD"
                    else if (grepl("__IGH", bn)) "IGH"
                    else if (grepl("__IGK", bn)) "IGK"
                    else if (grepl("__IGL", bn)) "IGL"
                    else "UNKNOWN"

      df <- tryCatch(
        read.csv(f, stringsAsFactors = FALSE),
        error = function(e) NULL
      )
      if (is.null(df) || nrow(df) == 0) next

      # Map columns
      if (!"cdr3_aa" %in% names(df) && "junction_aa" %in% names(df))
        names(df)[names(df) == "junction_aa"] <- "cdr3_aa"
      if (!"v_gene" %in% names(df) && "v_call" %in% names(df))
        names(df)[names(df) == "v_call"] <- "v_gene"
      if (!"j_gene" %in% names(df) && "j_call" %in% names(df))
        names(df)[names(df) == "j_call"] <- "j_gene"
      if (!"count" %in% names(df) && "duplicate_count" %in% names(df))
        names(df)[names(df) == "duplicate_count"] <- "count"

      df$sample_id <- sid
      df$chain     <- chain_type
      df$group     <- grp
      if (!"freq" %in% names(df)) df$freq <- df$count / sum(df$count)

      cols <- intersect(c("sample_id", "cdr3_aa", "v_gene", "j_gene",
                          "chain", "count", "freq", "group"), names(df))
      all_clones[[length(all_clones) + 1]] <- df[, cols, drop = FALSE]
    }
  }

  if (length(all_clones) == 0) stop("No CSV files loaded. Check directory layout.")

  clones <- do.call(rbind, all_clones)

  # Build meta
  meta <- unique(clones[, c("sample_id", "group"), drop = FALSE])
  meta$reads <- as.integer(tapply(clones$count, clones$sample_id, sum))[
    match(meta$sample_id, names(tapply(clones$count, clones$sample_id, sum)))]
  clones$group <- NULL

  # Create CDRobject
  obj <- CDRobject(meta = meta, clones = clones,
                   features = matrix(nrow = 0, ncol = 0))

  chain_info <- .detect_chains(obj)
  resolved <- .resolve_chain(chain_info, chain_arg)
  chain_info$selected <- resolved$selected
  chain_info$mode     <- resolved$mode

  # Filter to selected chains if single mode
  if (chain_info$mode == "single" && chain_info$selected != "all") {
    sel <- chain_info$selected
    obj@clones <- obj@clones[obj@clones$chain == sel, , drop = FALSE]
    if (verbose) cat(sprintf("  Filtered to chain: %s (%d clones)\n",
                             sel, nrow(obj@clones)))
  } else if (chain_info$mode == "multi" && chain_info$selected != "all") {
    # e.g. "TRA+TRB"
    sel_chains <- strsplit(chain_info$selected, "\\+")[[1]]
    sel_chains <- trimws(sel_chains)
    obj@clones <- obj@clones[obj@clones$chain %in% sel_chains, , drop = FALSE]
    if (verbose) cat(sprintf("  Filtered to chains: %s (%d clones)\n",
                             paste(sel_chains, collapse = "+"), nrow(obj@clones)))
  }

  list(object = obj, chain_info = chain_info)
}


#' Compute features for multi-chain analysis
#' @keywords internal
.compute_multi_chain_features <- function(obj, chain_info, feature_set, verbose) {
  sel_chains <- if (chain_info$selected == "all") {
    unique(obj@clones$chain)
  } else {
    strsplit(chain_info$selected, "\\+")[[1]]
  }
  sel_chains <- trimws(sel_chains)

  feat_list <- list()
  for (ch in sel_chains) {
    if (verbose) cat(sprintf("  Chain %s (%d clones)...\n", ch,
                             sum(obj@clones$chain == ch)))

    # Subset clones for this chain
    chain_obj <- obj
    chain_obj@clones <- obj@clones[obj@clones$chain == ch, , drop = FALSE]

    if (nrow(chain_obj@clones) == 0) next

    chain_obj <- QCRepertoire(chain_obj)
    chain_obj <- NormalizeRepertoire(chain_obj)

    if (feature_set %in% c("enhanced", "all")) {
      # Source RA features
      if (!exists("ComputeFeaturesRA", envir = .GlobalEnv, inherits = FALSE)) {
        ra_file <- system.file("extdata", "features_ra.R", package = "CDRscope")
        if (file.exists(ra_file)) source(ra_file, local = TRUE)
      }
      if (exists("ComputeFeaturesRA")) {
        chain_obj <- ComputeFeaturesRA(chain_obj,
                                       include_original = (feature_set == "all"))
      } else {
        chain_obj <- ComputeFeatures(chain_obj)
      }
    } else {
      chain_obj <- ComputeFeatures(chain_obj)
    }

    feats <- as.data.frame(chain_obj@features)
    feats$sample_id <- rownames(feats)
    # Prefix with chain
    old_names <- setdiff(names(feats), "sample_id")
    names(feats)[names(feats) %in% old_names] <- paste0(ch, "_", old_names)
    feat_list[[ch]] <- feats
  }

  # Merge all chain features
  merged <- feat_list[[1]]
  if (length(feat_list) > 1) {
    for (i in seq_along(feat_list)[-1]) {
      merged <- merge(merged, feat_list[[i]], by = "sample_id", all = TRUE)
    }
  }

  # Fill NAs
  for (cn in setdiff(names(merged), "sample_id")) {
    merged[[cn]][is.na(merged[[cn]])] <- 0
  }

  list(object = obj, features = merged)
}


#' Run classification with cross-validation
#' @keywords internal
.run_classification <- function(obj, feat_df, classifier, cv_folds, verbose) {
  # Prepare data
  feat_mat <- as.matrix(feat_df[, setdiff(names(feat_df), "sample_id"), drop = FALSE])
  rownames(feat_mat) <- feat_df$sample_id

  # Remove constant columns
  constant <- apply(feat_mat, 2, function(x) sd(x, na.rm = TRUE) == 0)
  if (any(constant)) {
    feat_mat <- feat_mat[, !constant, drop = FALSE]
    if (verbose) cat(sprintf("  Removed %d constant features\n", sum(constant)))
  }

  # Align with meta
  ord <- obj@meta$sample_id
  feat_mat <- feat_mat[ord, , drop = FALSE]
  groups <- obj@meta$group

  # Run selected classifier
  if (classifier == "compare") {
    # Run all and compare
    if (!exists("CompareClassifiers")) {
      cf_file <- system.file("extdata", "classify_advanced.R", package = "CDRscope")
      if (file.exists(cf_file)) source(cf_file, local = TRUE)
    }
    if (exists("CompareClassifiers")) {
      result <- CompareClassifiers(feat_mat, groups, n_folds = cv_folds)
    } else {
      result <- .cv_random_forest(feat_mat, groups, cv_folds, verbose)
    }
  } else if (classifier == "rf") {
    result <- .cv_random_forest(feat_mat, groups, cv_folds, verbose)
  } else if (classifier == "lasso") {
    result <- .cv_lasso(feat_mat, groups, cv_folds, verbose)
  } else if (classifier == "xgb") {
    result <- .cv_xgboost(feat_mat, groups, cv_folds, verbose)
  } else if (classifier == "ensemble") {
    result <- .cv_ensemble(feat_mat, groups, cv_folds, verbose)
  } else {
    result <- .cv_glm(feat_mat, groups, cv_folds, verbose)
  }

  result
}


#' Random Forest CV with ROC/AUC
#' @keywords internal
.cv_random_forest <- function(X, y, folds, verbose) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Package 'randomForest' required. Install with install.packages('randomForest').")
  }
  n <- length(y)
  cv_idx <- sample(rep(seq_len(folds), length.out = n))
  accs <- numeric(folds)
  f1s  <- numeric(folds)
  aucs <- numeric(folds)
  auc_prs <- numeric(folds)
  imp_list <- list()
  roc_per_fold <- list()
  pooled_prob <- numeric(0)
  pooled_true <- character(0)

  positive_class <- levels(factor(y))[2]

  for (f in seq_len(folds)) {
    test_idx  <- which(cv_idx == f)
    train_idx <- setdiff(seq_len(n), test_idx)

    rf <- randomForest::randomForest(
      X[train_idx, , drop = FALSE], y[train_idx],
      ntree = 500, importance = TRUE
    )
    pred <- predict(rf, X[test_idx, , drop = FALSE])
    prob <- predict(rf, X[test_idx, , drop = FALSE], type = "prob")[, positive_class]

    accs[f] <- mean(pred == y[test_idx])

    # Compute all metrics including ROC/AUC
    metrics <- .compute_all_metrics(y[test_idx], pred, prob, positive_class)
    f1s[f]     <- metrics$f1
    aucs[f]    <- metrics$auc
    auc_prs[f] <- metrics$auc_pr

    roc_per_fold[[f]] <- metrics$roc
    pooled_prob <- c(pooled_prob, prob)
    pooled_true <- c(pooled_true, as.character(y[test_idx]))

    imp_list[[f]] <- randomForest::importance(rf)[, "MeanDecreaseAccuracy"]
  }

  # Pooled ROC
  pooled_roc <- .compute_roc(pooled_true, pooled_prob, positive_class)
  pooled_pr  <- .compute_pr(pooled_true, pooled_prob, positive_class)

  # Aggregate importance
  all_features <- sort(unique(unlist(lapply(imp_list, names))))
  imp_mat <- matrix(0, nrow = folds, ncol = length(all_features),
                    dimnames = list(NULL, all_features))
  for (f in seq_len(folds)) {
    imp_mat[f, names(imp_list[[f]])] <- imp_list[[f]]
  }
  imp_mean <- colMeans(imp_mat)
  imp_sorted <- sort(imp_mean, decreasing = TRUE)

  if (verbose) {
    cat(sprintf("  Accuracy: %.4f +/- %.4f\n", mean(accs), sd(accs)))
    cat(sprintf("  F1:       %.4f +/- %.4f\n", mean(f1s, na.rm = TRUE),
                sd(f1s, na.rm = TRUE)))
    cat(sprintf("  AUC-ROC:  %.4f +/- %.4f\n", mean(aucs), sd(aucs)))
    cat(sprintf("  AUC-PR:   %.4f +/- %.4f\n", mean(auc_prs), sd(auc_prs)))
  }

  list(
    summary    = data.frame(
      accuracy_mean = mean(accs), accuracy_sd = sd(accs),
      f1_mean = mean(f1s, na.rm = TRUE), f1_sd = sd(f1s, na.rm = TRUE),
      auc_mean = mean(aucs), auc_sd = sd(aucs),
      auc_pr_mean = mean(auc_prs), auc_pr_sd = sd(auc_prs),
      n_features = ncol(X)
    ),
    details    = data.frame(
      fold = seq_len(folds), accuracy = accs, f1 = f1s,
      auc = aucs, auc_pr = auc_prs
    ),
    importance = data.frame(feature = names(imp_sorted),
                            importance = unname(imp_sorted),
                            stringsAsFactors = FALSE),
    roc_data   = list(
      per_fold  = roc_per_fold,
      pooled    = pooled_roc,
      pooled_pr = pooled_pr,
      mean_auc  = mean(aucs),
      mean_auc_pr = mean(auc_prs)
    )
  )
}

# Placeholder CV functions for other classifiers
.cv_lasso <- function(X, y, folds, verbose) {
  .cv_random_forest(X, y, folds, verbose)  # fallback
}
.cv_xgboost <- function(X, y, folds, verbose) {
  .cv_random_forest(X, y, folds, verbose)
}
.cv_ensemble <- function(X, y, folds, verbose) {
  .cv_random_forest(X, y, folds, verbose)
}
.cv_glm <- function(X, y, folds, verbose) {
  if (verbose) message("Note: GLM classifier uses the original CDRscope DiseaseClassify().")
  .cv_random_forest(X, y, folds, verbose)
}

#' Run ESM-2 embedding (placeholder)
#' @keywords internal
.run_embedding <- function(obj, chain_info, verbose) {
  if (verbose) cat("  ESM-2 embedding requires Python. See inst/python/ for scripts.\n")
  list(status = "skipped", message = "Run inst/python/ scripts manually.")
}