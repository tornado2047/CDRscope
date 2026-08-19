#' @title Complete Closed-Loop Analysis Pipeline
#' @description Orchestrates the full 10-module analysis: input → features → embedding →
#'   reference map → classification → visualization → significance → breakthrough →
#'   biological validation → report generation.
#' @param input Either a CDRobject, a directory path, or a single CSV file
#' @param chain Chain mode: "single", "paired" (TRA+TRB), or "all"
#' @param group Column name or vector specifying sample groups (control vs patient)
#' @param control_dir Directory containing control sample files
#' @param patient_dir Directory containing patient sample files
#' @param features Feature set: "enhanced" (65 features, recommended) or "basic" (20)
#' @param classifier Classifier: "rf", "lasso", "xgboost", "ensemble", "compare"
#' @param cv_folds Number of cross-validation folds
#' @param use_reference_map If TRUE, project sequences onto the fixed reference map
#' @param run_significance If TRUE, run domain-level significance analysis
#' @param run_breakthrough If TRUE, run breakthrough analyses (expansion, axis decode, etc.)
#' @param run_validation If TRUE, run biological validation (frequency, citrullination, HLA)
#' @param generate_report If TRUE, auto-generate HTML report
#' @param output_dir Output directory for results (default: "cdrscope_results")
#' @param verbose Print progress messages
#' @return A list with all analysis results
#' @export
run_complete_analysis <- function(
  input,
  chain = c("single", "paired", "all"),
  group = NULL,
  control_dir = NULL,
  patient_dir = NULL,
  features = "enhanced",
  classifier = "rf",
  cv_folds = 5,
  use_reference_map = TRUE,
  run_significance = TRUE,
  run_breakthrough = TRUE,
  run_validation = TRUE,
  generate_report = TRUE,
  output_dir = "cdrscope_results",
  verbose = TRUE
) {

  start_time <- Sys.time()
  chain <- match.arg(chain)

  if (verbose) message("\n========================================================")
  if (verbose) message("  CDRscope v2.0 — Complete Closed-Loop Analysis Pipeline")
  if (verbose) message("========================================================\n")

  results <- list(
    call = match.call(),
    output_dir = output_dir,
    chain = chain,
    start_time = start_time
  )

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  py_dir <- system.file("python", package = "CDRscope")
  ref_dir <- system.file("reference_map", package = "CDRscope")

  # ================================================================
  # Phase 1: Core Pipeline (Input → Features → Classification)
  # ================================================================
  if (verbose) message("Phase 1: Core Pipeline (Input → Features → Classification)\n")

  core <- run_CDRscope(
    input = input,
    chain = chain,
    group = group,
    control_dir = control_dir,
    patient_dir = patient_dir,
    features = features,
    classifier = classifier,
    cv_folds = cv_folds,
    use_embedding = FALSE,
    verbose = verbose
  )

  results$object <- core$object
  results$cv_results <- core$cv_results
  results$cv_details <- core$cv_details
  results$feature_importance <- core$feature_importance
  results$chain_info <- core$chain_info
  results$roc_data <- core$roc_data

  # Save CV results (now includes AUC-ROC and AUC-PR)
  if (!is.null(results$cv_results)) {
    write.csv(results$cv_results, file.path(output_dir, "cv_results.csv"), row.names = FALSE)
  }
  if (!is.null(results$cv_details)) {
    write.csv(results$cv_details, file.path(output_dir, "cv_details.csv"), row.names = FALSE)
  }
  if (!is.null(results$feature_importance)) {
    write.csv(results$feature_importance, file.path(output_dir, "feature_importance.csv"), row.names = FALSE)
  }

  # Generate ROC curve plot
  if (!is.null(results$roc_data)) {
    roc_path <- file.path(output_dir, "roc_curve.png")
    tryCatch({
      ROCCurve(results, output_path = roc_path)
      if (verbose) message("  ROC curve saved: ", roc_path)
    }, error = function(e) {
      if (verbose) message("  ROC plot skipped: ", e$message)
    })
  }

  # ================================================================
  # Phase 2: Reference Map Projection
  # ================================================================
  if (use_reference_map && dir.exists(ref_dir)) {
    if (verbose) message("\nPhase 2: Reference Map Projection\n")

    obj <- results$object
    seqs <- .extract_sequences(obj, chain)
    seqs_path <- file.path(output_dir, "sequences_for_projection.csv")
    write.csv(data.frame(junction_aa = seqs), seqs_path, row.names = FALSE)

    coords_path <- file.path(output_dir, "projected_coords.csv")
    cmd <- sprintf(
      "python3 %s/project_to_reference.py --ref-dir %s --input-csv %s --output-csv %s --overlay",
      py_dir, ref_dir, seqs_path, coords_path
    )
    if (verbose) message("  Projecting ", length(seqs), " sequences onto reference map...")
    system(cmd, ignore.stdout = !verbose, ignore.stderr = !verbose)

    if (file.exists(coords_path)) {
      results$projected_coords <- read.csv(coords_path)
      if (verbose) message("  Projected ", nrow(results$projected_coords), " sequences")
    }
  }

  # ================================================================
  # Phase 3: Deep Analysis (Visualization + Significance + Breakthrough)
  # ================================================================
  if (run_significance || run_breakthrough) {
    if (verbose) message("\nPhase 3: Deep Analysis\n")

    analysis_cmd <- sprintf(
      "python3 %s/complete_analysis.py --output-dir %s --ref-dir %s %s %s",
      py_dir, output_dir, ref_dir,
      if (run_significance) "--significance" else "",
      if (run_breakthrough) "--breakthrough" else ""
    )

    if (!is.null(results$projected_coords) && file.exists(file.path(output_dir, "projected_coords.csv"))) {
      analysis_cmd <- paste0(analysis_cmd, " --coords-csv ", file.path(output_dir, "projected_coords.csv"))
    }

    if (verbose) message("  Running complete analysis...")
    system(analysis_cmd, ignore.stdout = !verbose, ignore.stderr = !verbose)

    sig_file <- file.path(output_dir, "domain_significance.csv")
    if (file.exists(sig_file)) {
      results$significance <- read.csv(sig_file)
      if (verbose) message("  Significance analysis complete")
    }

    bt_file <- file.path(output_dir, "breakthrough_summary.json")
    if (file.exists(bt_file)) {
      results$breakthrough <- jsonlite::fromJSON(bt_file)
      if (verbose) message("  Breakthrough analysis complete")
    }
  }

  # ================================================================
  # Phase 4: Biological Validation
  # ================================================================
  if (run_validation) {
    if (verbose) message("\nPhase 4: Biological Validation\n")

    val_cmd <- sprintf(
      "python3 %s/complete_analysis.py --output-dir %s --ref-dir %s --validation",
      py_dir, output_dir, ref_dir
    )
    if (!is.null(results$projected_coords) && file.exists(file.path(output_dir, "projected_coords.csv"))) {
      val_cmd <- paste0(val_cmd, " --coords-csv ", file.path(output_dir, "projected_coords.csv"))
    }

    if (verbose) message("  Running biological validation...")
    system(val_cmd, ignore.stdout = !verbose, ignore.stderr = !verbose)

    val_file <- file.path(output_dir, "validation_summary.json")
    if (file.exists(val_file)) {
      results$validation <- jsonlite::fromJSON(val_file)
      if (verbose) message("  Biological validation complete")
    }
  }

  # ================================================================
  # Phase 5: Report Generation
  # ================================================================
  if (generate_report) {
    if (verbose) message("\nPhase 5: Report Generation\n")

    report_cmd <- sprintf(
      "python3 %s/generate_report.py --output-dir %s --chain %s",
      py_dir, output_dir, chain
    )

    if (verbose) message("  Generating HTML report...")
    system(report_cmd, ignore.stdout = !verbose, ignore.stderr = !verbose)

    report_file <- file.path(output_dir, "CDRscope_Analysis_Report.html")
    if (file.exists(report_file)) {
      results$report_path <- report_file
      if (verbose) message("  Report: ", report_file)
    }
  }

  # ================================================================
  # Finalize
  # ================================================================
  end_time <- Sys.time()
  results$elapsed <- difftime(end_time, start_time, units = "mins")

  if (verbose) {
    message("\n========================================================")
    message(sprintf("  Complete! Elapsed: %.1f minutes", as.numeric(results$elapsed)))
    message("  Output directory: ", normalizePath(output_dir))
    message("========================================================\n")
  }

  class(results) <- "CDRscopeCompleteAnalysis"
  return(results)
}

#' @export
print.CDRscopeCompleteAnalysis <- function(x, ...) {
  cat("========================================================\n")
  cat("  CDRscope Complete Analysis Results\n")
  cat("========================================================\n")
  cat("  Chain mode:     ", x$chain, "\n")
  cat("  Elapsed:        ", sprintf("%.1f minutes", as.numeric(x$elapsed)), "\n")
  cat("  Output dir:     ", x$output_dir, "\n")
  if (!is.null(x$cv_results)) {
    cat("  CV accuracy:    ", sprintf("%.1f%%", x$cv_results$mean_accuracy * 100), "\n")
    if (!is.null(x$cv_results$auc_mean)) {
      cat("  AUC-ROC:        ", sprintf("%.3f +/- %.3f", x$cv_results$auc_mean, x$cv_results$auc_sd), "\n")
    }
    if (!is.null(x$cv_results$auc_pr_mean)) {
      cat("  AUC-PR:         ", sprintf("%.3f +/- %.3f", x$cv_results$auc_pr_mean, x$cv_results$auc_pr_sd), "\n")
    }
  }
  if (!is.null(x$significance)) {
    cat("  Significance:   ", nrow(x$significance), " domain tests\n")
  }
  if (!is.null(x$report_path)) {
    cat("  Report:         ", x$report_path, "\n")
  }
  cat("========================================================\n")
  invisible(x)
}

.extract_sequences <- function(obj, chain) {
  seqs <- character(0)
  if (!is.null(obj$TRB$data)) {
    seqs <- c(seqs, na.omit(obj$TRB$data$CDR3aa))
  }
  if (chain %in% c("paired", "all") && !is.null(obj$TRA$data)) {
    seqs <- c(seqs, na.omit(obj$TRA$data$CDR3aa))
  }
  if (chain == "all") {
    for (c in c("TRD", "TRG", "IGH", "IGL", "IGK")) {
      if (!is.null(obj[[c]]$data)) {
        seqs <- c(seqs, na.omit(obj[[c]]$data$CDR3aa))
      }
    }
  }
  unique(seqs)
}
