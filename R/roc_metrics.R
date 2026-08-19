# ===========================================================================
# CDRscope v2.1 — ROC / AUC / PR metrics
# Adds ROC curve, AUC, PR curve, and sensitivity/specificity to the
# cross-validation evaluation.
# ===========================================================================

#' Compute ROC curve and AUC from binary predictions
#'
#' @param y_true True labels (factor or character, 2 levels)
#' @param y_prob Predicted probability for the positive class
#' @param positive_class Character; which level is "positive". If NULL,
#'   uses the second level of factor(y_true).
#' @return List with: auc, tpr, fpr, thresholds, youden (best threshold)
#' @keywords internal
.compute_roc <- function(y_true, y_prob, positive_class = NULL) {
  y <- factor(y_true)
  if (is.null(positive_class)) {
    positive_class <- levels(y)[2]
  }
  y_bin <- as.integer(y == positive_class)

  n_pos <- sum(y_bin == 1)
  n_neg <- sum(y_bin == 0)
  if (n_pos == 0 || n_neg == 0) {
    return(list(auc = 0.5, tpr = c(0, 1), fpr = c(0, 1),
                thresholds = c(1, 0), youden = 0.5,
                sensitivity = 0, specificity = 1))
  }

  ord <- order(y_prob, decreasing = TRUE)
  y_sorted <- y_bin[ord]
  prob_sorted <- y_prob[ord]

  cum_pos <- cumsum(y_sorted)
  cum_neg <- cumsum(1 - y_sorted)

  tpr <- c(0, cum_pos / n_pos)
  fpr <- c(0, cum_neg / n_neg)
  thresholds <- c(1, prob_sorted)

  # Remove duplicate FPR points for clean AUC
  dup <- duplicated(fpr)
  if (any(dup)) {
    keep <- !dup | c(FALSE, !dup[-length(dup)])
    fpr_u <- fpr[keep]
    tpr_u <- tpr[keep]
  } else {
    fpr_u <- fpr
    tpr_u <- tpr
  }

  # AUC via trapezoidal rule
  auc <- sum(diff(fpr_u) * (tpr_u[-1] + tpr_u[-length(tpr_u)]) / 2)
  auc <- max(0, min(1, auc))

  # Youden's J statistic for optimal threshold
  youden_idx <- which.max(tpr - fpr)

  list(
    auc         = auc,
    tpr         = tpr,
    fpr         = fpr,
    thresholds  = thresholds,
    youden      = thresholds[youden_idx],
    sensitivity = tpr[youden_idx],
    specificity = 1 - fpr[youden_idx]
  )
}

#' Compute Precision-Recall curve and AUC
#'
#' @param y_true True labels
#' @param y_prob Predicted probability for positive class
#' @param positive_class Character; positive level
#' @return List with: auc_pr, precision, recall, thresholds
#' @keywords internal
.compute_pr <- function(y_true, y_prob, positive_class = NULL) {
  y <- factor(y_true)
  if (is.null(positive_class)) {
    positive_class <- levels(y)[2]
  }
  y_bin <- as.integer(y == positive_class)

  n_pos <- sum(y_bin == 1)
  if (n_pos == 0) {
    return(list(auc_pr = 0, precision = c(0, 1), recall = c(0, 0)))
  }

  ord <- order(y_prob, decreasing = TRUE)
  y_sorted <- y_bin[ord]

  cum_tp <- cumsum(y_sorted)
  cum_fp <- cumsum(1 - y_sorted)

  precision <- cum_tp / (cum_tp + cum_fp)
  recall    <- cum_tp / n_pos

  # Prepend (0, 1) for precision, (0, 0) for recall
  precision <- c(1, precision)
  recall    <- c(0, recall)

  # AUC via trapezoidal rule (recall on x-axis)
  auc_pr <- sum(diff(recall) * (precision[-1] + precision[-length(precision)]) / 2)
  auc_pr <- max(0, min(1, auc_pr))

  list(auc_pr = auc_pr, precision = precision, recall = recall)
}

#' Compute all binary classification metrics from CV results
#'
#' @param y_true Vector of true labels
#' @param y_pred Vector of predicted labels
#' @param y_prob Vector of predicted probabilities for positive class
#' @param positive_class Character; positive level
#' @return List with accuracy, f1, auc, auc_pr, sensitivity, specificity,
#'   precision, recall, roc, pr
#' @keywords internal
.compute_all_metrics <- function(y_true, y_pred, y_prob, positive_class = NULL) {
  y <- factor(y_true)
  if (is.null(positive_class)) {
    positive_class <- levels(y)[2]
  }

  y_bin  <- as.integer(y == positive_class)
  pred_bin <- as.integer(factor(y_pred) == positive_class)

  tp <- sum(y_bin == 1 & pred_bin == 1)
  fp <- sum(y_bin == 0 & pred_bin == 1)
  fn <- sum(y_bin == 1 & pred_bin == 0)
  tn <- sum(y_bin == 0 & pred_bin == 0)

  accuracy    <- (tp + tn) / (tp + fp + fn + tn)
  precision   <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall      <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1          <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  specificity <- if (tn + fp > 0) tn / (tn + fp) else 0

  roc <- .compute_roc(y_true, y_prob, positive_class)
  pr  <- .compute_pr(y_true, y_prob, positive_class)

  list(
    accuracy    = accuracy,
    f1          = f1,
    precision   = precision,
    recall      = recall,
    sensitivity = recall,
    specificity = specificity,
    auc         = roc$auc,
    auc_pr      = pr$auc_pr,
    youden      = roc$youden,
    roc         = roc,
    pr          = pr,
    confusion   = matrix(c(tn, fp, fn, tp), nrow = 2,
                         dimnames = list(c("Neg", "Pos"), c("Neg", "Pos")))
  )
}


#' Plot ROC curves from cross-validation results
#'
#' @param cv_results A results list from \code{run_CDRscope()} or
#'   \code{run_complete_analysis()} that contains \code{roc_data}.
#' @param output_path File path to save the plot (PNG). If NULL, returns the
#'   ggplot object.
#' @return Invisibly returns a ggplot object. Saves a PNG if output_path is set.
#' @export
#' @importFrom grDevices png dev.off
ROCCurve <- function(cv_results, output_path = NULL) {
  roc_data <- NULL
  if (!is.null(cv_results$roc_data)) {
    roc_data <- cv_results$roc_data
  } else if (!is.null(cv_results$cv_results$roc_data)) {
    roc_data <- cv_results$cv_results$roc_data
  } else {
    stop("No ROC data found. Run classification with cv_folds > 1.")
  }

  # Build data frame for plotting
  plot_df <- data.frame()
  for (f in seq_along(roc_data$per_fold)) {
    fold_roc <- roc_data$per_fold[[f]]
    plot_df <- rbind(plot_df, data.frame(
      fold = paste0("Fold ", f),
      fpr  = fold_roc$fpr,
      tpr  = fold_roc$tpr
    ))
  }

  # Add pooled ROC
  if (!is.null(roc_data$pooled)) {
    plot_df <- rbind(plot_df, data.frame(
      fold = "Pooled",
      fpr  = roc_data$pooled$fpr,
      tpr  = roc_data$pooled$tpr
    ))
  }

  p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = fpr, y = tpr, color = fold)) +
    ggplot2::geom_line(size = 0.8, alpha = 0.7) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                         color = "grey50", alpha = 0.5) +
    ggplot2::scale_color_manual(values = c(
      rep("#3D6A8C", length(roc_data$per_fold)),
      "#D97742"
    )) +
    ggplot2::labs(
      title = sprintf("ROC Curve (AUC = %.3f)", roc_data$mean_auc),
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)",
      color = ""
    ) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      legend.position = "bottom",
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(hjust = 0.5)
    ) +
    ggplot2::coord_equal()

  if (!is.null(output_path)) {
    ggplot2::ggsave(output_path, p, width = 6, height = 6, dpi = 150)
    message(sprintf("ROC curve saved: %s", output_path))
  }

  invisible(p)
}
