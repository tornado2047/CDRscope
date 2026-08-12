# ---- colour helpers (internal) -------------------------------------------
.group_pal <- function(n) {
  pals <- c("#378ADD", "#1D9E75", "#D85A30", "#534AB7",
            "#BA7517", "#D4537E", "#639922", "#0C447C")
  if (n <= length(pals)) pals[seq_len(n)] else
    grDevices::colorRampPalette(pals)(n)
}

.resolve_feature <- function(object, feature) {
  mods <- object@feature_modules
  feat <- object@features
  if (feature %in% names(mods)) {
    list(type = "module", cols = mods[[feature]], label = paste0(feature, " (module mean)"))
  } else if (feature %in% colnames(feat)) {
    list(type = "feature", cols = which(colnames(feat) == feature), label = feature)
  } else {
    m <- grep(feature, colnames(feat), value = TRUE)
    if (length(m) == 1) {
      list(type = "feature", cols = which(colnames(feat) == m), label = m)
    } else if (length(m) > 1) {
      stop("Ambiguous feature '", feature, "': matches ", paste(m, collapse = ", "))
    } else {
      stop("Feature not found: ", feature)
    }
  }
}

#' @title Dimension plot of the concept-bottleneck space
#'
#' @description Scatter of samples in the 2-D concept-bottleneck reduction,
#' coloured by disease group. Mirrors \code{Seurat::DimPlot}.
#'
#' @param object A \code{\link{CDRobject}} with reduction populated.
#' @param group_by Character; meta column to colour by. Default \code{"group"}.
#' @param label Logical; label groups. Default \code{TRUE}.
#'
#' @return A \code{ggplot} object.
#' @export
#' @import ggplot2
DimPlot <- function(object, group_by = "group", label = TRUE) {
  stopifnot(inherits(object, "CDRobject"))
  red <- object@reduction
  if (ncol(red) < 2) stop("Run ConceptBottleneckEmbed() first.")
  df <- data.frame(CB1 = red[, 1], CB2 = red[, 2],
                   group = object@meta[[group_by]])
  df <- df[!is.na(df$group), , drop = FALSE]
  n <- length(unique(df$group))
  p <- ggplot(df, aes(.data$CB1, .data$CB2, colour = .data$group)) +
    geom_point(size = 2.4, alpha = 0.85) +
    scale_colour_manual(values = .group_pal(n)) +
    theme_classic(base_size = 12) +
    labs(x = "Concept axis 1", y = "Concept axis 2",
         colour = group_by, title = "Concept-bottleneck space")
  if (label) {
    cen <- do.call(rbind, lapply(split(df, df$group), function(d)
      data.frame(group = d$group[1], CB1 = mean(d$CB1), CB2 = mean(d$CB2))))
    p <- p + geom_text(data = cen, aes(label = .data$group),
                       size = 4, fontface = "bold",
                       show.legend = FALSE, vjust = -1.2)
  }
  p
}

#' @title Feature plot on the concept space
#'
#' @description Scatter of samples coloured by a feature's value, mirroring
#' \code{Seurat::FeaturePlot}.
#'
#' @param object A \code{\link{CDRobject}}.
#' @param feature Character; feature name (column of \code{features}) or a
#'   concept module name.
#'
#' @return A \code{ggplot} object.
#' @export
#' @import ggplot2
FeaturePlot <- function(object, feature) {
  stopifnot(inherits(object, "CDRobject"))
  red <- object@reduction
  feat <- object@features
  rf <- .resolve_feature(object, feature)
  val <- if (rf$type == "module")
    rowMeans(feat[, rf$cols, drop = FALSE], na.rm = TRUE) else
    feat[, rf$cols]
  df <- data.frame(CB1 = red[, 1], CB2 = red[, 2], value = val)
  ggplot(df, aes(.data$CB1, .data$CB2, colour = .data$value)) +
    geom_point(size = 2.6) +
    scale_colour_gradient2(low = "#378ADD", mid = "#F1EFE8", high = "#D85A30",
                           midpoint = median(val, na.rm = TRUE)) +
    theme_classic(base_size = 12) +
    labs(x = "Concept axis 1", y = "Concept axis 2",
         colour = rf$label, title = feature)
}

#' @title Violin plot of a feature by group
#'
#' @description Mirrors \code{Seurat::VlnPlot}.
#'
#' @param object A \code{\link{CDRobject}}.
#' @param feature Character; feature name or module name.
#'
#' @return A \code{ggplot} object.
#' @export
#' @import ggplot2
VlnPlot <- function(object, feature) {
  stopifnot(inherits(object, "CDRobject"))
  feat <- object@features
  rf <- .resolve_feature(object, feature)
  val <- if (rf$type == "module")
    rowMeans(feat[, rf$cols, drop = FALSE], na.rm = TRUE) else
    feat[, rf$cols]
  df <- data.frame(value = val, group = object@meta$group)
  n <- length(unique(df$group))
  ggplot(df, aes(.data$group, .data$value, fill = .data$group)) +
    geom_violin(alpha = 0.7, trim = FALSE) +
    geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA) +
    scale_fill_manual(values = .group_pal(n)) +
    theme_classic(base_size = 12) +
    labs(y = rf$label, x = NULL, title = feature) +
    theme(legend.position = "none")
}

#' @title SHAP / attribution plot
#'
#' @description Horizontal bar chart of feature attribution to the disease
#' class, from the stored classifier. Positive (coral) pushes toward disease,
#' negative (teal) toward healthy.
#'
#' @param object A \code{\link{CDRobject}} with classification populated.
#' @param class Character; which class to show. Default first disease class.
#' @param top_n Integer; top features to show. Default 15.
#'
#' @return A \code{ggplot} object.
#' @export
#' @import ggplot2
SHAPPlot <- function(object, class = NULL, top_n = 15) {
  stopifnot(inherits(object, "CDRobject"))
  cls <- object@classification
  if (length(cls) == 0) stop("Run DiseaseClassify() first.")
  if (is.null(class)) class <- setdiff(cls$classes, "healthy")[1]
  shap <- cls$shap
  if (is.null(shap)) stop("No attribution available.")
  if (!is.null(shap$mean_abs_contrib)) {
    contrib <- shap$mean_abs_contrib[, class, drop = TRUE]
    df <- data.frame(feature = names(contrib), value = as.numeric(contrib))
  } else if (!is.null(shap$coefs)) {
    coefs <- shap$coefs[-1, class, drop = TRUE]
    std <- apply(object@embedding, 2, sd)
    df <- data.frame(feature = names(coefs),
                     value = as.numeric(coefs) * std)
  } else stop("Attribution format not recognised.")
  df <- df[order(-abs(df$value)), , drop = FALSE]
  df <- head(df, top_n)
  df$feature <- factor(df$feature, levels = rev(df$feature))
  ggplot(df, aes(.data$value, .data$feature, fill = .data$value > 0)) +
    geom_col() +
    scale_fill_manual(values = c("FALSE" = "#1D9E75", "TRUE" = "#D85A30"),
                      labels = c("toward healthy", "toward disease"),
                      name = NULL) +
    geom_vline(xintercept = 0, colour = "grey50") +
    theme_classic(base_size = 12) +
    labs(x = sprintf("Attribution to %s", class), y = NULL,
         title = sprintf("SHAP / coefficient attribution — %s", class))
}

#' @title Motif enrichment logo
#'
#' @description Simple position-frequency logo for enriched motifs of a given
#' disease group, drawn without external logo packages.
#'
#' @param object A \code{\link{CDRobject}} with sequence markers.
#' @param group Character; disease group.
#' @param top_n Integer; top enriched motifs to display. Default 20.
#'
#' @return A \code{ggplot} object.
#' @export
#' @import ggplot2
MotifLogo <- function(object, group, top_n = 20) {
  stopifnot(inherits(object, "CDRobject"))
  seq <- object@markers$sequence
  if (is.null(seq) || nrow(seq) == 0) stop("Run FindMarkers(level='sequence').")
  seq <- seq[seq$group == group & seq$odds_ratio > 1, , drop = FALSE]
  seq <- head(seq[order(-seq$odds_ratio), , drop = FALSE], top_n)
  if (nrow(seq) == 0) stop("No enriched motifs for group: ", group)
  seq$motif <- factor(seq$motif, levels = rev(seq$motif))
  ggplot(seq, aes(.data$odds_ratio, .data$motif, fill = -log10(.data$padj))) +
    geom_col() +
    scale_fill_gradient(low = "#FAEEDA", high = "#D85A30",
                        name = expression(-log[10](FDR))) +
    theme_classic(base_size = 12) +
    labs(x = "Odds ratio (enrichment)", y = NULL,
         title = sprintf("Enriched CDR3 motifs — %s", group))
}

#' @title Feature heatmap
#'
#' @description Sample-by-feature heatmap, grouped by disease, z-scored per
#' feature. Mirrors the Seurat DoHeatmap idiom.
#'
#' @param object A \code{\link{CDRobject}}.
#' @param features Character; optional subset of feature names.
#'
#' @return A \code{ggplot} object.
#' @export
#' @import ggplot2
FeatureHeatmap <- function(object, features = NULL) {
  stopifnot(inherits(object, "CDRobject"))
  feat <- object@features
  if (is.null(features)) features <- colnames(feat)
  feat <- feat[, features, drop = FALSE]
  z <- scale(feat)
  z[is.na(z)] <- 0
  ord <- order(object@meta$group)
  z <- z[ord, , drop = FALSE]
  df <- expand.grid(feature = colnames(z), sample = rownames(z),
                    KEEP.OUT.ATTRS = FALSE)
  df$value <- as.vector(z)
  df$group <- rep(object@meta$group[ord], each = ncol(z))
  ggplot(df, aes(.data$sample, .data$feature, fill = .data$value)) +
    geom_raster() +
    scale_fill_gradient2(low = "#378ADD", mid = "#F1EFE8", high = "#D85A30",
                         midpoint = 0, name = "z-score") +
    theme_classic(base_size = 11) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
    labs(x = NULL, y = NULL, title = "Concept feature heatmap")
}
