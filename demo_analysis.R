cat("=== CDRscope analysis demo (with figures) ===\n\n")
suppressPackageStartupMessages(library(CDRscope))
library(ggplot2)

figdir <- "docs/figures"
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

cat("1. Build toy repertoire (4 disease groups)\n")
obj <- fetch_toy_data(n_samples = 40, n_clones_per_sample = 300, seed = 1)
print(obj)

cat("\n2. QC + Normalize\n")
obj <- QCRepertoire(obj) |> NormalizeRepertoire()
cat("   clones kept:", nrow(obj@clones), "\n")

cat("\n3. Compute six-concept features\n")
obj <- ComputeFeatures(obj, verbose = FALSE)
cat("   features:", ncol(obj@features), "| modules:",
    paste(names(obj@feature_modules), collapse = ", "), "\n")

cat("\n4. Concept-bottleneck embedding\n")
obj <- ConceptBottleneckEmbed(obj, ndim = 10)

cat("\n5. Disease classification + SHAP\n")
obj <- DiseaseClassify(obj, use_shap = TRUE, verbose = FALSE)
cat("   accuracy:", round(obj@classification$train_accuracy, 3), "\n")

cat("\n6. Marker discovery\n")
obj <- FindMarkers(obj, level = "both")
cat("   stat sig:", nrow(obj@markers$statistical_sig),
    "| seq sig:", nrow(obj@markers$sequence_sig), "\n")

cat("\n7. Saving figures to ", figdir, "/\n", sep = "")
ggsave(file.path(figdir, "01_dimplot.png"), DimPlot(obj),
       width = 6, height = 4.4, dpi = 150)
ggsave(file.path(figdir, "02_featureplot_convergence.png"),
       FeaturePlot(obj, "convergence"), width = 6, height = 4.4, dpi = 150)
ggsave(file.path(figdir, "03_featureplot_shannon.png"),
       FeaturePlot(obj, "shannon"), width = 6, height = 4.4, dpi = 150)
ggsave(file.path(figdir, "04_vlnplot_shannon.png"),
       VlnPlot(obj, "shannon"), width = 5.5, height = 4, dpi = 150)
ggsave(file.path(figdir, "05_vlnplot_convergence.png"),
       VlnPlot(obj, "convergence"), width = 5.5, height = 4, dpi = 150)
ggsave(file.path(figdir, "06_shapplot.png"), SHAPPlot(obj),
       width = 6.5, height = 4.5, dpi = 150)
ggsave(file.path(figdir, "07_heatmap.png"), FeatureHeatmap(obj),
       width = 7, height = 5, dpi = 150)
ggsave(file.path(figdir, "08_motiflogo_infection.png"),
       MotifLogo(obj, group = "infection"), width = 6, height = 4.5, dpi = 150)
ggsave(file.path(figdir, "09_motiflogo_autoimmune.png"),
       MotifLogo(obj, group = "autoimmune"), width = 6, height = 4.5, dpi = 150)
cat("   saved 9 figures\n")

cat("\n8. Export marker tables\n")
write.csv(obj@markers$statistical_sig, file.path(figdir, "..", "statistical_markers.csv"),
          row.names = FALSE)
write.csv(head(obj@markers$sequence_sig, 100), file.path(figdir, "..", "sequence_markers_top100.csv"),
          row.names = FALSE)
cat("   saved statistical_markers.csv + sequence_markers_top100.csv\n")

cat("\n=== DONE — figures in docs/figures/, tables in docs/ ===\n")
