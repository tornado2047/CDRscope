cat("=== CDRscope end-to-end demo ===\n\n")
suppressPackageStartupMessages(library(CDRscope))

cat("1. Build toy repertoire\n")
obj <- fetch_toy_data(n_samples = 40, n_clones_per_sample = 300, seed = 1)
print(obj)

cat("\n2. QC + Normalize\n")
obj <- QCRepertoire(obj)
obj <- NormalizeRepertoire(obj)
cat("   removed:", obj@misc$qc$removed, "| kept:", obj@misc$qc$kept, "\n")

cat("\n3. Compute six-concept features\n")
obj <- ComputeFeatures(obj, verbose = FALSE)
cat("   feature matrix:", nrow(obj@features), "samples x",
    ncol(obj@features), "features\n")
cat("   modules:", paste(names(obj@feature_modules), collapse=", "), "\n")

cat("\n4. Concept-bottleneck embedding\n")
obj <- ConceptBottleneckEmbed(obj, ndim = 10)
cat("   embedding:", paste(dim(obj@embedding), collapse=" x "), "\n")

cat("\n5. Disease classification (logistic + SHAP)\n")
obj <- DiseaseClassify(obj, method = "logistic", use_shap = TRUE, verbose = FALSE)
cat("   classes:", paste(obj@classification$classes, collapse=", "), "\n")
cat("   train accuracy:", round(obj@classification$train_accuracy, 3), "\n")
cat("   shap method:", obj@classification$shap$method, "\n")

cat("\n6. Marker discovery (two layers)\n")
obj <- FindMarkers(obj, level = "both")
cat("   statistical-level markers:", nrow(obj@markers$statistical),
    "| sig (FDR<0.05):", nrow(obj@markers$statistical_sig), "\n")
cat("   sequence-level motifs:", nrow(obj@markers$sequence),
    "| sig:", nrow(obj@markers$sequence_sig), "\n")
cat("   public disease clonotypes:", nrow(obj@markers$public_clonotypes), "\n")

cat("\n7. Top statistical markers\n")
print(head(obj@markers$statistical, 6))

cat("\n8. Top sequence markers (enriched motifs)\n")
print(head(obj@markers$sequence_sig[order(-obj@markers$sequence_sig$odds_ratio), ], 6))

cat("\n9. Visualisation smoke test\n")
p1 <- DimPlot(obj); cat("   DimPlot:", class(p1)[1], "\n")
p2 <- FeaturePlot(obj, "convergence"); cat("   FeaturePlot:", class(p2)[1], "\n")
p3 <- VlnPlot(obj, "shannon"); cat("   VlnPlot:", class(p3)[1], "\n")
p4 <- SHAPPlot(obj); cat("   SHAPPlot:", class(p4)[1], "\n")
p5 <- FeatureHeatmap(obj); cat("   FeatureHeatmap:", class(p5)[1], "\n")
p6 <- MotifLogo(obj, group = "infection"); cat("   MotifLogo:", class(p6)[1], "\n")

cat("\n=== ALL OK ===\n")
