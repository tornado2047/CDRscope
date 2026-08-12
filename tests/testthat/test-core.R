test_that("toy data builds a CDRobject", {
  obj <- fetch_toy_data(n_samples = 16, n_clones_per_sample = 120, seed = 1)
  expect_s4_class(obj, "CDRobject")
  expect_true(nrow(obj@meta) == 16)
  expect_true(nrow(obj@clones) > 0)
})

test_that("QC + normalize keep samples consistent", {
  obj <- fetch_toy_data(n_samples = 16, n_clones_per_sample = 120, seed = 2)
  obj <- QCRepertoire(obj)
  obj <- NormalizeRepertoire(obj)
  expect_true(all(obj@meta$sample_id %in% unique(obj@clones$sample_id)))
})

test_that("feature matrix has six modules", {
  obj <- fetch_toy_data(n_samples = 16, n_clones_per_sample = 150, seed = 3)
  obj <- QCRepertoire(obj) |> NormalizeRepertoire() |>
    ComputeFeatures(verbose = FALSE)
  expect_equal(length(obj@feature_modules), 6)
  expect_true(nrow(obj@features) == 16)
})

test_that("embedding + classification runs end to end", {
  obj <- fetch_toy_data(n_samples = 24, n_clones_per_sample = 150, seed = 4)
  obj <- QCRepertoire(obj) |> NormalizeRepertoire() |>
    ComputeFeatures(verbose = FALSE) |>
    ConceptBottleneckEmbed(ndim = 5) |>
    DiseaseClassify(use_shap = FALSE, verbose = FALSE)
  expect_true(!is.null(obj@classification$predictions))
  expect_true(obj@classification$train_accuracy > 0)
})

test_that("markers return two layers", {
  obj <- fetch_toy_data(n_samples = 24, n_clones_per_sample = 150, seed = 5)
  obj <- QCRepertoire(obj) |> NormalizeRepertoire() |>
    ComputeFeatures(verbose = FALSE) |>
    FindMarkers(level = "both")
  expect_true(!is.null(obj@markers$statistical))
  expect_true(!is.null(obj@markers$sequence))
})
