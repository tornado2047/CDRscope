#' @title Project CDR3 sequences onto a fixed reference UMAP map
#'
#' @description Projects new CDR3 sequences onto a pre-built reference map
#'   so that different projects with different inputs all map to the SAME
#'   2D coordinate space for unified comparison and visualization.
#'
#'   The reference map is a trained neural network (480->2) that approximates
#'   the UMAP manifold learned from a reference dataset (956,377 CDR3
#'   sequences from TRA+TRB). Once built, any new CDR3 sequence can be
#'   projected by computing its ESM-2 embedding and passing it through
#'   the saved network weights.
#'
#' @param sequences Character vector of CDR3 amino acid sequences.
#' @param ref_dir Path to the reference map directory (containing
#'   \code{ref_mapper.pt}, \code{ref_config.json}, etc.).
#' @param python Path to the Python interpreter with \code{transformers}
#'   and \code{torch} installed. If \code{NULL}, uses \code{reticulate}.
#' @param batch_size Integer; ESM-2 batch size. Default 256.
#' @param verbose Logical; print progress. Default TRUE.
#'
#' @return A data.frame with columns: sequence, length, net_charge,
#'   hydrophobicity, aromatic_frac, umap1, umap2, and property classes.
#'
#' @export
#' @importFrom utils read.csv write.csv
ProjectToReferenceMap <- function(sequences, ref_dir,
                                   python = NULL, batch_size = 256,
                                   verbose = TRUE) {
  sequences <- unique(sequences[!is.na(sequences) & nchar(sequences) > 0])
  if (length(sequences) == 0) stop("No valid sequences provided.")

  ref_config <- file.path(ref_dir, "ref_config.json")
  if (!file.exists(ref_config)) {
    stop("Reference map not found at: ", ref_dir,
         "\nRun build_reference_map.py first, or use system.file('reference_map', package='CDRscope').")
  }

  tmp_dir <- tempdir()
  input_csv <- file.path(tmp_dir, "ref_input_seqs.csv")
  output_csv <- file.path(tmp_dir, "ref_projected_coords.csv")

  write.csv(data.frame(junction_aa = sequences, stringsAsFactors = FALSE),
            input_csv, row.names = FALSE)

  py_script <- system.file("python", "project_to_reference.py", package = "CDRscope")
  if (!file.exists(py_script)) {
    stop("project_to_reference.py not found in package. ",
         "Run: python inst/python/project_to_reference.py --ref-dir ",
         ref_dir, " --input-csv ", input_csv)
  }

  cmd <- if (!is.null(python)) {
    sprintf("%s %s --ref-dir %s --input-csv %s --output-csv %s --batch-size %d",
            python, py_script, ref_dir, input_csv, output_csv, batch_size)
  } else {
    sprintf("python3 %s --ref-dir %s --input-csv %s --output-csv %s --batch-size %d",
            py_script, ref_dir, input_csv, output_csv, batch_size)
  }

  if (verbose) message("Projecting ", length(sequences), " sequences onto reference map...")
  if (verbose) message("  Command: ", cmd)

  status <- system(cmd, ignore.stdout = !verbose, ignore.stderr = !verbose)
  if (status != 0) {
    stop("Python projection failed (exit ", status, "). ",
         "Ensure transformers + torch are installed.")
  }

  result <- read.csv(output_csv, stringsAsFactors = FALSE)
  if (verbose) message("Projected ", nrow(result), " sequences to reference space.")
  result
}

#' @title Plot reference map with optional new data overlay
#'
#' @description Visualizes the fixed reference map and optionally overlays
#'   new data points projected by \code{\link{ProjectToReferenceMap}}.
#'
#' @param ref_dir Path to the reference map directory.
#' @param new_data Optional data.frame from \code{ProjectToReferenceMap}.
#' @param color_by Column name in new_data to color points by.
#' @param sample_n Integer; number of reference points to subsample for plotting.
#'
#' @return A ggplot object.
#' @export
#' @importFrom ggplot2 ggplot aes geom_point scale_color_manual labs coord_equal theme_minimal
ReferenceMapPlot <- function(ref_dir, new_data = NULL, color_by = "chain",
                              sample_n = 30000) {
  coords_file <- file.path(ref_dir, "ref_coords.npy")
  meta_file <- file.path(ref_dir, "ref_metadata.csv")
  if (!file.exists(meta_file)) {
    meta_file <- file.path(ref_dir, "ref_metadata.csv.gz")
  }
  if (!file.exists(meta_file)) stop("Reference metadata not found: ", meta_file)

  ref_meta <- read.csv(meta_file, stringsAsFactors = FALSE)
  n_ref <- nrow(ref_meta)
  if (n_ref > sample_n) {
    set.seed(42)
    idx <- sample(n_ref, sample_n)
    ref_meta <- ref_meta[idx, ]
  }

  p <- ggplot2::ggplot() +
    ggplot2::geom_point(data = ref_meta, ggplot2::aes(x = umap1, y = umap2),
               color = "grey85", size = 0.3, alpha = 0.4) +
    ggplot2::labs(title = "CDRscope Reference Map",
         x = "UMAP1", y = "UMAP2") +
    ggplot2::coord_equal() +
    ggplot2::theme_minimal()

  if (!is.null(new_data)) {
    if (color_by %in% colnames(new_data)) {
      p <- p + ggplot2::geom_point(data = new_data,
                           ggplot2::aes(x = umap1, y = umap2, color = .data[[color_by]]),
                           size = 0.8, alpha = 0.7)
    } else {
      p <- p + ggplot2::geom_point(data = new_data,
                           ggplot2::aes(x = umap1, y = umap2),
                           color = "#D97742", size = 0.8, alpha = 0.7)
    }
  }

  p
}
