#' @title Fetch TCR-epitope records from VDJdb (online)
#'
#' @description Scrapes the public VDJdb TCR-epitope database
#' (\url{https://vdjdb.cdr3.net/database}) and returns a tidy data.frame of
#' CDR3-amino-acid / V / J / epitope / MHC / species records. Requires the
#' \code{httr} package (Suggests). Falls back to an offline message if the
#' network or package is unavailable.
#'
#' @param species Character; filter species (e.g. \code{"HomoSapiens"}).
#'   Default \code{NULL} returns all.
#' @param gene Character; chain gene filter, e.g. \code{"TRB"}.
#'   Default \code{NULL} returns all.
#' @param limit Integer; maximum number of records to return.
#'   Default \code{10000}.
#' @param verbose Logical; print progress. Default \code{TRUE}.
#'
#' @return A \code{data.frame} with columns: \code{cdr3_aa, v_gene, j_gene,
#'   gene, species, mhc_a, mhc_b, epitope, antigen_species}.
#'
#' @export
#' @importFrom utils read.delim
fetch_vdjdb <- function(species = NULL, gene = NULL, limit = 10000,
                        verbose = TRUE) {
  url <- "https://vdjdb.cdr3.net/database/search"
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("Package 'httr' is required for online fetch. ",
         "Install it with install.packages('httr'), or use fetch_toy_data() ",
         "for an offline toy repertoire.")
  }
  if (verbose) message("Fetching TCR records from VDJdb ...")
  body <- list(
    "species"    = if (!is.null(species)) species else "any",
    "gene"       = if (!is.null(gene)) gene else "any",
    "limit"      = limit,
    "sort"       = "vdjdb.score",
    "order"      = "desc"
  )
  res <- tryCatch(
    httr::POST(url, body = body, encode = "form",
               httr::user_agent("CDRscope/0.1.0")),
    error = function(e) NULL
  )
  if (is.null(res) || httr::status_code(res) != 200) {
    stop("Failed to reach VDJdb. Check network, or use fetch_toy_data().")
  }
  txt <- httr::content(res, "text", encoding = "UTF-8")
  tc <- textConnection(txt)
  on.exit(close(tc))
  df <- tryCatch(read.delim(tc, header = TRUE, quote = "",
                            stringsAsFactors = FALSE),
                 error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) {
    stop("VDJdb returned no parseable records.")
  }
  out <- data.frame(
    cdr3_aa        = df$cdr3,
    v_gene         = df$v,
    j_gene         = df$j,
    gene           = df$gene,
    species        = df$species,
    mhc_a          = df$mhc.a,
    mhc_b          = df$mhc.b,
    epitope        = df$epitope,
    antigen_species= df$antigen.species,
    stringsAsFactors = FALSE
  )
  if (verbose) message(sprintf("Retrieved %d records.", nrow(out)))
  out
}

#' @title Read a local repertoire file
#'
#' @description Reads a clone table (TSV/CSV) from disk and constructs an
#' initial \code{\link{CDRobject}}. Expected columns: \code{sample_id,
#' cdr3_aa, v_gene, j_gene, chain, count}. A \code{group} column (disease
#' label) is optional and pulled into metadata.
#'
#' @param file Path to a TSV or CSV file.
#' @param sep Field separator. Default auto (\code{\\t} for .tsv, \code{,}
#'   otherwise).
#' @param meta Optional sample-level metadata \code{data.frame}.
#'
#' @return A \code{CDRobject} with \code{meta} and \code{clones} populated.
#'
#' @export
#' @importFrom utils read.delim read.csv
ReadRepertoire <- function(file, sep = NULL, meta = NULL) {
  if (is.null(sep)) sep <- if (grepl("\\.tsv$", file, ignore.case = TRUE)) "\t" else ","
  clones <- if (sep == "\t")
    read.delim(file, stringsAsFactors = FALSE) else
    read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
  req <- c("sample_id", "cdr3_aa", "v_gene", "j_gene", "chain", "count")
  missing <- setdiff(req, colnames(clones))
  if (length(missing)) stop("Missing required columns: ",
                            paste(missing, collapse = ", "))
  if (!"freq" %in% colnames(clones)) clones$freq <- NA_real_
  if ("group" %in% colnames(clones)) {
    meta_local <- unique(clones[, c("sample_id", "group"), drop = FALSE])
    clones$group <- NULL
  } else {
    meta_local <- data.frame(sample_id = unique(clones$sample_id),
                             group = NA_character_, stringsAsFactors = FALSE)
  }
  if (!is.null(meta)) meta_local <- merge(meta_local, meta,
                                          by = "sample_id", all.x = TRUE)
  meta_local$reads <- as.integer(tapply(clones$count, clones$sample_id, sum))[
    match(meta_local$sample_id, names(tapply(clones$count, clones$sample_id, sum)))]
  CDRobject(meta = meta_local, clones = clones,
            features = matrix(nrow = 0, ncol = 0))
}

#' @title Generate a toy TCR/BCR repertoire
#'
#' @description Builds a small synthetic repertoire for demonstration: four
#' disease groups (healthy, infection, autoimmune, tumor) with group-specific
#' convergence and diversity signatures. No network required.
#'
#' @param n_samples Integer; total samples (split across groups). Default 40.
#' @param n_clones_per_sample Integer; clones per sample. Default 300.
#' @param seed Integer; RNG seed. Default 1.
#'
#' @return A \code{CDRobject} with \code{meta} and \code{clones}.
#'
#' @export
fetch_toy_data <- function(n_samples = 40, n_clones_per_sample = 300, seed = 1) {
  set.seed(seed)
  groups <- c("healthy", "infection", "autoimmune", "tumor")
  n_per <- ceiling(n_samples / length(groups))
  sample_ids <- sprintf("S%02d", seq_len(n_samples))
  meta <- data.frame(
    sample_id = sample_ids,
    group = rep(groups, each = n_per)[seq_len(n_samples)],
    stringsAsFactors = FALSE
  )
  aas <- strsplit("ACDEFGHIKLMNPQRSTVWY", "")[[1]]
  v_genes <- paste0("TRBV", sprintf("%02d", 1:20))
  j_genes <- paste0("TRBJ", sprintf("%02d", 1:13))

  make_cdr3 <- function(n, len_mean = 14, len_sd = 2, motif = NULL,
                        motif_freq = 0) {
    len <- pmax(10, as.integer(round(rnorm(n, len_mean, len_sd))))
    inner_len <- len - 2
    out <- vapply(inner_len, function(L)
      paste(sample(aas, L, replace = TRUE), collapse = ""), character(1))
    if (!is.null(motif) && motif_freq > 0) {
      n_motif <- as.integer(motif_freq * n)
      idx <- sample(seq_len(n), n_motif)
      ml <- nchar(motif)
      for (i in idx) {
        L <- nchar(out[i])
        if (L >= ml) {
          pos <- sample(seq_len(L - ml + 1), 1)
          substr(out[i], pos, pos + ml - 1) <- motif
        }
      }
    }
    paste0("C", out, sample(c("F", "W"), n, replace = TRUE))
  }

  clones_list <- lapply(seq_len(n_samples), function(i) {
    grp <- meta$group[i]
    cfg <- switch(grp,
      healthy    = list(n = n_clones_per_sample, m = 14, s = 2,
                        motif = NULL, mf = 0),
      infection  = list(n = n_clones_per_sample, m = 13, s = 1.8,
                        motif = "CASSL", mf = 0.15),
      autoimmune = list(n = n_clones_per_sample, m = 12, s = 2.2,
                        motif = "CASSD", mf = 0.18),
      tumor      = list(n = n_clones_per_sample, m = 13, s = 2.5,
                        motif = "CASSP", mf = 0.10),
      list(n = n_clones_per_sample, m = 14, s = 2, motif = NULL, mf = 0)
    )
    cdr3 <- make_cdr3(cfg$n, cfg$m, cfg$s, cfg$motif, cfg$mf)
    counts <- switch(grp,
      healthy    = as.integer(rexp(cfg$n, rate = 1/3)),
      infection  = c(rep(50:20, length.out = min(40, cfg$n)),
                     as.integer(rexp(max(0, cfg$n - 40), rate = 1/2))),
      autoimmune = c(rep(40:15, length.out = min(30, cfg$n)),
                     as.integer(rexp(max(0, cfg$n - 30), rate = 1/2))),
      tumor      = as.integer(rexp(cfg$n, rate = 1/1.5)),
      as.integer(rexp(cfg$n, rate = 1/2))
    )
    counts <- pmax(1L, counts)
    data.frame(
      sample_id = meta$sample_id[i],
      cdr3_aa   = cdr3,
      v_gene    = sample(v_genes, cfg$n, replace = TRUE),
      j_gene    = sample(j_genes, cfg$n, replace = TRUE),
      chain     = "TRB",
      count     = counts,
      stringsAsFactors = FALSE
    )
  })
  clones <- do.call(rbind, clones_list)
  rownames(clones) <- NULL
  clones$freq <- ave(clones$count, clones$sample_id,
                     FUN = function(x) x / sum(x))
  meta$reads <- as.integer(tapply(clones$count, clones$sample_id, sum))
  CDRobject(meta = meta, clones = clones,
            features = matrix(nrow = 0, ncol = 0))
}
