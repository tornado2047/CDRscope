#' @title Disease classification with SHAP attribution
#'
#' @description Fits an interpretable classifier on the concept-bottleneck
#' embedding to separate disease groups. Default is one-vs-rest logistic
#' regression (linear, fully interpretable). Attribute importance is computed
#' via SHAP when the \code{iml} package is available, otherwise via
#' standardised coefficients as an interpretable proxy. Predictions and
#' attributions are stored in the \code{classification} slot.
#'
#' @param object A \code{\link{CDRobject}} with embedding populated.
#' @param method Character; \code{"logistic"} (default) or \code{"tree"}.
#' @param use_shap Logical; attempt SHAP via \code{iml}. Default \code{TRUE}.
#' @param verbose Logical. Default \code{TRUE}.
#'
#' @return A \code{CDRobject} with \code{classification} populated.
#' @export
#' @importFrom stats glm predict model.matrix
DiseaseClassify <- function(object, method = c("logistic", "tree"),
                            use_shap = TRUE, verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))
  method <- match.arg(method)
  emb <- object@embedding
  grp <- object@meta$group
  if (ncol(emb) == 0) stop("Run ConceptBottleneckEmbed() first.")
  if (all(is.na(grp))) stop("No group labels in meta; cannot train classifier.")
  ok <- !is.na(grp)
  X <- emb[ok, , drop = FALSE]
  y <- factor(grp[ok])
  classes <- levels(y)

  if (verbose) message(sprintf("Training %s classifier on %d samples, %d classes",
                               method, sum(ok), length(classes)))
  preds <- rep(NA_character_, nrow(emb))
  names(preds) <- rownames(emb)
  fit_list <- list()
  coefs <- matrix(0, nrow = ncol(X) + 1, ncol = length(classes),
                  dimnames = list(c("(Intercept)", colnames(X)), classes))
  if (method == "logistic") {
    if (length(classes) == 2) {
      form <- as.formula("y ~ .")
      fit <- glm(y ~ ., data = data.frame(y = y, X, check.names = FALSE),
                 family = binomial())
      fit_list[[classes[2]]] <- fit
      coefs[, classes] <- 0
      coefs[, classes[2]] <- coef(fit)
      pr <- predict(fit, type = "response")
      pred <- ifelse(pr > 0.5, classes[2], classes[1])
      preds[rownames(X)] <- as.character(pred)
    } else {
      prob_mat <- matrix(0, nrow = nrow(X), ncol = length(classes))
      colnames(prob_mat) <- classes
      for (cl in classes) {
        yb <- ifelse(y == cl, 1, 0)
        fit <- glm(yb ~ ., data = data.frame(yb = yb, X, check.names = FALSE),
                   family = binomial())
        fit_list[[cl]] <- fit
        coefs[, cl] <- coef(fit)
        prob_mat[, cl] <- predict(fit, type = "response")
      }
      preds[rownames(X)] <- classes[apply(prob_mat, 1, which.max)]
    }
  } else {
    if (!requireNamespace("rpart", quietly = TRUE))
      stop("method='tree' requires the 'rpart' package.")
    fit <- rpart::rpart(y ~ ., data = data.frame(y = y, X, check.names = FALSE))
    fit_list[["tree"]] <- fit
    preds[rownames(X)] <- as.character(predict(fit, type = "class"))
    coefs <- matrix(NA, nrow = ncol(X) + 1, ncol = length(classes),
                    dimnames = list(c("(Intercept)", colnames(X)), classes))
  }
  acc <- mean(preds[rownames(X)] == as.character(y), na.rm = TRUE)
  if (verbose) message(sprintf("Training accuracy: %.3f", acc))

  shap <- NULL
  if (use_shap && method == "logistic") {
    shap <- tryCatch({
      if (requireNamespace("iml", quietly = TRUE)) {
        dframe <- data.frame(X, check.names = FALSE)
        pred_fun <- function(model, newdata) {
          if (length(classes) == 2) {
            p <- predict(model[[classes[2]]], newdata, type = "response")
            return(data.frame(healthy = 1 - p, disease = p))
          }
          out <- vapply(classes, function(cl)
            predict(model[[cl]], newdata, type = "response"), numeric(nrow(newdata)))
          out
        }
        predictor <- iml::Predictor$new(
          model = fit_list, data = dframe,
          predict.function = pred_fun, y = y)
        effects <- iml::FeatureEffect$new(predictor, feature = colnames(X)[1],
                                           method = "pdp")
        list(method = "iml", predictor = predictor,
             classes = classes, coefs = coefs)
      } else {
        std <- apply(X, 2, sd)
        contrib <- sweep(abs(coefs[-1, , drop = FALSE]), 1, std, "*")
        list(method = "std_coef_proxy",
             classes = classes,
             mean_abs_contrib = contrib)
      }
    }, error = function(e) {
      std <- apply(X, 2, sd)
      contrib <- sweep(abs(coefs[-1, , drop = FALSE]), 1, std, "*")
      list(method = "std_coef_proxy",
           classes = classes, mean_abs_contrib = contrib)
    })
  }
  object@classification <- list(
    method = method,
    classes = classes,
    fits = fit_list,
    coefficients = coefs,
    predictions = preds,
    train_accuracy = acc,
    shap = shap
  )
  object
}
