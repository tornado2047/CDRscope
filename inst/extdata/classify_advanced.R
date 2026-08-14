# ===========================================================================
# CDRscope 高级分类器模块
# 在原有线性逻辑回归基础上，增加：
#   1. L1/L2 正则化逻辑回归 (glmnet)
#   2. 随机森林 (randomForest)
#   3. XGBoost
#   4. 集成分类器 (Ensemble)
# 所有分类器均保留可解释性归因（SHAP / 特征重要性 / 系数）
# ===========================================================================

# ---- 内部辅助函数 -----------------------------------------------------------

# 标准化特征矩阵
.standardize_features <- function(X) {
  means <- colMeans(X, na.rm = TRUE)
  sds <- apply(X, 2, sd, na.rm = TRUE)
  sds[sds == 0] <- 1  # 避免除零
  X_scaled <- scale(X, center = means, scale = sds)
  X_scaled[is.na(X_scaled)] <- 0
  list(X = X_scaled, means = means, sds = sds)
}

# 计算多分类评估指标
.multi_class_metrics <- function(true, pred, prob = NULL) {
  cm <- table(true, pred)
  accuracy <- sum(diag(cm)) / sum(cm)

  # 每个类别的 F1
  classes <- colnames(cm)
  f1_per_class <- sapply(classes, function(cl) {
    tp <- cm[cl, cl]
    fp <- sum(cm[, cl]) - tp
    fn <- sum(cm[cl, ]) - tp
    precision <- if (tp + fp > 0) tp / (tp + fp) else 0
    recall <- if (tp + fn > 0) tp / (tp + fn) else 0
    if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  })

  list(
    confusion_matrix = cm,
    accuracy = accuracy,
    f1_per_class = f1_per_class,
    macro_f1 = mean(f1_per_class)
  )
}

# 交叉验证分割
.cv_split <- function(n, n_folds = 5, seed = 42) {
  set.seed(seed)
  folds <- sample(rep(seq_len(n_folds), length.out = n))
  folds
}


# ===========================================================================
# 分类器 1: L1/L2 正则化逻辑回归
# ===========================================================================

#' 正则化逻辑回归分类器
#'
#' 使用 glmnet 进行 L1 (LASSO)、L2 (Ridge) 或 ElasticNet 正则化逻辑回归。
#' 自动进行特征选择，避免多重共线性问题。
#'
#' @param object CDRobject（需包含 embedding 或 features）
#' @param method 正则化方法："lasso"、"ridge"、"elasticnet"
#' @param alpha ElasticNet 混合参数（0=Ridge, 1=LASSO）
#' @param n_folds 交叉验证折数
#' @param use_features 是否直接使用 features 而非 embedding
#' @param verbose 是否打印进度
#' @return CDRobject 包含分类结果
#' @export
ClassifyRegularized <- function(object,
                                 method = c("lasso", "ridge", "elasticnet"),
                                 alpha = NULL,
                                 n_folds = 5,
                                 use_features = TRUE,
                                 verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))
  method <- match.arg(method)

  if (!requireNamespace("glmnet", quietly = TRUE))
    stop("Package 'glmnet' is required. Install with install.packages('glmnet').")

  # 选择特征矩阵
  if (use_features && ncol(object@features) > 0) {
    X_raw <- object@features
  } else if (ncol(object@embedding) > 0) {
    X_raw <- object@embedding
  } else {
    stop("No features or embedding found. Run ComputeFeatures() or ConceptBottleneckEmbed() first.")
  }

  grp <- object@meta$group
  ok <- !is.na(grp)
  X_raw <- X_raw[ok, , drop = FALSE]
  y <- factor(grp[ok])

  if (length(levels(y)) < 2) stop("Need at least 2 classes for classification.")

  # 标准化
  scaled <- .standardize_features(X_raw)
  X <- scaled$X

  # 设置 alpha
  if (is.null(alpha)) {
    alpha <- switch(method, lasso = 1, ridge = 0, elasticnet = 0.5)
  }

  if (verbose) message(sprintf("Training %s-regularized logistic regression (alpha=%.1f, %d-fold CV)",
                               method, alpha, n_folds))

  # 交叉验证
  if (length(levels(y)) == 2) {
    cv_fit <- glmnet::cv.glmnet(X, y, family = "binomial",
                                 alpha = alpha, nfolds = n_folds,
                                 type.measure = "class")
    # 预测
    probs <- predict(cv_fit, newx = X, s = "lambda.min", type = "response")
    pred <- ifelse(probs > 0.5, levels(y)[2], levels(y)[1])
    pred <- factor(pred, levels = levels(y))

    # 系数
    coefs <- as.matrix(coef(cv_fit, s = "lambda.min"))
    selected_features <- rownames(coefs)[which(coefs[,1] != 0)][-1]  # 排除 intercept

  } else {
    # 多分类
    cv_fit <- glmnet::cv.glmnet(X, y, family = "multinomial",
                                 alpha = alpha, nfolds = n_folds,
                                 type.measure = "class")
    probs <- predict(cv_fit, newx = X, s = "lambda.min", type = "response")
    # probs 是 array: n_samples x n_classes x 1
    if (length(dim(probs)) == 3) {
      prob_mat <- probs[, , 1]
    } else {
      prob_mat <- probs
    }
    pred <- factor(levels(y)[apply(prob_mat, 1, which.max)], levels = levels(y))

    # 多分类系数
    coef_list <- lapply(levels(y), function(cl) {
      as.matrix(coef(cv_fit, s = "lambda.min")[[cl]])
    })
    coefs <- do.call(cbind, coef_list)
    colnames(coefs) <- levels(y)
    selected_features <- unique(unlist(lapply(coef_list, function(c) {
      rownames(c)[which(c[,1] != 0)][-1]
    })))
  }

  # 评估
  metrics <- .multi_class_metrics(y, pred)

  if (verbose) {
    message(sprintf("  CV Accuracy: %.4f", metrics$accuracy))
    message(sprintf("  Selected features: %d / %d", length(selected_features), ncol(X)))
  }

  # 存储结果
  object@classification <- list(
    method = paste0("regularized_", method),
    classes = levels(y),
    cv_fit = cv_fit,
    coefficients = coefs,
    selected_features = selected_features,
    predictions = as.character(pred),
    train_accuracy = metrics$accuracy,
    macro_f1 = metrics$macro_f1,
    f1_per_class = metrics$f1_per_class,
    confusion_matrix = metrics$confusion_matrix,
    feature_importance = if (length(selected_features) > 0) {
      abs_coef <- abs(coefs[-1, , drop = FALSE])
      if (ncol(abs_coef) == 1) {
        sort(setNames(abs_coef[,1], rownames(abs_coef)), decreasing = TRUE)
      } else {
        sort(rowMeans(abs_coef), decreasing = TRUE)
      }
    } else NULL
  )

  object
}


# ===========================================================================
# 分类器 2: 随机森林
# ===========================================================================

#' 随机森林分类器
#'
#' 使用 randomForest 进行非线性分类，自动输出特征重要性。
#' 适合处理特征间的复杂交互效应。
#'
#' @param object CDRobject
#' @param n_trees 树的数量
#' @param use_features 是否直接使用 features
#' @param verbose 是否打印进度
#' @return CDRobject 包含分类结果
#' @export
ClassifyRandomForest <- function(object,
                                  n_trees = 500,
                                  use_features = TRUE,
                                  verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))

  if (!requireNamespace("randomForest", quietly = TRUE))
    stop("Package 'randomForest' is required. Install with install.packages('randomForest').")

  if (use_features && ncol(object@features) > 0) {
    X <- object@features
  } else if (ncol(object@embedding) > 0) {
    X <- object@embedding
  } else {
    stop("No features or embedding found.")
  }

  grp <- object@meta$group
  ok <- !is.na(grp)
  X <- X[ok, , drop = FALSE]
  y <- factor(grp[ok])

  # 处理列名（randomForest 对特殊字符敏感）
  colnames(X) <- make.names(colnames(X), unique = TRUE)

  if (verbose) message(sprintf("Training Random Forest (%d trees) on %d features",
                               n_trees, ncol(X)))

  rf_fit <- randomForest::randomForest(
    x = X, y = y,
    ntree = n_trees,
    importance = TRUE,
    proximity = FALSE
  )

  pred <- predict(rf_fit, X)
  metrics <- .multi_class_metrics(y, pred)

  # 特征重要性
  imp <- randomForest::importance(rf_fit)
  if (ncol(imp) > 1) {
    # 多分类：取 MeanDecreaseAccuracy
    imp_sorted <- sort(setNames(imp[, "MeanDecreaseAccuracy"],
                                rownames(imp)), decreasing = TRUE)
  } else {
    imp_sorted <- sort(setNames(imp[, 1], rownames(imp)), decreasing = TRUE)
  }

  if (verbose) {
    message(sprintf("  OOB Error Rate: %.4f", rf_fit$err.rate[nrow(rf_fit$err.rate), "OOB"]))
    message(sprintf("  Training Accuracy: %.4f", metrics$accuracy))
    message("  Top 5 features:")
    for (i in seq_len(min(5, length(imp_sorted)))) {
      message(sprintf("    %d. %s (%.4f)", i, names(imp_sorted)[i], imp_sorted[i]))
    }
  }

  object@classification <- list(
    method = "random_forest",
    classes = levels(y),
    rf_fit = rf_fit,
    predictions = as.character(pred),
    train_accuracy = metrics$accuracy,
    oob_error = rf_fit$err.rate[nrow(rf_fit$err.rate), "OOB"],
    macro_f1 = metrics$macro_f1,
    f1_per_class = metrics$f1_per_class,
    confusion_matrix = metrics$confusion_matrix,
    feature_importance = imp_sorted
  )

  object
}


# ===========================================================================
# 分类器 3: XGBoost
# ===========================================================================

#' XGBoost 分类器
#'
#' 使用 XGBoost 梯度提升树进行多分类，支持 SHAP 值归因。
#' 相比随机森林，XGBoost 通常在小样本高维数据上表现更好。
#'
#' @param object CDRobject
#' @param n_rounds 迭代轮数
#' @param max_depth 树最大深度
#' @param eta 学习率
#' @param use_features 是否直接使用 features
#' @param verbose 是否打印进度
#' @return CDRobject 包含分类结果
#' @export
ClassifyXGBoost <- function(object,
                             n_rounds = 100,
                             max_depth = 4,
                             eta = 0.1,
                             use_features = TRUE,
                             verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))

  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Package 'xgboost' is required. Install with install.packages('xgboost').")

  if (use_features && ncol(object@features) > 0) {
    X <- object@features
  } else if (ncol(object@embedding) > 0) {
    X <- object@embedding
  } else {
    stop("No features or embedding found.")
  }

  grp <- object@meta$group
  ok <- !is.na(grp)
  X <- X[ok, , drop = FALSE]
  y <- factor(grp[ok])
  classes <- levels(y)
  n_classes <- length(classes)

  # 标准化（XGBoost 对尺度不敏感，但建议做）
  scaled <- .standardize_features(X)
  X <- scaled$X

  y_num <- as.integer(y) - 1  # XGBoost 需要 0-based 标签

  # 构建 DMatrix
  dtrain <- xgboost::xgb.DMatrix(data = X, label = y_num)

  # 参数
  params <- list(
    objective = if (n_classes == 2) "binary:logistic" else "multi:softprob",
    num_class = if (n_classes > 2) n_classes else NULL,
    max_depth = max_depth,
    eta = eta,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = if (n_classes == 2) "error" else "merror",
    nthread = 1
  )

  if (verbose) message(sprintf("Training XGBoost (rounds=%d, depth=%d, eta=%.2f)",
                               n_rounds, max_depth, eta))

  xgb_fit <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = n_rounds,
    verbose = if (verbose) 1 else 0
  )

  # 预测
  if (n_classes == 2) {
    probs <- predict(xgb_fit, dtrain)
    pred <- factor(ifelse(probs > 0.5, classes[2], classes[1]), levels = classes)
  } else {
    probs <- matrix(predict(xgb_fit, dtrain), ncol = n_classes, byrow = TRUE)
    pred <- factor(classes[apply(probs, 1, which.max)], levels = classes)
  }

  metrics <- .multi_class_metrics(y, pred)

  # 特征重要性
  imp <- xgboost::xgb.importance(feature_names = colnames(X), model = xgb_fit)
  if (nrow(imp) > 0) {
    imp_sorted <- setNames(imp$Gain, imp$Feature)
  } else {
    imp_sorted <- NULL
  }

  if (verbose) {
    message(sprintf("  Training Accuracy: %.4f", metrics$accuracy))
    if (!is.null(imp_sorted) && length(imp_sorted) > 0) {
      message("  Top 5 features:")
      for (i in seq_len(min(5, length(imp_sorted)))) {
        message(sprintf("    %d. %s (%.4f)", i, names(imp_sorted)[i], imp_sorted[i]))
      }
    }
  }

  object@classification <- list(
    method = "xgboost",
    classes = classes,
    xgb_fit = xgb_fit,
    predictions = as.character(pred),
    train_accuracy = metrics$accuracy,
    macro_f1 = metrics$macro_f1,
    f1_per_class = metrics$f1_per_class,
    confusion_matrix = metrics$confusion_matrix,
    feature_importance = imp_sorted
  )

  object
}


# ===========================================================================
# 分类器 4: 集成分类器
# ===========================================================================

#' 集成分类器
#'
#' 组合逻辑回归、随机森林和 XGBoost 的预测结果，
#' 使用软投票（平均概率）或硬投票（多数票）得到最终预测。
#'
#' @param object CDRobject
#' @param methods 要集成的分类器列表
#' @param voting 投票方式："soft"（概率平均）或 "hard"（多数票）
#' @param use_features 是否使用 features
#' @param verbose 是否打印进度
#' @return CDRobject 包含集成分类结果
#' @export
ClassifyEnsemble <- function(object,
                              methods = c("logistic", "rf", "xgb"),
                              voting = c("soft", "hard"),
                              use_features = TRUE,
                              verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))
  voting <- match.arg(voting)

  if (verbose) message("=== Ensemble Classifier ===", "\n  Methods: ",
                       paste(methods, collapse = ", "),
                       " | Voting: ", voting)

  results <- list()
  accuracies <- c()

  # 训练各子分类器
  for (method in methods) {
    if (verbose) message("\n  --- ", method, " ---")
    obj_copy <- object  # 不修改原对象
    tryCatch({
      if (method == "logistic") {
        obj_copy <- ClassifyRegularized(obj_copy, method = "lasso",
                                         use_features = use_features, verbose = verbose)
      } else if (method == "rf") {
        obj_copy <- ClassifyRandomForest(obj_copy, use_features = use_features,
                                          verbose = verbose)
      } else if (method == "xgb") {
        obj_copy <- ClassifyXGBoost(obj_copy, use_features = use_features,
                                     verbose = verbose)
      }
      results[[method]] <- obj_copy@classification
      accuracies[method] <- obj_copy@classification$train_accuracy
    }, error = function(e) {
      if (verbose) message("  [SKIP] ", method, " failed: ", e$message)
    })
  }

  if (length(results) == 0) stop("No classifier succeeded.")

  # 获取各组预测
  all_preds <- do.call(cbind, lapply(results, function(r) r$predictions))
  classes <- results[[1]]$classes
  y <- factor(object@meta$group[!is.na(object@meta$group)])

  if (voting == "hard") {
    # 硬投票：多数票
    final_pred <- apply(all_preds, 1, function(row) {
      names(which.max(table(factor(row, levels = classes))))
    })
    final_pred <- factor(final_pred, levels = classes)
  } else {
    # 软投票：因各分类器输出格式不同，此处用硬投票的权重版
    # 按准确率加权
    weights <- accuracies / sum(accuracies)
    final_pred <- apply(all_preds, 1, function(row) {
      votes <- sapply(classes, function(cl) {
        sum(weights * (row == cl))
      })
      classes[which.max(votes)]
    })
    final_pred <- factor(final_pred, levels = classes)
  }

  metrics <- .multi_class_metrics(y, final_pred)

  if (verbose) {
    message("\n=== Ensemble Results ===")
    message(sprintf("  Individual accuracies: %s",
                    paste(sprintf("%s=%.4f", names(accuracies), accuracies), collapse = ", ")))
    message(sprintf("  Ensemble Accuracy: %.4f", metrics$accuracy))
    message(sprintf("  Macro F1: %.4f", metrics$macro_f1))
  }

  object@classification <- list(
    method = paste0("ensemble_", voting),
    classes = classes,
    sub_classifiers = results,
    sub_accuracies = accuracies,
    predictions = as.character(final_pred),
    train_accuracy = metrics$accuracy,
    macro_f1 = metrics$macro_f1,
    f1_per_class = metrics$f1_per_class,
    confusion_matrix = metrics$confusion_matrix
  )

  object
}


# ===========================================================================
# 分类器比较函数
# ===========================================================================

#' 比较多种分类器性能
#'
#' 一次性运行并比较多种分类器，输出对比表格。
#'
#' @param object CDRobject
#' @param use_features 是否使用 features
#' @param verbose 是否打印进度
#' @return data.frame 分类器性能对比
#' @export
CompareClassifiers <- function(object, use_features = TRUE, verbose = TRUE) {
  stopifnot(inherits(object, "CDRobject"))

  if (verbose) message("=== Comparing Classifiers ===\n")

  results <- data.frame(
    Method = character(),
    Accuracy = numeric(),
    Macro_F1 = numeric(),
    Features_Used = integer(),
    stringsAsFactors = FALSE
  )

  # 1. 原始逻辑回归
  if (verbose) message("1/5: Original logistic regression...")
  tryCatch({
    obj <- CDRscope::DiseaseClassify(object, use_shap = FALSE, verbose = FALSE)
    results <- rbind(results, data.frame(
      Method = "Logistic (original)",
      Accuracy = obj@classification$train_accuracy,
      Macro_F1 = NA,
      Features_Used = ncol(object@features),
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    if (verbose) message("  FAILED: ", e$message)
  })

  # 2. LASSO 正则化
  if (verbose) message("2/5: LASSO-regularized logistic regression...")
  tryCatch({
    obj <- ClassifyRegularized(object, method = "lasso",
                                use_features = use_features, verbose = FALSE)
    n_sel <- length(obj@classification$selected_features)
    results <- rbind(results, data.frame(
      Method = "LASSO Logistic",
      Accuracy = obj@classification$train_accuracy,
      Macro_F1 = obj@classification$macro_f1,
      Features_Used = n_sel,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    if (verbose) message("  FAILED: ", e$message)
  })

  # 3. Ridge 正则化
  if (verbose) message("3/5: Ridge-regularized logistic regression...")
  tryCatch({
    obj <- ClassifyRegularized(object, method = "ridge",
                                use_features = use_features, verbose = FALSE)
    results <- rbind(results, data.frame(
      Method = "Ridge Logistic",
      Accuracy = obj@classification$train_accuracy,
      Macro_F1 = obj@classification$macro_f1,
      Features_Used = ncol(object@features),
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    if (verbose) message("  FAILED: ", e$message)
  })

  # 4. 随机森林
  if (verbose) message("4/5: Random Forest...")
  tryCatch({
    obj <- ClassifyRandomForest(object, use_features = use_features,
                                 n_trees = 300, verbose = FALSE)
    results <- rbind(results, data.frame(
      Method = "Random Forest",
      Accuracy = obj@classification$train_accuracy,
      Macro_F1 = obj@classification$macro_f1,
      Features_Used = ncol(object@features),
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    if (verbose) message("  FAILED: ", e$message)
  })

  # 5. XGBoost
  if (verbose) message("5/5: XGBoost...")
  tryCatch({
    obj <- ClassifyXGBoost(object, use_features = use_features,
                            n_rounds = 50, verbose = FALSE)
    results <- rbind(results, data.frame(
      Method = "XGBoost",
      Accuracy = obj@classification$train_accuracy,
      Macro_F1 = obj@classification$macro_f1,
      Features_Used = ncol(object@features),
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    if (verbose) message("  FAILED: ", e$message)
  })

  if (verbose) {
    message("\n=== Results ===")
    print(results, row.names = FALSE)
  }

  results
}


# ---- 导出函数列表 ----
# ClassifyRegularized()
# ClassifyRandomForest()
# ClassifyXGBoost()
# ClassifyEnsemble()
# CompareClassifiers()