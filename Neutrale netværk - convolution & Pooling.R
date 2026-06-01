library(imager)
library(reticulate)
use_virtualenv("~/.virtualenvs/r-tensorflow", required = TRUE)
library(tensorflow)
library(plotly)

target_width  <- 16L
target_height <- 16L
img_pattern   <- "\\.(jpg|jpeg|png|bmp|tif|tiff)$"

med  <- list.files("Billeder/Med objekt(TRUE)",   img_pattern, full.names = TRUE, ignore.case = TRUE)
uden <- list.files("Billeder/Uden objekt(False)", img_pattern, full.names = TRUE, ignore.case = TRUE)

alle_filer <- c(med, uden)
labels     <- c(rep(1L, length(med)), rep(0L, length(uden)))

pixel_matrix <- do.call(rbind, lapply(alle_filer, \(f)
  as.numeric(grayscale(resize(load.image(f), target_width, target_height)))
))

set.seed(7421)
idx <- sample(nrow(pixel_matrix))
X   <- pixel_matrix[idx, ]
y   <- labels[idx]

X_billeder <- array(0, dim = c(nrow(X), target_height, target_width, 1L))
for (i in seq_len(nrow(X)))
  X_billeder[i, , , 1] <- matrix(X[i, ], nrow = target_height, ncol = target_width, byrow = TRUE)

n_train <- floor(0.8 * nrow(X))
X_train <- X_billeder[1:n_train, , , , drop = FALSE]
y_train <- matrix(y[1:n_train], ncol = 1)
X_test  <- X_billeder[(n_train + 1):nrow(X), , , , drop = FALSE]
y_test  <- matrix(y[(n_train + 1):nrow(X)], ncol = 1)

tf <- tensorflow::tf

model <- tf$keras$Sequential(list(
  tf$keras$layers$Input(shape = tuple(target_height, target_width, 1L)),
  tf$keras$layers$Conv2D(filters = 16L, kernel_size = tuple(3L, 3L), activation = "relu"),
  tf$keras$layers$MaxPooling2D(pool_size = tuple(2L, 2L)),
  tf$keras$layers$Conv2D(filters = 32L, kernel_size = tuple(3L, 3L), activation = "relu"),
  tf$keras$layers$MaxPooling2D(pool_size = tuple(2L, 2L)),
  tf$keras$layers$Flatten(),
  tf$keras$layers$Dense(units = 32L, activation = "relu"),
  tf$keras$layers$Dense(units = 1L,  activation = "sigmoid")
))

model$compile(optimizer = "adam", loss = "binary_crossentropy", metrics = list("accuracy"))

historik <- model$fit(X_train, y_train, epochs = 20L, batch_size = 8L, validation_split = 0.2, verbose = 1L)

eval_tal <- as.numeric(model$evaluate(X_test, y_test, verbose = 0L))
cat("Test loss:", round(eval_tal[1], 4), "\n")
cat("Test accuracy:", round(eval_tal[2], 4), "\n")

historik_df <- data.frame(
  epoch          = seq_along(historik$history$loss),
  train_loss     = as.numeric(historik$history$loss),
  val_loss       = as.numeric(historik$history$val_loss),
  train_accuracy = as.numeric(historik$history$accuracy),
  val_accuracy   = as.numeric(historik$history$val_accuracy)
)

p_loss <- plot_ly(historik_df, x = ~epoch) %>% 
  add_lines(y = ~train_loss, name = "loss",     line = list(color = "steelblue"),
            hovertemplate = "Epoch: %{x}<br>loss: %{y:.4f}<extra></extra>") |>
  add_lines(y = ~val_loss,   name = "val_loss", line = list(color = "seagreen"),
            hovertemplate = "Epoch: %{x}<br>val_loss: %{y:.4f}<extra></extra>") |>
  layout(title = "Loss - CNN", xaxis = list(title = "Epoch"), yaxis = list(title = "Loss"))

p_accuracy <- plot_ly(historik_df, x = ~epoch) %>% 
  add_lines(y = ~train_accuracy, name = "accuracy",     line = list(color = "darkorange"),
            hovertemplate = "Epoch: %{x}<br>accuracy: %{y:.4f}<extra></extra>") |>
  add_lines(y = ~val_accuracy,   name = "val_accuracy", line = list(color = "firebrick"),
            hovertemplate = "Epoch: %{x}<br>val_accuracy: %{y:.4f}<extra></extra>") |>
  layout(title = "Accuracy - CNN", xaxis = list(title = "Epoch"), yaxis = list(title = "Accuracy"))

subplot(p_loss, p_accuracy, nrows = 2, shareX = TRUE, titleY = TRUE)
