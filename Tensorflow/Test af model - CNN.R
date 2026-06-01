library(imager)

sti <- "Billeder/Med objekt(TRUE)/IMG_3885.jpg"

x <- load.image(sti) %>% grayscale() %>% resize(16L, 16L) %>% as.numeric() %>% array(dim = c(1L, 16L, 16L, 1L))

pred_prob <- model$predict(x, verbose = 0L) %>%  as.numeric()

if (pred_prob > 0.5) print("Der er et objekt på billedet") else print("Der ikke et objekt på billedet")
