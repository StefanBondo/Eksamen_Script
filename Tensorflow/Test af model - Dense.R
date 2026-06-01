library(imager)

sti <- "Billeder/Med objekt(TRUE)/dit_billede.jpg"

x <- load.image(sti) %>% grayscale() %>% resize(16L, 16L) %>% as.numeric() %>% matrix(nrow = 1)

pred_prob <- model$predict(x, verbose = 0L) %>% as.numeric()

if (pred_prob > 0.5) print("Der er et objekt på billedet") else print("Der ikke et objekt på billedet")
