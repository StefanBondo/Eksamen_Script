library(topicmodels)
library(tidytext)
library(tidyverse)

#Document_unit skal ændres efter det relevante datasæt
dtm <- words %>% count(document_unit, word) %>% cast_dtm(document_unit, word, n)

lda_model <- LDA(dtm, k = 2, control = list(seed = 1234))

gamma <- tidy(lda_model, matrix = "gamma")
gamma %>% arrange(document, topic)

beta <- tidy(lda_model, matrix = "beta")
beta %>% group_by(topic) %>% slice_max(beta, n = 10) %>% ungroup()

ggplot(gamma, aes(x = factor(topic), y = gamma)) +
  geom_boxplot() +
  facet_wrap(~document) +
  labs(x = "Emne", y = "Gamma")
