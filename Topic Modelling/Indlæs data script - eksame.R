library(httr)
library(jsonlite)
library(tidyverse)
library(tidytext)
library(stopwords)
library(readxl)

# --- 1. Hent data via API ---

url <- ""

res <- GET(url)
df <- fromJSON(content(res, as = "text", encoding = "UTF-8"), flatten = TRUE)

df <- read.csv()
df <- readRDS()
df <- read_xlsx()

# --- 2. Tokenisering --- #Find relevante kolonne og erstat "text_kolonne"
words <- df %>% 
  unnest_tokens(word, text_kolonne)

# --- 3. Stopord ---
da_stop_words <- tibble(word = stopwords("da", source = "snowball"))
ekstra_stop_words <- tibble(word = c("ord1", "ord2"))
all_stop_words <- bind_rows(get_stopwords() %>%  select(word), da_stop_words, ekstra_stop_words) %>%  distinct()

ord <- words %>% 
  anti_join(all_stop_words, by = "word") %>% 
  filter(str_detect(word, "[a-zA-ZæøåÆØÅ]"), str_length(word) > 2) %>% 
  count(word, sort = TRUE)


