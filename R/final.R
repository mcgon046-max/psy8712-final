# Script Settings and Resources 
library(tidyverse)
library(rollama) # Best package for API calls to Ollama apparently 
library(tidymodels) # ML tasks - used over caret because I want to build a tidymodels skill set 
library(tm) # Robust text handling w/ built in corpus handling 
library(textstem) # Lemmatizatoin
library(RWeka) # n-gram tokenization 
library(stm) # Topic modeling 

# Data Import and Cleaning 

df_import <- read_csv("data/glassdoor_reviews.csv") # Initial import read_csv as it creates a tibble for tidyverse, and is better to work with 

df_clean <- df_import |>
  select(
    headline,
    overall_rating, 
    pros,
    cons
  ) |> # selecting text columns and ratings
  slice_sample(n = 10000) |> # Randomly samples rows for machine learning tasks, compute on entire data set was very high
  drop_na() |> # dropping NAs 
  mutate(
    review_text = paste(headline, pros, cons, sep = " ")
  ) |> # Mutate to put all the review text into a single string for NLP 
  select(
    overall_rating,
    review_text
  ) 

# Analysis 

## Create corpus 
corpus <- VCorpus(VectorSource(df_clean$review_text)) # creates the corpus using VCorpus

## Text pre-processing using tm and other packages 
corpus_prep <- corpus |>
  tm_map(content_transformer(str_to_lower)) |> # Makes everything lowercase 
  tm_map(removeNumbers) |> # Gets rid of numbers 
  tm_map(removePunctuation) |> # Gets rid of punctuation 
  tm_map(content_transformer(lemmatize_strings)) |> # Lemmatizing for text stems 
  tm_map(removeWords, stopwords("en")) |> # Gets rid of stopwords 
  tm_map(stripWhitespace) # Gets rid of the whitespace from the lemmetization and the stopword removal 

## N-gram tokenizer 
myTokenizer <- function(x) { 
  NGramTokenizer(
    x, Weka_control(min=1, max=2
  )) 
}

## Document term matrix creation (makes text a matrix)
DTM <- DocumentTermMatrix(
  corpus_prep, 
  control = list(tokenize = myTokenizer)
  )

## Removing sparse terms 
slimmed_dtm <- removeSparseTerms(DTM, .95) # set the threshold to .95 meaning term is allowed to be empty in 95% of text 

## Converting back to dataframe 
tokens_df <- as.matrix(slimmed_dtm) |> # Turns this into a matrix 
  as.tibble() |> # Makes it a tibble for ML tasks, chosen as opposed to base R df due to tidy models
  rename_with(make.names) # Naming cols so that tidy models work 

## Final ML token df 
ml_data_tokens <- bind_cols(
  overall_rating = df_clean$overall_rating, 
  tokens_df
)

## topic analysis 
dfm2stm <- readCorpus(slimmed_dtm, type = "slam") # Read corpus to reformat the dtm for the k search

## K-search algo 
# kresult <- searchK(
#   documents = dfm2stm$documents,
#   vocab = dfm2stm$vocab,
#   K = seq(2, 20, by = 2), # Tests K values from 2 - 20 in 2 topic increments, up to 20 
#   verbose = TRUE # Needed to check on progress 
# )
 ### Commented out so it doesn't rerun on source click 



## saving kresult plot to viz 
png("figs/kfig.png", width = 16, height = 9, units = "in", res = 300)

#plot
plot(kresult)

# Turn the device off to finalize the file save
dev.off()

### Note: I had to use base R not ggsave here to get the png in order for it save correctly 

### It Appears that 6 topics are the best option here. 









