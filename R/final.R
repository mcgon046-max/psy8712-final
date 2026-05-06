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
  tm_map(content_transformer(lemmatize_words)) |> # Lemmatizing for text stems 
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
slimmed_dtm <- removeSparseTerms(DTM, .95) # set the threshhold to .90 meaning terms must appear in 95% of docs 

## Converting back to dataframe 





