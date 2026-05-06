# Script Settings and Resources 
library(tidyverse)
library(rollama) # Best package for API calls to Ollama apparently 
library(caret) # ML tasks 
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
  ) |> # selecting text columns and ratings, 
  drop_na() |> # dropping NAs 
  mutate(
    review_text = paste(headline, pros, cons, sep = " ")
  ) |> # Mutate to put all the review text into a single string for NLP 
  slice_sample(n = 10000) # Randomly samples rows for machine learning tasks, compute on entire data set was very high



