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

### It Appears that 6 topics are the best option here. This is because it retains semantic coherence, has the highest held out likelihood, has low residuals, and isn't overfit

## Topic model with k = 6
topic_model <- stm(
  documents = dfm2stm$documents, 
  vocab = dfm2stm$vocab, 
  K = 6, 
  verbose = TRUE
)

labelTopics(topic_model)

### Commented in topic output of above call: 

# Topic 1 Top Words:
#   Highest Prob: good, pay, management, salary, company, benefit, work 
# FREX: salary, pay, good, management, bad, good company, low 
# Lift: good company, bad, salary, low, good, pay, little 
# Score: good company, good, pay, salary, bad, company, management 

# Topic 2 Top Words:
#   Highest Prob: work, place, place work, career, learn, great place, great 
# FREX: place work, place, great place, work, career, learn, good place 
# Lift: great place, place work, good place, place, career, learn, work 
# Score: great place, place, place work, good place, career, work, learn 

# Topic 3 Top Words:
#   Highest Prob: work, hour, environment, nice, flexible, friendly, hard 
# FREX: environment, flexible, hour, nice, work environment, friendly, work 
# Lift: work environment, environment, flexible, nice, friendly, hour, hard 
# Score: work environment, flexible, environment, hour, nice, work, friendly 

# Topic 4 Top Words:
#   Highest Prob: great, company, people, opportunity, culture, lot, benefit 
# FREX: great, culture, opportunity, big, company, great company, people 
# Lift: great company, big, culture, great, opportunity, change, company 
# Score: great company, great, company, culture, opportunity, people, big

# Topic 5 Top Words:
#   Highest Prob: work, balance, life, work life, life balance, good work, good 
# FREX: life, balance, work life, life balance, work, good work, growth 
# Lift: life balance, work life, life, balance, good work, work, growth 
# Score: life balance, good work, work life, life, balance, work, good 

# Topic 6 Top Words:
#   Highest Prob: get, much, job, time, work, can, manager 
# FREX: get, long, job, will, manager, time, year 
# Lift: long, will, customer, get, like, year, manager 
# Score: long, job, get, staff, manager, will, much 

### These topics are quite clean, Based on the highest prob and frex I am outputting a tibble with my proposed names
### These names are based on the Highest prob and FREX
### T1: heavily focused on financial aspects and managment
### T2: Development and learning opportunities came out as best here
### T3: These words seemed to center around the actual work enviornment - the actual "vibes" of the office 
### T4: Words centered around the overall company culture 
### T5: Words here are very clearly related to worklife balance 
### T6: This topic appears to be all about grinding and stress due to managers 

topic_table <- tibble(
  Topic = paste0("Topic ", 1:6),
  Name = c(
    "Compensation & Management",
    "Career Development & Learning",
    "Work Environment & Flexibility",
    "Company Culture & Opportunity",
    "Work-Life Balance",
    "Daily Grind & Managerial Stress"
  )
)


## CSV output for git scraping 
topic_table |> 
  write_csv("out/topics.csv")

## Getting embeddings from local LLM model (nomic-embed-text)

### Pulling the model 
pull_model("nomic-embed-text") # pull_model actually calls the Ollama api to get it loaded into my R enviornment 

### Generating the embeddings 
raw_embeddings <- embed_text(
  text = df_clean$review_text,
  model = "nomic-embed-text"
) # Embed text actually generates the vector embeddings from the text using the LLM model 




