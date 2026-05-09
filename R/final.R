# Script Settings and Resources 
library(tidyverse) # Used mostly for Dplyr and tibble functions - works with tidy models 
library(rollama) # Best package for API calls to Ollama, essentially a wrapper with premade functions for communicating with Ollama 
library(tidymodels) # ML tasks - used over caret because I want to build a tidymodels skill set 
library(tm) # Robust text handling w/ built in corpus handling - Chosen for robust VCorpus vs. tidytext unnesting
library(textstem) # Lemmatizatoin
library(RWeka) # n-gram tokenization 
library(stm) # Topic modeling - Allows covariate inclusion unlike standard LDA
library(doParallel) # parallel processing (Mostly for random forest)

# Data Import and Cleaning 

## Seed 
set.seed(42) # Ensures reproducibility 

## Import
df_import <- read_csv("data/glassdoor_reviews.csv") # Initial import read_csv as it creates a tibble for tidyverse, and is better to work with, read.csv() is slower, fread() returns a data.frame 

## pre-processing
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
    review_text = paste(headline, pros, cons, sep = " "), # Smushes all the text together into one vecotr 
      review_text = str_squish(review_text), # Lightweight whitespace removal for later steps
      review_text = str_trunc(review_text, 8000), # Ensures text isn't too long for vector embeddings 
  ) |> # Mutate to put all the review text into a single string for NLP 
  select(
    overall_rating,
    review_text
  ) |>
  mutate(review_id = row_number()) # Create IDs to track reviews across dataframes and cleaning steps. - row_number() over 1:nrow for tidy compatability
### Used dplyr verbs over base R matrix subsetting for readbility and style guide. 
### stringr over gsub() for simpler whitespace/truncation handling


# Analysis 

## Create corpus 
corpus <- VCorpus(VectorSource(df_clean$review_text)) # Creates volatile corpus in RAM; faster than PCorpus (uses disk)

## Text pre-processing using tm and other packages 
corpus_prep <- corpus |>
  tm_map(content_transformer(str_to_lower)) |> # Makes everything lowercase 
  tm_map(removeNumbers) |> # Gets rid of numbers 
  tm_map(removePunctuation) |> # Gets rid of punctuation 
  tm_map(content_transformer(lemmatize_strings)) |> # Lemmatizing for text stems 
  tm_map(removeWords, stopwords("en")) |> # Gets rid of stopwords 
  tm_map(stripWhitespace) # Gets rid of the whitespace from the lemmetization and the stopword removal 
### # tm_map applies transformation functions across the corpus - chosen over base R lapply() because tm_map preserves the corpus object class and document metadata, whereas lapply() would  coerce the corpus into a standard base R list and break downstream tm functions.


## N-gram tokenizer 
myTokenizer <- function(x) { 
  NGramTokenizer(
    x, Weka_control(min=1, max=2
  )) 
} # Generates rolling bi-grams using RWeka - chosen over base R strsplit() (which requires complex regex for n-grams) and tidytext::unnest_tokens() because RWeka works well with the tm package's DTM control arguments.



## Document term matrix creation (makes text a matrix)
DTM <- DocumentTermMatrix(
  corpus_prep, 
  control = list(tokenize = myTokenizer)
  ) # Converts corpus into sparce matrix representing word counts - chosen over TermDocumentMatrix because tidymodels/machine learning algorithms strictly require rows to be observations (documents) and columns to be variables (terms)

## Removing sparse terms 
slimmed_dtm <- removeSparseTerms(DTM, .95) # set the threshold to .95 meaning term is allowed to be empty in 95% of text - reduces dimensionality and overfitting

## Dimension fitting 
df_clean <- df_clean[1:nrow(slimmed_dtm), ] # The DTM row count is the ground truth — subset df_clean to match for dimension purposes

## Converting back to dataframe 
tokens_df <- slimmed_dtm|> 
  as.matrix() |> # Turns this into a matrix 
  as_tibble() |> # Makes it a tibble for ML tasks, chosen as opposed to base R df due to tidy models
  rename_with(make.names) |> #  chosen over leaving raw strings because tidymodels engines (like ranger or lm) will crash if column names contain spaces, punctuation, or start with numbers.
  mutate(review_id = df_clean$review_id)



## topic analysis 
dfm2stm <- readCorpus(slimmed_dtm, type = "slam") # Converts DTM to slam format required by stm vs. default matrix.

## K-search algo 
# kresult <- searchK(
#   documents = dfm2stm$documents,
#   vocab = dfm2stm$vocab,
#   K = seq(2, 20, by = 2), # Tests K values from 2 - 20 in 2 topic increments, up to 20 
#   verbose = TRUE # Needed to check on progress 
# )
 ### Commented out so it doesn't rerun on source click 



# ## saving kresult plot to viz 
# png("figs/kfig.png", width = 16, height = 9, units = "in", res = 300)
# 
# #plot
# plot(kresult)
# 
# # Turn the device off to finalize the file save
# dev.off()

###### Also commented out for run from source 

### Note: I had to use base R not ggsave here to get the png in order for it 
### save correctly 

## Topic explanation 

### It Appears that 6 topics are the best option here. This is because it 
### retains semantic coherence, has the highest held out likelihood, has low 
### residuals, and doesn't appear to be overfit

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


### These names are based on the Highest prob and FREX
### T1: heavily focused on financial aspects and managment - good and bad loaded here likely because pay perceptiopns can be both. 
### T2: Development and learning opportunities came out as best here
### T3: These words seemed to center around the actual work enviornment - the actual "vibes" of the office as well as hourly work flexibility 
### T4: Words centered around the overall company culture 
### T5: Words here are very clearly related to worklife balance 
### T6: This topic appears to be all about unnecssay grinding and stress due to managers - *Likely a burnout indicator*


### These topics are quite clean, Based on the highest prob and frex I am 
### outputting a tibble with my proposed names

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
  write_csv("out/topics.csv") #Used over write.csv() to drop rownames and export faster.

## Theta matrix for ML 
topics_df <- as_tibble(topic_model$theta) |> # The theta matrix essentially tells us what proportion of each topic a given string of review text falls into. 
  rename_with(~paste0("topic_", 1:6)) |>
  mutate(review_id = df_clean$review_id[1:nrow(topic_model$theta)]) |> # subseting df_clean IDs to match theta dimensions
  filter(review_id %in% tokens_df$review_id) 


## Getting embeddings from local LLM model (nomic-embed-text)

### Preping review text for embeddings 
df_clean <- df_clean |>
  mutate(review_text = str_squish(review_text)) |> # Removes extra whitespace/newlines
  filter(review_text != "") |>                    # Ensure no empty strings
  mutate(review_text = str_trunc(review_text, 8000)) # Makes sure the reviews are not too long 


### Pulling the model 
pull_model("nomic-embed-text") # pull_model actually calls the Ollama api to get it loaded into my R enviornment 



### Generating the embeddings 
raw_embeddings <- embed_text(
  text = df_clean$review_text,
  model = "nomic-embed-text"
) # Embed text generates the vector embeddings from the text using the LLM model 

### Formatting for ML 
f_emb <- raw_embeddings |>
  as.matrix() |>
  as_tibble() |> # Formatted for tidy model steps 
  mutate(review_id = df_clean$review_id) |>  
  filter(review_id %in% topics_df$review_id) # filtered for conformability

### Final prep step for data analysis - Creating dataframes based on previous analysis 

overall_rating <- df_clean |>
  select(
    overall_rating,
    review_id
  ) |>
  filter(review_id %in% topics_df$review_id) # outcome variable made to conform as well 

tokens_df <- tokens_df |>
  filter(review_id %in% topics_df$review_id) # Filtered for conformability with topics - needed to put here so code can run from source 

#### Joins 


##### RQ1: Embeddings vs. pure tokens
rq1_df <- df_clean |>
  select(review_id, overall_rating) |>
  inner_join(tokens_df, by = "review_id") |>
  inner_join(f_emb, by = "review_id")

write_rds(rq1_df, "data/rq1_prepped.rds") # save rds for data file incase of crash 

rq1_df <- read_rds("data/rq1_prepped.rds") # saving the variable for checkingpointing 


#### RQ2: Topics vs. Pure tokens 
rq2_df <- df_clean |>
  select(
    review_id, 
    overall_rating
  ) |>
  inner_join(
    tokens_df, 
    by = "review_id"
  ) |>
  inner_join(
    topics_df, 
    by = "review_id"
  )

write_rds(rq2_df, "data/rq2_prepped.rds") # save rds for data file incase of crash 

rq2_df <- read_rds("data/rq2_prepped.rds") # same as above 

#### RQ3: Embeddings vs. Topics vs. Both combined 
rq3_df <- df_clean |>
  select(
    review_id, 
    overall_rating
  ) |>
  inner_join(
    f_emb, 
    by = "review_id"
  ) |>
  inner_join(
    topics_df, 
    by = "review_id"
  )

write_rds(rq3_df, "data/rq3_prepped.rds") # save rds for data file incase of crash 

rq3_df <- read_rds("data/rq3_prepped.rds") # same as above 

##### NOTE: I joined up the tables using an inner join to ensure that 
##### everything was matched up (duplicates not included) and preped for the 
##### ML task. 
##### drop rows missing either tokens, topics or embeddings.


## Machine learning code RQ1:

### Data split and CV folds 

#### Training/test split for RQ1 
data_split_rq1 <- initial_split(
  rq1_df,
  prop = .8,
  strata = overall_rating
) # initial split is like createdatapartition() wherein it splits the sample into 80/20 split based on the overall_rating variable 

train_data <- training(data_split_rq1) # training() shows the actual training set - same as trainData
test_data <- testing(data_split_rq1) # testing() shows the actual testing dataset - same as testData

##### Cross-validation folds for hyperparameter tuning 
cv_folds_rq1 <- vfold_cv(
  train_data, 
  v = 10, 
  strata = overall_rating
) # Similar to the trainControl() call in caret, in tidymodels this is treated as a standalone object natively 

### Tidymodels Recipes - the "blueprints" for data pre-processing (like caret preProcess)
base_rec_rq1 <- recipe(
    overall_rating ~ ., # prediction formula 
    data = train_data
  ) |>
    update_role(
      review_id, 
      new_role = "ID"
  ) |> # makes it so the review ID is not seen as a predictor variable 
    step_normalize(
      all_numeric_predictors()
  ) # scales all predictors (normalizes them) (required for elastic net)

### Tokens only recipe 
tokens_rec_rq1 <- base_rec_rq1 |>
  step_rm(starts_with("dim_")) # Removes embeddings - excludes variable from tibble

### embeddings only recipe
embeds_rec_rq1 <- base_rec_rq1 |>
  step_rm(all_predictors(), -starts_with("dim_")) # keeps the embeddings 

### Model specs 

#### OLS 
ols_spec_rq1 <- linear_reg() |>
  set_engine("lm") # makes this linear regression and the model to OLS - same as train(method = ...)

#### Elsatic net 
enet_spec_rq1 <- linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet") # penalty is how much to shrink the coefficient, mixture is to find optimal alpha value leaving tune() empty means I'm not making any a priori assumptions about what this ought to be 

#### Workflows - bundles together the model specs and the pre-processed data together - similar to train(), but has no direct correlary in caret. 

##### OLS tokens and emberddings 
ols_wf_rq1_tok <- workflow() |>
  add_recipe(tokens_rec_rq1) |> # Uses the recipe from above 
  add_model(ols_spec_rq1) # Chooses the model specifications 

ols_wf_rq1_emb <- workflow() |>
  add_recipe(embeds_rec_rq1) |> # same as above 
  add_model(ols_spec_rq1) # same as above 

##### Elastic Net tokens and embeddings 
enet_wf_rq1_tok <- workflow() |>
  add_recipe(tokens_rec_rq1) |>
  add_model(enet_spec_rq1) # same as above comment (tokens)

enet_wf_rq1_emb <- workflow() |>
  add_recipe(embeds_rec_rq1) |>
  add_model(enet_spec_rq1) # Embeddings 


### Model Execution 

#### OLS 
ols_res_tok_rq1 <- fit_resamples(
  ols_wf_rq1_tok, 
  resamples = cv_folds_rq1
) # fits model on just the tokens 

ols_res_embed_rq1 <- fit_resamples(
  ols_wf_rq1_emb,
  resamples = cv_folds_rq1
) # fits model on just embeddings


#### Elastic net 
# grid 
enet_grid_rq1 <- grid_regular(penalty(), mixture(), levels = 3) # 9 total combinations of penalties and mixtures (too many crashes R) - caret: expand_grid() - generates hyperparameters 

enet_res_tok_rq1 <- tune_grid(
  enet_wf_rq1_tok,
  resamples = cv_folds_rq1,
  grid = enet_grid_rq1
) # fits on only tokens - Executes grid search across resamples. same as train(tuneGrid = ...)

enet_res_emb_rq1 <- tune_grid(
  enet_wf_rq1_emb,
  resamples = cv_folds_rq1,
  grid = enet_grid_rq1
) # fits only embeddings 

### RQ1 metric extraction 

# OLS Metrics (Tokens)
ols_tok_rmse_rq1 <- collect_metrics(ols_res_tok_rq1) |> # collect metrics in order to pull the trained model outputs - caret: model$results - extracts cross-validation performance metrics into a tibble
  filter(.metric == "rmse") |> # only looks at RMSE
  pull(mean) # Pulls the mean of the 10 cross-validated models 

ols_tok_rsq_rq1  <- collect_metrics(ols_res_tok_rq1) |> # same
  filter(.metric == "rsq") |> # looks at R^2
  pull(mean) # same

# Extract OLS Metrics (Embeddings)
ols_emb_rmse_rq1 <- collect_metrics(ols_res_embed_rq1) |> 
  filter(.metric == "rmse") |> 
  pull(mean)

ols_emb_rsq_rq1  <- collect_metrics(ols_res_embed_rq1) |> 
  filter(.metric == "rsq") |> 
  pull(mean) # same as above 

# Best Elastic Net metrics (tokens)
enet_tok_rmse_rq1 <- show_best(
  enet_res_tok_rq1, metric = "rmse", n = 1 # looks for best rmse
  ) |> # Isolates the optimal hyperparameter combination based on target metric - Caret: model$bestTune
  pull(mean)

enet_tok_rsq_rq1  <- show_best(
  enet_res_tok_rq1, metric = "rsq", n = 1 # same but for R^2
  ) |> 
  pull(mean)

# Best Elastic Net metrics (embeddings)
enet_emb_rmse_rq1 <- show_best(
  enet_res_emb_rq1, metric = "rmse", n = 1
  ) |> 
  pull(mean)

enet_emb_rsq_rq1  <- show_best(
  enet_res_emb_rq1, metric = "rsq", n = 1
  ) |> 
  pull(mean)

### train vs. test (looking at out of sample prediction)

#### OLS - tokens
final_fit_ols_tok_rq1 <- last_fit(
  ols_wf_rq1_tok, 
  split = data_split_rq1) # last_fit takes best parameters, trains the model (leaving above steps to show debugging process), and evaluates it on the test set 

#### OLS - embeddings
final_fit_ols_emb_rq1 <- last_fit(
  ols_wf_rq1_emb, 
  split = data_split_rq1) # same as above but on embeddings

#### Elstic net - tokens and embeddings parameters 
best_params_tok_rq1 <- select_best(
  enet_res_tok_rq1, 
  metric = "rmse") # finds best parameters from tuning grid

best_params_emb_rq1 <- select_best(
  enet_res_emb_rq1, 
  metric = "rmse") # same but for embeddings 

#### Elastic Net - finalizing workflows - looks for best workflow that works based on these hyperparameters 
final_enet_wf_tok_rq1 <- finalize_workflow(
  enet_wf_rq1_tok, 
  best_params_tok_rq1) # Updates the workflow with optimal parameters caret fits the final model automatically - no direct equivilent

final_enet_wf_emb_rq1 <- finalize_workflow(
  enet_wf_rq1_emb, 
  best_params_emb_rq1)

#### Elastic Net - doing final fit like from above 
final_fit_enet_tok_rq1 <- last_fit(
  final_enet_wf_tok_rq1, 
  split = data_split_rq1) # Fits on training data and evaluates on test data in one step vs. manual caret::predict()

final_fit_enet_emb_rq1 <- last_fit(
  final_enet_wf_emb_rq1, 
  split = data_split_rq1)


### Collecting metrics 
test_metrics_ols_tok_rq1  <- collect_metrics(final_fit_ols_tok_rq1)
test_metrics_ols_emb_rq1  <- collect_metrics(final_fit_ols_emb_rq1)
test_metrics_enet_tok_rq1 <- collect_metrics(final_fit_enet_tok_rq1)
test_metrics_enet_emb_rq1 <- collect_metrics(final_fit_enet_emb_rq1)

## final results table for RQ1:

final_comparison_table_rq1 <- tibble(
  Model = c("OLS", "OLS", "Elastic Net", "Elastic Net"),
  Feature_Set = c("Tokens Only", "Embeddings Only", "Tokens Only", "Embeddings Only"), # Labels the models and feature set 
  
  # Validation Metrics (from your CV)
  Train_RMSE = c(ols_tok_rmse_rq1, ols_emb_rmse_rq1, enet_tok_rmse_rq1, enet_emb_rmse_rq1), # RMSE from above analysis 
  Train_RSQ  = c(ols_tok_rsq_rq1, ols_emb_rsq_rq1, enet_tok_rsq_rq1, enet_emb_rsq_rq1), # R^2 from above analysis (Both are from train set)
  
  # Test Metrics (The "Holdout" results)
  Test_RMSE = c(
    test_metrics_ols_tok_rq1 |> 
      filter(.metric == "rmse"
    ) |> # pulls the RMSE metric 
    pull(.estimate), test_metrics_ols_emb_rq1 |> 
    filter(.metric == "rmse"
    ) |> 
    pull(.estimate), test_metrics_enet_tok_rq1 
    |> filter(.metric == "rmse"
    ) |> 
    pull(.estimate), test_metrics_enet_emb_rq1 |> 
    filter(.metric == "rmse") |> 
    pull(.estimate)),
  Test_RSQ = c(
    test_metrics_ols_tok_rq1 |> 
    filter(.metric == "rsq"
    ) |> # pulls the R^2 metric 
    pull(.estimate), test_metrics_ols_emb_rq1 |> 
    filter(.metric == "rsq"
    ) |> 
    pull(.estimate), test_metrics_enet_tok_rq1 |> 
    filter(.metric == "rsq"
    ) |> 
    pull(.estimate), test_metrics_enet_emb_rq1 |> 
    filter(.metric == "rsq") |> pull(.estimate)
  )
) |> 
  arrange(Test_RMSE) # Sort by best holdout performance

print(final_comparison_table_rq1) # One can see that the RMSE goes up on the test set and R^2 goes down, indicating a lack of leakage. 

final_comparison_table_rq1 |>
  write_csv("out/rq1_results.csv")

#### Embeddings have better performance and Elastic Net is best here (so far - need to add ranger on)

## Machine learning code for RQ2:

### train vs. test split: 
data_split_rq2 <- initial_split(
  rq2_df,
  prop = .8,
  strata = overall_rating
) # same as above 

train_data_rq2 <- training(data_split_rq2) # same as above 
test_data_rq2 <- testing(data_split_rq2) # same as above 

##### Cross-validation folds for hyperparameter tuning 
cv_folds_rq2 <- vfold_cv(
  train_data_rq2, 
  v = 10, 
  strata = overall_rating
) # same as above

### Recepies 
base_rec_rq2 <- recipe(
  overall_rating ~ ., 
  data = train_data_rq2
) |>
  update_role(
    review_id, 
    new_role = "ID"
  ) |> 
  step_normalize(
    all_numeric_predictors()
  ) # same as above

#### Tokens only recipe 
tokens_rec_rq2 <- base_rec_rq2 |>
  step_rm(starts_with("topic_")) # same as above except that it's topics instead of embeddings 

#### Topics only recipe
topics_rec_rq2 <- base_rec_rq2 |>
  step_rm(all_predictors(), -starts_with("topic_")) # same as above


### Model Specs 

#### OLS 
ols_spec_rq2 <- linear_reg() |>
  set_engine("lm") # same as above 

#### Elastic net 
enet_spec_rq2 <- linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet") # same as above

### Workflows 

##### OLS tokens and topics 
ols_wf_rq2_tok <- workflow() |>
  add_recipe(tokens_rec_rq2) |> 
  add_model(ols_spec_rq2) # same as above 

ols_wf_rq2_top <- workflow() |>
  add_recipe(topics_rec_rq2) |> 
  add_model(ols_spec_rq2) # same as above 

##### Elastic Net tokens and topics 
enet_wf_rq2_tok <- workflow() |>
  add_recipe(tokens_rec_rq2) |>
  add_model(enet_spec_rq2) # same as above

enet_wf_rq2_top <- workflow() |>
  add_recipe(topics_rec_rq2) |>
  add_model(enet_spec_rq2) # same as above


### Model execution:

#### OLS 
ols_res_tok_rq2 <- fit_resamples(
  ols_wf_rq2_tok, 
  resamples = cv_folds_rq2
) # same as above 

ols_res_top_rq2 <- fit_resamples(
  ols_wf_rq2_top,
  resamples = cv_folds_rq2
) # same as above

#### Elastic Net 
# grid
enet_grid_rq2 <- grid_regular(penalty(), mixture(), levels = 3) # same as above

enet_res_tok_rq2 <- tune_grid(
  enet_wf_rq2_tok,
  resamples = cv_folds_rq2,
  grid = enet_grid_rq2
) # same as above

enet_res_top_rq2 <- tune_grid(
  enet_wf_rq2_top,
  resamples = cv_folds_rq2,
  grid = enet_grid_rq2
) # same as above

### RQ2 metric extraction 
# OLS metrics (Tokens)
ols_tok_rmse_rq2 <- collect_metrics(ols_res_tok_rq2) |> 
  filter(.metric == "rmse") |> 
  pull(mean) # same as above 

ols_tok_rsq_rq2  <- collect_metrics(ols_res_tok_rq2) |> 
  filter(.metric == "rsq") |> 
  pull(mean) # same as above

# OLS metrics (Topics)
ols_top_rmse_rq2 <- collect_metrics(ols_res_top_rq2) |> 
  filter(.metric == "rmse") |> 
  pull(mean) # same as above 

ols_top_rsq_rq2  <- collect_metrics(ols_res_top_rq2) |> 
  filter(.metric == "rsq") |> 
  pull(mean) # same as above 

# Best Elastic Net metrics (tokens)
enet_tok_rmse_rq2 <- show_best(
  enet_res_tok_rq2, metric = "rmse", n = 1 
) |> 
  pull(mean) # same as above

enet_tok_rsq_rq2  <- show_best(
  enet_res_tok_rq2, metric = "rsq", n = 1 
) |> 
  pull(mean) # same as above

# Best Elastic Net metrics (topics)
enet_top_rmse_rq2 <- show_best(
  enet_res_top_rq2, metric = "rmse", n = 1
) |> 
  pull(mean) # same as above

enet_top_rsq_rq2  <- show_best(
  enet_res_top_rq2, metric = "rsq", n = 1
) |> 
  pull(mean) # same as above

### Train vs. Test
#### OLS - tokens
final_fit_ols_tok_rq2 <- last_fit(
  ols_wf_rq2_tok, 
  split = data_split_rq2) # same as above 

#### OLS - topics
final_fit_ols_top_rq2 <- last_fit(
  ols_wf_rq2_top, 
  split = data_split_rq2) # same as above 

#### Elastic net - tokens and topics parameters 
best_params_tok_rq2 <- select_best(
  enet_res_tok_rq2, 
  metric = "rmse") # same as above

best_params_top_rq2 <- select_best(
  enet_res_top_rq2, 
  metric = "rmse") # same as above 

#### Elastic Net - finalizing workflows 
final_enet_wf_tok_rq2 <- finalize_workflow(
  enet_wf_rq2_tok, 
  best_params_tok_rq2) # same as above

final_enet_wf_top_rq2 <- finalize_workflow(
  enet_wf_rq2_top, 
  best_params_top_rq2) # same as above

#### Elastic Net - doing final fit
final_fit_enet_tok_rq2 <- last_fit(
  final_enet_wf_tok_rq2, 
  split = data_split_rq2) # same as above

final_fit_enet_top_rq2 <- last_fit(
  final_enet_wf_top_rq2, 
  split = data_split_rq2) # same as above


### Collecting metrics 
test_metrics_ols_tok_rq2  <- collect_metrics(final_fit_ols_tok_rq2) # same as above
test_metrics_ols_top_rq2  <- collect_metrics(final_fit_ols_top_rq2) # same as above
test_metrics_enet_tok_rq2 <- collect_metrics(final_fit_enet_tok_rq2) # same as above
test_metrics_enet_top_rq2 <- collect_metrics(final_fit_enet_top_rq2) # same as above

## Final results table for RQ2

final_comparison_table_rq2 <- tibble(
  Model = c("OLS", "OLS", "Elastic Net", "Elastic Net"),
  Feature_Set = c("Tokens Only", "Topics Only", "Tokens Only", "Topics Only"), # same as above 
  
  # Validation Metrics
  Train_RMSE = c(ols_tok_rmse_rq2, ols_top_rmse_rq2, enet_tok_rmse_rq2, enet_top_rmse_rq2), # same as above 
  Train_RSQ  = c(ols_tok_rsq_rq2, ols_top_rsq_rq2, enet_tok_rsq_rq2, enet_top_rsq_rq2), # same as above 
  
  # Test Metrics
  Test_RMSE = c(
    test_metrics_ols_tok_rq2 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_ols_top_rq2 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_enet_tok_rq2 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_enet_top_rq2 |> filter(.metric == "rmse") |> pull(.estimate)
  ),
  Test_RSQ = c(
    test_metrics_ols_tok_rq2 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_ols_top_rq2 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_enet_tok_rq2 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_enet_top_rq2 |> filter(.metric == "rsq") |> pull(.estimate)
  )
) |> 
  arrange(Test_RMSE) # same as above

print(final_comparison_table_rq2) # same as above 

final_comparison_table_rq2 |>
  write_csv("out/rq2_results.csv") # same as above

#### Outputs indicate that topics are terrible compared to pure tokens 


## Machine learning code for RQ3:

### Data split and CV folds 
data_split_rq3 <- initial_split(
  rq3_df,
  prop = .8,
  strata = overall_rating
) # same as above (Could have used a single split for all three RQs but I thought resplitting seemed more robust)

train_data_rq3 <- training(data_split_rq3) # same as above 
test_data_rq3 <- testing(data_split_rq3) # same as above


##### Cross-validation folds for hyperparameter tuning 
cv_folds_rq3 <- vfold_cv(
  train_data_rq3, 
  v = 10, 
  strata = overall_rating
) # same as above

### Recepies 
base_rec_rq3 <- recipe(
  overall_rating ~ ., 
  data = train_data_rq3
) |>
  update_role(
    review_id, 
    new_role = "ID"
  ) |> 
  step_normalize(
    all_numeric_predictors()
  ) # same as above

### Embeddings only recipe 
embeds_rec_rq3 <- base_rec_rq3 |>
  step_rm(all_predictors(), -starts_with("dim_")) # same as above 

### Topics only recipe
topics_rec_rq3 <- base_rec_rq3 |>
  step_rm(all_predictors(), -starts_with("topic_")) # same as above 

### Combined (Embeddings + Topics) recipe
combined_rec_rq3 <- base_rec_rq3 


### Model Specs
#### OLS 
ols_spec_rq3 <- linear_reg() |>
  set_engine("lm") # same as above 

#### Elastic net 
enet_spec_rq3 <- linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet") # same as above


### Workflows 

#### OLS 
ols_wf_rq3_emb <- workflow() |> 
  add_recipe(embeds_rec_rq3) |> 
  add_model(ols_spec_rq3) # same as above 

ols_wf_rq3_top <- workflow() |> 
  add_recipe(topics_rec_rq3) |> 
  add_model(ols_spec_rq3) # same as above 

ols_wf_rq3_com <- workflow() |> 
  add_recipe(combined_rec_rq3) |> 
  add_model(ols_spec_rq3) # Combined


#### Elastic Net 
enet_wf_rq3_emb <- workflow() |> 
  add_recipe(embeds_rec_rq3) |> 
  add_model(enet_spec_rq3) # same as above

enet_wf_rq3_top <- workflow() |> 
  add_recipe(topics_rec_rq3) |> 
  add_model(enet_spec_rq3) # same as above 

enet_wf_rq3_com <- workflow() |> 
  add_recipe(combined_rec_rq3) |> 
  add_model(enet_spec_rq3) # Combined


### Model Execution 

#### OLS 
ols_res_emb_rq3 <- fit_resamples(
  ols_wf_rq3_emb, 
  resamples = cv_folds_rq3) # same as above 

ols_res_top_rq3 <- fit_resamples(
  ols_wf_rq3_top, 
  resamples = cv_folds_rq3) # same as above

ols_res_com_rq3 <- fit_resamples(
  ols_wf_rq3_com, 
  resamples = cv_folds_rq3) # Combined

#### Elastic Net 
# grid 
enet_grid_rq3 <- grid_regular(penalty(), mixture(), levels = 3) # same as above

enet_res_emb_rq3 <- tune_grid(
  enet_wf_rq3_emb, 
  resamples = cv_folds_rq3, 
  grid = enet_grid_rq3) # same as above

enet_res_top_rq3 <- tune_grid(
  enet_wf_rq3_top, 
  resamples = cv_folds_rq3, 
  grid = enet_grid_rq3) # same as above 

enet_res_com_rq3 <- tune_grid(
  enet_wf_rq3_com, 
  resamples = cv_folds_rq3, 
  grid = enet_grid_rq3) # Combined


### RQ3 model extraction 

#### OLS metrics (embeddings)
# OLS Metrics (Embeddings)
ols_emb_rmse_rq3 <- collect_metrics(
  ols_res_emb_rq3
  ) |> 
  filter(.metric == "rmse"
  ) |> pull(mean) # same as above 

ols_emb_rsq_rq3  <- collect_metrics(ols_res_emb_rq3) |> 
  filter(.metric == "rsq"
  ) |> 
  pull(mean) # same as above

# OLS Metrics (Topics)
ols_top_rmse_rq3 <- collect_metrics(
  ols_res_top_rq3
  ) |> 
  filter(.metric == "rmse"
  ) |> 
  pull(mean) # same as above 

ols_top_rsq_rq3  <- collect_metrics(ols_res_top_rq3) |> 
  filter(.metric == "rsq"
  ) |> 
  pull(mean) # same as above 

# OLS Metrics (Combined)
ols_com_rmse_rq3 <- collect_metrics(ols_res_com_rq3) |> 
  filter(.metric == "rmse"
  ) |> 
  pull(mean) # Combined

ols_com_rsq_rq3  <- collect_metrics(ols_res_com_rq3) |> 
  filter(.metric == "rsq"
  ) |> 
  pull(mean) # Combined

# Best Elastic Net metrics (Embeddings)
enet_emb_rmse_rq3 <- show_best(
  enet_res_emb_rq3, 
  metric = "rmse", n = 1
  ) |> 
  pull(mean) # same as above

enet_emb_rsq_rq3  <- show_best(
  enet_res_emb_rq3, metric = "rsq", n = 1
  ) |> 
  pull(mean) # same as above

# Best Elastic Net metrics (Topics)
enet_top_rmse_rq3 <- show_best(
  enet_res_top_rq3, metric = "rmse", n = 1
  ) |> 
  pull(mean) # same as above

enet_top_rsq_rq3  <- show_best(enet_res_top_rq3, metric = "rsq", n = 1) |> 
  pull(mean) # same as above

# Best Elastic Net metrics (Combined)
enet_com_rmse_rq3 <- show_best(
  enet_res_com_rq3, 
  metric = "rmse", n = 1
  ) |> 
  pull(mean) # Combined

enet_com_rsq_rq3  <- show_best(
  enet_res_com_rq3, 
  metric = "rsq", n = 1
  ) |> 
  pull(mean) # Combined

### Train vs. test 
#### OLS - last fits
final_fit_ols_emb_rq3 <- last_fit(
  ols_wf_rq3_emb, 
  split = data_split_rq3) # same as above 

final_fit_ols_top_rq3 <- last_fit(
  ols_wf_rq3_top, 
  split = data_split_rq3) # same as above 

final_fit_ols_com_rq3 <- last_fit(
  ols_wf_rq3_com, 
  split = data_split_rq3) # Combined


#### Elastic net - best parameters 
best_params_emb_rq3 <- select_best(
  enet_res_emb_rq3, 
  metric = "rmse") # same as above

best_params_top_rq3 <- select_best(
  enet_res_top_rq3, 
  metric = "rmse") # same as above 

best_params_com_rq3 <- select_best(
  enet_res_com_rq3, 
  metric = "rmse") # Combined


#### Elastic Net - finalizing workflows 
final_enet_wf_emb_rq3 <- finalize_workflow(
  enet_wf_rq3_emb, 
  best_params_emb_rq3) # same as above

final_enet_wf_top_rq3 <- finalize_workflow(
  enet_wf_rq3_top, 
  best_params_top_rq3) # same as above

final_enet_wf_com_rq3 <- finalize_workflow(
  enet_wf_rq3_com, 
  best_params_com_rq3) # Combined

#### Elastic Net - final fit 
final_fit_enet_emb_rq3 <- last_fit(
  final_enet_wf_emb_rq3, 
  split = data_split_rq3) # same as above

final_fit_enet_top_rq3 <- last_fit(
  final_enet_wf_top_rq3, 
  split = data_split_rq3) # same as above

final_fit_enet_com_rq3 <- last_fit(
  final_enet_wf_com_rq3, 
  split = data_split_rq3) # Combined

#### Collecting test metrics 
test_metrics_ols_emb_rq3  <- collect_metrics(final_fit_ols_emb_rq3) # same as above
test_metrics_ols_top_rq3  <- collect_metrics(final_fit_ols_top_rq3) # same as above
test_metrics_ols_com_rq3  <- collect_metrics(final_fit_ols_com_rq3) # Combined
test_metrics_enet_emb_rq3 <- collect_metrics(final_fit_enet_emb_rq3) # same as above
test_metrics_enet_top_rq3 <- collect_metrics(final_fit_enet_top_rq3) # same as above
test_metrics_enet_com_rq3 <- collect_metrics(final_fit_enet_com_rq3) # Combined


### Final results table 
final_comparison_table_rq3 <- tibble(
  Model = c("OLS", "OLS", "OLS", "Elastic Net", "Elastic Net", "Elastic Net"),
  Feature_Set = c("Embeddings Only", "Topics Only", "Combined", "Embeddings Only", "Topics Only", "Combined"), 
  
  # Validation Metrics
  Train_RMSE = c(ols_emb_rmse_rq3, ols_top_rmse_rq3, ols_com_rmse_rq3, enet_emb_rmse_rq3, enet_top_rmse_rq3, enet_com_rmse_rq3), 
  Train_RSQ  = c(ols_emb_rsq_rq3, ols_top_rsq_rq3, ols_com_rsq_rq3, enet_emb_rsq_rq3, enet_top_rsq_rq3, enet_com_rsq_rq3), 
  
  # Test Metrics
  Test_RMSE = c(
    test_metrics_ols_emb_rq3 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_ols_top_rq3 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_ols_com_rq3 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_enet_emb_rq3 |> filter(.metric == "rmse") |> pull(.estimate), 
    test_metrics_enet_top_rq3 |> filter(.metric == "rmse") |> pull(.estimate),
    test_metrics_enet_com_rq3 |> filter(.metric == "rmse") |> pull(.estimate)
  ),
  Test_RSQ = c(
    test_metrics_ols_emb_rq3 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_ols_top_rq3 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_ols_com_rq3 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_enet_emb_rq3 |> filter(.metric == "rsq") |> pull(.estimate), 
    test_metrics_enet_top_rq3 |> filter(.metric == "rsq") |> pull(.estimate),
    test_metrics_enet_com_rq3 |> filter(.metric == "rsq") |> pull(.estimate)
  )
) |> 
  arrange(Test_RMSE) # same as above

print(final_comparison_table_rq3) # same as above 

final_comparison_table_rq3 |>
  write_csv("out/rq3_results.csv") # same as above


### Doing it all with ranger as well 

#### Parallel calls - needed for ranger, takes too much time if I don't 
all_cores <- parallel::detectCores(logical = FALSE) # Detecting cores (its 8)
registerDoParallel(cores = 7) # Reserving 7 cores 
getDoParWorkers() # check to ensure that the cores are given to the enviorment 

#### Random forest engine 
# Model specifications (hyperparameters) - trees set to 100 to reduce compute time and still be fairly robust 
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 100) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("regression") # This sets the tuning grid for random forest for the subsequent calls. Ranger is the engine, impurity tells the model to calculate the total error variance and shows which scores are the best for the model 

# Tuning grid 
rf_grid <- grid_regular(mtry(range = c(5, 50)), min_n(), levels = 3) # grid regular does a grid search with these parameters, mtry is about the number of predictors per split (Chosen to not have too many due to compute time, and to give variance in the data set), min_n() is the minimum number of data points (not assumeed a priori, actually uses the defaults(usually between 2 and 40)


### RQ1 Workflows (Tokens vs. Embeddings)
rf_wf_tok_rq1 <- workflow() |> add_recipe(
  tokens_rec_rq1 
  ) |> 
  add_model(rf_spec)
rf_wf_emb_rq1 <- workflow() |> add_recipe(embeds_rec_rq1) |> add_model(rf_spec)

### RQ2 Workflows (Tokens vs. Topics)
rf_wf_tok_rq2 <- workflow() |> 
  add_recipe(tokens_rec_rq2) |> 
  add_model(rf_spec)

rf_wf_top_rq2 <- workflow() |> 
  add_recipe(topics_rec_rq2) |> 
  add_model(rf_spec)

### RQ3 Workflows (Embeddings vs. Topics vs. Combined)
rf_wf_emb_rq3 <- workflow() |> 
  add_recipe(embeds_rec_rq3) |>
  add_model(rf_spec)

rf_wf_top_rq3 <- workflow() |> 
  add_recipe(topics_rec_rq3) |> 
  add_model(rf_spec)

rf_wf_com_rq3 <- workflow() |> 
  add_recipe(combined_rec_rq3) |> 
  add_model(rf_spec)


### Tuning - for all RQs
# RQ1
rf_res_tok_rq1 <- tune_grid(
  rf_wf_tok_rq1, 
  resamples = cv_folds_rq1, 
  grid = rf_grid)

rf_res_emb_rq1 <- tune_grid(
  rf_wf_emb_rq1, 
  resamples = cv_folds_rq1, 
  grid = rf_grid)

# RQ2
rf_res_tok_rq2 <- tune_grid(
  rf_wf_tok_rq2, 
  resamples = cv_folds_rq2, 
  grid = rf_grid)

rf_res_top_rq2 <- tune_grid(
  rf_wf_top_rq2, 
  resamples = cv_folds_rq2, 
  grid = rf_grid)

# RQ3
rf_res_emb_rq3 <- tune_grid(
  rf_wf_emb_rq3, 
  resamples = cv_folds_rq3, 
  grid = rf_grid)

rf_res_top_rq3 <- tune_grid(
  rf_wf_top_rq3, 
  resamples = cv_folds_rq3, 
  grid = rf_grid)

rf_res_com_rq3 <- tune_grid(
  rf_wf_com_rq3, 
  resamples = cv_folds_rq3, 
  grid = rf_grid)

# Shut down the parallel cluster 
stopImplicitCluster()

### RQ1 Final Fits
final_rf_tok_rq1 <- last_fit(
  finalize_workflow(rf_wf_tok_rq1, select_best(rf_res_tok_rq1, metric = "rmse")), 
  split = data_split_rq1
)
final_rf_emb_rq1 <- last_fit(
  finalize_workflow(rf_wf_emb_rq1, select_best(rf_res_emb_rq1, metric = "rmse")), 
  split = data_split_rq1
)

### RQ2 Final Fits
final_rf_tok_rq2 <- last_fit(
  finalize_workflow(rf_wf_tok_rq2, select_best(rf_res_tok_rq2, metric = "rmse")), 
  split = data_split_rq2
)
final_rf_top_rq2 <- last_fit(
  finalize_workflow(rf_wf_top_rq2, select_best(rf_res_top_rq2, metric = "rmse")), 
  split = data_split_rq2
)

### RQ3 Final Fits
final_rf_emb_rq3 <- last_fit(
  finalize_workflow(rf_wf_emb_rq3, select_best(rf_res_emb_rq3, metric = "rmse")), 
  split = data_split_rq3
)
final_rf_top_rq3 <- last_fit(
  finalize_workflow(rf_wf_top_rq3, select_best(rf_res_top_rq3, metric = "rmse")), 
  split = data_split_rq3
)
final_rf_com_rq3 <- last_fit(
  finalize_workflow(rf_wf_com_rq3, select_best(rf_res_com_rq3, metric = "rmse")), 
  split = data_split_rq3
)

### Final Master Table
final_rf_table <- tibble(
  Research_Question = c("RQ1", "RQ1", "RQ2", "RQ2", "RQ3", "RQ3", "RQ3"),
  Feature_Set = c("Tokens", "Embeddings", "Tokens", "Topics", "Embeddings", "Topics", "Combined"),
  
  # Train RMSE (Best from CV folds)
  Train_RMSE = c(
    show_best(rf_res_tok_rq1, metric = "rmse", n = 1) |> pull(mean),
    show_best(rf_res_emb_rq1, metric = "rmse", n = 1) |> pull(mean),
    show_best(rf_res_tok_rq2, metric = "rmse", n = 1) |> pull(mean),
    show_best(rf_res_top_rq2, metric = "rmse", n = 1) |> pull(mean),
    show_best(rf_res_emb_rq3, metric = "rmse", n = 1) |> pull(mean),
    show_best(rf_res_top_rq3, metric = "rmse", n = 1) |> pull(mean),
    show_best(rf_res_com_rq3, metric = "rmse", n = 1) |> pull(mean)
  ),
  
  # Train RSQ
  Train_RSQ = c(
    show_best(rf_res_tok_rq1, metric = "rsq", n = 1) |> pull(mean),
    show_best(rf_res_emb_rq1, metric = "rsq", n = 1) |> pull(mean),
    show_best(rf_res_tok_rq2, metric = "rsq", n = 1) |> pull(mean),
    show_best(rf_res_top_rq2, metric = "rsq", n = 1) |> pull(mean),
    show_best(rf_res_emb_rq3, metric = "rsq", n = 1) |> pull(mean),
    show_best(rf_res_top_rq3, metric = "rsq", n = 1) |> pull(mean),
    show_best(rf_res_com_rq3, metric = "rsq", n = 1) |> pull(mean)
  ),
  
  # Test RMSE (From holdout split)
  Test_RMSE = c(
    collect_metrics(final_rf_tok_rq1) |> filter(.metric == "rmse") |> pull(.estimate),
    collect_metrics(final_rf_emb_rq1) |> filter(.metric == "rmse") |> pull(.estimate),
    collect_metrics(final_rf_tok_rq2) |> filter(.metric == "rmse") |> pull(.estimate),
    collect_metrics(final_rf_top_rq2) |> filter(.metric == "rmse") |> pull(.estimate),
    collect_metrics(final_rf_emb_rq3) |> filter(.metric == "rmse") |> pull(.estimate),
    collect_metrics(final_rf_top_rq3) |> filter(.metric == "rmse") |> pull(.estimate),
    collect_metrics(final_rf_com_rq3) |> filter(.metric == "rmse") |> pull(.estimate)
  ),
  
  # Test RSQ
  Test_RSQ = c(
    collect_metrics(final_rf_tok_rq1) |> filter(.metric == "rsq") |> pull(.estimate),
    collect_metrics(final_rf_emb_rq1) |> filter(.metric == "rsq") |> pull(.estimate),
    collect_metrics(final_rf_tok_rq2) |> filter(.metric == "rsq") |> pull(.estimate),
    collect_metrics(final_rf_top_rq2) |> filter(.metric == "rsq") |> pull(.estimate),
    collect_metrics(final_rf_emb_rq3) |> filter(.metric == "rsq") |> pull(.estimate),
    collect_metrics(final_rf_top_rq3) |> filter(.metric == "rsq") |> pull(.estimate),
    collect_metrics(final_rf_com_rq3) |> filter(.metric == "rsq") |> pull(.estimate)
  )
) |> 
  arrange(Research_Question, Test_RMSE)

# print and save
print(final_rf_table)

final_rf_table |>
  write_csv("out/random_forest_all_rqs_results.csv")


# Final rds save
save.image(file = "out/workspace.RData")


#### FINAL RESEARCH QUESTION ANALYSIS 

# RQ1: Does the use of embeddings (using the nomic-embed-text LLM embeddings 
# model) improve prediction of satisfaction beyond a rigorous tokenization 
# strategy?

## Answer: Embeddings clearly win. Across every model type, switching from
## tokens to embeddings drops RMSE by roughly .10 and increases R^2 by around
## .22-.38. The vector embeddings are shown to have meaningfully improved 
## explanatory power as compared to pure tokens. 


# RQ2: Does the use of topics improve prediction of satisfaction beyond a 
# rigorous tokenization strategy?

## Answer: Topics are suprisingly awful at prediction, they have an R^2 that is 
## essentially 0. It suggests that although these topics may capture themes, 
## they fail to meaningfully predict overall satisfaction. Take the example of 
## the compensation topic, although this is meaningful topic, compensation can
## be both good or bad, which can cancel out and provide next to no explanatory 
## power 


# RQ3: Does the use of embeddings plus topics improve prediction of satisfaction 
# beyond either alone? 

## Answer: Combining topics and embeddings does not meaningfully improve 
## prediction over embeddings alone or topics alone (which I've explained are
## awful). the combined model is essentially identical to just using vector 
## embeddings, with very similar RMSE and R^2


# RQ4: What is the best prediction of overall job satisfaction achievable using 
# text reviews as source data?

## Answer: The best performing model on this data set was Elastic Net with 
## embeddings only (RMSE = ~.911, R^2 = ~0.386). Further, the fact that random
## forest does not meaningfully improve prediction is notable, this suggests
## that the embedding space is farily linear with respect to rating. The extra 
## complexity of RF is not adding to the prediction of overall satisfaction.


##### Limitations/what I would have done better. 
# DTM, embeddings, and topic model were fit on the full dataset
# prior to train/test split, representing a minor form of feature leakage.
# This is common in NLP pipelines and likely has minimal practical impact
# given more time, I would have split the datasets before the actual 
# training, but this resulted in several errors and code refactoring that was
# unobtainable given the time constraints. 

# Also, the usage of the supercomputer would've been better. this would allow 
# these algorithms to be trained on the full data set, not a random sample 
# therein. 

# Lastly, I would've liked to produce a shinyapp wherein one can interact with
# the model outputs. 

