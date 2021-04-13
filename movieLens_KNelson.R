## ------------------------------------------------------------------------------------------------------------------
# Capstone Project - MovieLens  ----------------------------------------------

# Kevin Nelson 
# Submitted Friday April 13, 2021
# movieLens_KNelson.R

# For a more detailed description of this project, please refer to the .Rmd file 
# titled moviesLens_KevinNelson.Rmd.
# That .Rmd file expands on the code presented in this .R file.

# The objective was to design a program that predicts movie ratings and calculates the 
# resulting RMSE. With the dataset provided, there are several variables we could 
# analyze to develop this model. 

# Factors to potentially consider
#  - Average rating of the movie
#  - Average rating of genre
#  - User's preference towards a genre
#  - Average rating of all movies (for movies with low ratings totals)
#  - When the rating was made
#  - When the movie was released
#  - Rating time in relation to release time 
#  - Genre and release year as variables are somewhat baked into the movie's rating 
#    already, may be better to relate these things to the user's preference instead 

# Not all factors will be considered for a variety of reasons. 
# For example, rating time in relation to release time is not helpful since ratings 
# only took place in the last 15 years of the range of movie release years. 

# The ratings go from 0.5 to 5.0 in increments of 0.5. Our model should put a cap on 
# predictions so they can not fall below/above this min/max.

####################################################################################



## ----Libraries, results = FALSE, message = FALSE, warning = FALSE--------------------------------------------------

### INTRODUCTION

# Load Libraries ------------------------------------------------------------

## Caret package
if(!require(caret)){
  install.packages("caret")
  library(caret)
}
## Data.table package
if(!require(data.table)){
  install.packages("data.table")
  library(data.table)
}
## Dplyr package
if(!require(dplyr)){
  install.packages("dplyr")
  library(dplyr)
}
## Ggplot2 package
if(!require(ggplot2)){
  install.packages("ggplot2")
  library(ggplot2)
}
## Ggrepel package
if(!require(ggrepel)){
  install.packages("ggrepel")
  library(ggrepel)
}
## Gtools package
if(!require(gtools)){
  install.packages("gtools")
  library(gtools)
}
## Lubridate package
if(!require(lubridate)){
  install.packages("lubridate")
  library(lubridate)
}
## Stringr package
if(!require(stringr)){
  install.packages("stringr")
  library(stringr)
}
## Tidyverse package
if(!require(tidyverse)){
  install.packages("tidyverse")
  library(tidyverse)
}


## ----Load Data, results = FALSE, message = FALSE, warning = FALSE--------------------------------------------------

# Load Movies Dataset -----------------------------------------------------
# This section is the code provided from the class with the steps to download the edx and 
# validation datasets that will be used throughout this project.

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies<-as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, 
                                  times = 1, 
                                  p = 0.1, 
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



## ----Define RMSE function and find mean, results = FALSE, message = FALSE, warning = FALSE-------------------------

# Define basic variables and functions -------------------------------------

## None of the following variables need to be set to TRUE to review this code and project.
## Optimized values from tuning have been included as alternatives to these logical 
# variables.

# Set to true if tuning based on original plan (data set and user level lambdas)
original_tuning = FALSE   

# Set to true if tuning simplified version (single lambda on data set level variables)
simplified_tuning = FALSE 

# Set to true if tuning modified simplified version (different lambdas across data set 
# level variables)
modified_tuning = FALSE  
# Set variable that activates the parts of this code used for tuning. 
# Warning: Turning tuning on when reviewing the code will activate some sections of the 
# code that are time consuming. 

# Logicals to set if not performing tuning
noTuning_oneLambda = TRUE ## FINAL VERSION OF CODE SET TO TRUE
noTuning_multipleLambdas = FALSE

options(dplyr.summarise.inform = FALSE) # Turn off the warnings created by the group_by 
# function so to not congest the console during operation.


# Create a function that calculates the RMSE
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Create a variable that is the mean of all rating in the edx dataset
mu <- mean(edx$rating) 

# Create new column that extracts release year from title column ----------

# The original dataset combined movie title and release year in a string within a 
# single column. Release year may be a variable that informs this model and the following 
# piece of code extracts that information from the title column and isolates it for 
# additional analysis. 

# Add column for release year
edx <- edx %>% mutate(release_year = str_extract(title, "(?<=\\()\\d{4}?(?=\\))"))

######################################################################################


## ----Test for variation in genre, results = FALSE, message = FALSE, warning = FALSE--------------------------------

### METHODS & ANALYSIS

# Create training and test sets from edx dataset --------------------------

# Create a separate test and training set from the edx dataset
#  The test set has been sliced as 10% of the original dataset
set.seed(25, sample.kind = "Rounding")
index <- createDataPartition(y = edx$rating, times = 10, p = 0.10, list = FALSE)

# Cross validation will ultimately will be used but the first partition from the previous 
# step will be taken to establish a default training set. Testing sets will be 
# established during cross validation. 
edx_train <- edx[-index[ , 1], ]



## ------------------------------------------------------------------------------------------------------------------
# Test to see variation in ratings across genres --------------------------

# Based on the classwork, it is already known that the user and movie are heavily 
# influential on developing an effective rating system. The remaining variables at 
# our disposal which also play a role are genre and release year. This section 
# explores the variation within those variables. 

# Create variable that groups by genre
genreRating <- edx %>%
  group_by(genres) %>%
  summarize(rating = mean(rating), n = n()) 

# Graph to show ratings per genre
ggplot(data = genreRating, aes(x = genres, y = rating, size = n)) +
  geom_point() +
  geom_text_repel(data = filter(genreRating, rating > 4.2 | rating < 2), 
                  aes(label = genres)) +
  theme(axis.text.x = element_blank()) +
  ggtitle(label = 'Average Rating as a Function of Genre',
          subtitle = 'Genres with Average Ratings < 2 & > 4.2 Are Labeled') +
  xlab('Genre') +
  ylab('Average Rating')



## ----Check to see if regularization is required--------------------------------------------------------------------

# Examine top and bottom 10 rated genres to see if regularization is required

# Bottom 10
slice_min(genreRating, order_by = rating, n = 10)

# Top 10
slice_max(genreRating, order_by = rating, n = 10)

# It's clear there is variation throughout the genres variable and that regularization 
# is required on the data level. But can we go one level deeper? It is intuitive to 
# suspect that each user has their own unique preferences and may prefer some genres 
# more than others. Let's take a look at a single user to see how their preferences 
# may vary. 



## ----Check to see variation for a single user by genre-------------------------------------------------------------

# Filter out users who have less than 200 ratings overall
edx_filter <- edx %>% 
  group_by(userId) %>%
  summarize(total_user_ratings = n()) %>%
  filter(total_user_ratings > 200)

# Pull a sample from the data set
set.seed(23, sample.kind = "Rounding")
edx_sample <- sample(edx_filter$userId, 1)

# Plot that user's ratings by genre with this code
edx %>% 
  filter(userId == edx_sample) %>%
  group_by(genres) %>%
  summarize(rating = mean(rating), n = n()) %>%
  ggplot(aes(x = genres, y = rating, size = n)) +
  geom_point() + 
  geom_text_repel(aes(label = genres), size = 1) +
  theme(axis.text.x = element_blank()) +
  ggtitle(paste0('User ID ', edx_sample, ': Average Rating as a Function of Genre')) +
  xlab('Genre') +
  ylab('Average Rating')



## ----Check for variation by release year---------------------------------------------------------------------------

# Test to see variation across release years -----------------------------

# In the same way that user's may have preferences for certain genres, they also may have 
# preferences for movies of a certain time. Some people like newer movies with their 
# increased pacing and improved graphics while others may be nostalgic for the style from 
# the 1970s and 1980s. It is worth exploring the variation those preferences create in our 
# data set to see if it can be exploited to improve our model.

# Create variable that groups ratings by release year
releaseRating <- edx %>%
  group_by(release_year) %>%
  summarize(release_rating = mean(rating), n = n())

# Plot average rating vs. release year with point size corresponding to number of ratings
ggplot(data = releaseRating, aes(x = release_year, y = release_rating)) +
  geom_point(aes(size = n)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Average Rating as a Function of Release Year') +
  xlab('Release Year') +
  ylab('Average Rating')



## ----Check to see if regularization is required for release year---------------------------------------------------

# Examine top and bottom 10 rated genres to see if regularization is required

# Bottom 10
slice_min(releaseRating, order_by = release_rating, n = 10)

# Top 10
slice_max(releaseRating, order_by = release_rating, n = 10)

# Regularization at the data set level doesn't appear to be necessary based on the number 
# of ratings for each but let's explore if going an additional level down, to the user 
# preference, may be worth inclusion in our model using the same user from the previous 
# section.



## ----Plot a single users preference to release year----------------------------------------------------------------

# Plot ratings by release year for a single user
edx %>% 
  filter(userId == edx_sample) %>%
  group_by(release_year) %>%
  summarize(rating = mean(rating), n = n()) %>%
  ggplot(aes(x = release_year, y = rating, size = n, label = release_year)) +
  geom_point() + 
  geom_text_repel() +
  theme(axis.text.x = element_blank()) +
  ggtitle(paste0('User ID ', edx_sample, 
                 ': Average Rating as a Function of Release Year')) +
  xlab('Release Year') +
  ylab('Average Rating')

# Looking a single user is not a sufficient sample size to correctly say that all users 
# have a preference for particular eras of movies, but it is reasonable to suspect that 
# be the case. We can use regularization at the user level to correct for this.



## ----Expand dataset to prepare for regularization tuning-----------------------------------------------------------
# Regularization Tuning ---------------------------------------------------

# Based on the exploration from the previous setting, we will try to develop and optimize 
# a recommendation system that considers a movie's average rating on a data set level and 
# genre and release year ratings based on each individual user's preference. The 
# information uncovered in the previous section will lead us to regularize for user 
# preference based on non-regularized release year average ratings and user preference 
# based on regularized genre average ratings. With four lambda parameters under 
# consideration, it is intuitive to suspect that each will be unique in an optimized 
# version of this model. The next section aims to perform that optimization. 

# The following section will only be run if the original_tuning variable is set to TRUE 
# at the beginning of the code.
if (original_tuning == TRUE){ 
  
  # This section creates the numerator and part of the 
  # denominator for the regularization equation 
  # in an effort to reduce the burden on the loop. 
  
  # Movie
  reg_movie_sum <- edx_train %>% 
    group_by(movieId) %>% 
    summarize(movie_s = sum(rating - mu),  movie_m = mean(rating), movie_n = n())   
  
  # Genre (on data set level)
  reg_genre_sum <- edx_train %>% 
    group_by(genres) %>% 
    summarize(genre_s = sum(rating - mu),  genre_m = mean(rating), genre_n = n()) 
  
  # Release year averages
  reg_relyr_mean <- edx_train %>% # Acquire the average ratings per release year to set up 
    group_by(release_year) %>%    # the regularization calculations based on them
    summarize(release_mean = mean(rating)) %>%
    ungroup()
  
  # Release year averages by user
  reg_relyr_sum <- edx_train %>%
    left_join(reg_relyr_mean, by = 'release_year') %>%
    group_by(userId, release_year) %>%  # Regularization (on data set level)
    summarize(relyr_s = sum(rating - release_mean),  
              relyr_m = mean(rating), 
              relyr_n = n()) 
  
  edx_train <- edx_train %>%  # Add the new columns to the default training set
    left_join(reg_movie_sum, by = 'movieId') %>%
    left_join(reg_genre_sum, by = 'genres') %>%
    left_join(reg_relyr_sum, by = c('userId', 'release_year'))
}
 
  


## ----Create Regularization Parameter Tuning Function---------------------------------------------------------------

# Lambda Parameters Tuning Function ---------------------------------------

# This section defines a function that will intake lambdas and a training set and output 
# an RMSE. It was originally designed to process four different lambdas, going through 
# each permutation within the desired range. Unfortunately, the computation for that was 
# on the order of days, if not weeks, and the code was streamlined to intake two 
# different lambdas, one for the parameters being tuned on the data set level and one 
# for the parameters being tuned on the user level. 

# This function is ultimately nested into the next function defined in this code, 
# designed to use cross validation to train and test the model on different slices of 
# the training set. 

# INPUTS 
# - lambdas: a single regularization parameter value for tuning
# - setLevel: determines whether testing takes place on data or user level
#             TRUE for data level lambdas, FALSE for user level lambdas
# - locked_lambda: defined value for lambdas not specific for testing in setLevel
# - cross_val_train: training set from one of the cross validation slices

# OUTPUTS 
# - RMSE for the given parameters

tuning_function <- function(lambdas, setLevel, locked_lambda, cross_val_train){
  
  # TRUE if training the lambda used for the 
  # dataset level regularization (movies, genre #1)  
  if(setLevel == TRUE){
    tune_lambda1 <- lambdas
    tune_lambda2 <- locked_lambda
  # FALSE if training the lambda used for the 
  # user level regularization (genre #2, release year)  
  } else { 
    tune_lambda1 <- locked_lambda
    tune_lambda2 <- lambdas
  }
  
  # Create a temporary training set based on the inputted cross validation selection 
  # and modify data frame to include regularization parameters based on previously 
  # added columns. 
  cross_val_temp <- cross_val_train %>% 
    mutate(movie_b = movie_s / (movie_n + tune_lambda1), 
           relyr_b = relyr_s / (relyr_n + tune_lambda2),
           genre_reg = genre_m + (genre_s / (genre_n + tune_lambda1)))
  
  # A second level of regularization is required for genres, accounting for user 
  # preference based on the regularized average rating of each genre, calculated 
  # in the previous step. 
  user_reg_genre <- cross_val_temp %>% 
    group_by(userId, genres) %>%
    summarize(genre_b = sum(rating - genre_reg)/(n() + tune_lambda2))
  
  # Add to overall table and create predictions
  cross_val_temp <- cross_val_temp %>%
    left_join(user_reg_genre, by = c('userId', 'genres')) %>%
    mutate(prediction = mu + movie_b + genre_b + relyr_b)
  
  # Keep track of progress with this line.
  print(paste0('Iteration complete with lambdas ', 
               tune_lambda1, ' and ', tune_lambda2, '.')) 
  
  # Calculate and return RMSE for an iteration
  RMSE(cross_val_temp$rating, cross_val_temp$prediction)
}



## ----Cross Validation with Tuning Function to Find Optimal Lambdas-------------------------------------------------

# Run Optimization Functions ----------------------------------------------

k = 10 # Set the number of cross validation samples to 10

if (original_tuning == TRUE){
  userLevel_lambdas <- seq(0.5, 15, 0.5) # Set range of lambdas for user level testing
  dataLevel_lambdas <- seq(5, 150, 5)    # Set range of lambdas for data level testing
  # The ranges above may seem arbitrary but were deemed appropriate based on 
  # experimentation with the functions and resulting RMSEs. 
  
  # The current version of the functions can only test a single lambda parameter at a 
  # time to save on computation time. Because of this, one has to be locked to start 
  # the process. The data level lambdas were selected for tuning first and the user 
  # level lambda was set at 3.
  
  # Tune the data level lambdas
  dataLevel_tune <- tuning_loop(tuning_lambdas = dataLevel_lambdas, 
                                k = k, 
                                lambda_dataset = TRUE, 
                                locked_lambda = 3)
  
  # Save the average RMSE across cross validation sets
  dataLevel_RMSE <- mean(dataLevel_tune$RMSE) 
  # Find the best lambda setting from the output for input into user level tuning
  finalData_lambda <- dataLevel_tune$Lambdas[which.min(dataLevel_tune$RMSE)]
  
  # Tune the user level lambdas with the tuned data level lambda from the previous step
  userLevel_tune <- tuning_loop(tuning_lambdas = userLevel_lambdas, 
                                k = k, 
                                lambda_dataset = FALSE, 
                                locked_lambda = finalData_lambda)
  
  # Save the average RMSE across cross validation sets
  userLevel_RMSE <- mean(userLevel_tune$RMSE)
  # Find the best user level lambda
  finalUser_lambda <- userLevel_tune$Lambdas[which.min(userLevel_tune$RMSE)]
  
} else {
  finalData_lambda <- 60   # Data level lambda as determined by optimization
  finalUser_lambda <- 1.5  # User level lambda as determined by optimization
}




## ------------------------------------------------------------------------------------------------------------------
# Cross Validation to Test Optimized Model on Different Sets --------------

if (original_tuning){
  # Update default training set by performing process with the final lambda 
  # parameters as determined in last step
  edx_train_complete <- edx_train %>% 
    mutate(movie_b = movie_s / (movie_n + finalData_lambda),
           relyr_b = relyr_s / (relyr_n + finalUser_lambda),
           genre_reg = genre_m + (genre_s / (genre_n + finalData_lambda)))
  
  user_reg_genre <- edx_train_complete %>% 
    group_by(userId, genres) %>%
    summarize(genre_b = sum(rating - genre_reg)/(n() + finalUser_lambda))
  
  edx_train_complete <- edx_train_complete %>%
    left_join(user_reg_genre, by = c('userId', 'genres')) 
  
  # Columns to add to data set
  # - movie_b: regularization parameter for each individual movie
  # - genre_b: regularization parameter for genre preference for each user
  # - relyr_b: regularization parameter for release year preference for each user
  
  # Isolate the columns desired and isolate the unique rows 
  movie_fct <- edx_train_complete %>% select(movieId, movie_b) %>% unique()
  genre_fct <- edx_train_complete %>% select(userId, genres, genre_b) %>% unique()
  relyr_fct <- edx_train_complete %>% select(userId, release_year, relyr_b) %>% unique()
  
  # Create a vector to contain the RMSE results 
  RMSE_testSets <- vector(mode = 'numeric', length = ncol(index))
  
  
  # Use cross validation to loop through the different test sets created in the 
  # introduction of this project
  for (i in 1:ncol(index)){
    
    edx_test <- edx[index[ , i], ]  # Create a temporary test set
    
    # Add columns to the test set and create predictions
    edx_test <- edx_test %>%
      inner_join(movie_fct, by = 'movieId') %>%
      inner_join(genre_fct, by = c('userId', 'genres')) %>%
      inner_join(relyr_fct, by = c('userId', 'release_year')) %>%
      mutate(prediction = mu + movie_b + genre_b + relyr_b)
    
    # Output RMSE values from each set
    RMSE_testSets[i] <- RMSE(edx_test$rating, edx_test$prediction) 
    
    # Keep track of progress as the loop runs.
    print(paste0('Iteration ', i, ' is complete.')) 
  }
  # Determine final RMSE for optimized model. 
  finalRMSE <- mean(RMSE_testSets)
  
}
 

## MODEL HAS BEEN OVERTRAINED, MUST CORRECT 
# While the overall RMSE looks good, tt becomes abundantly clear in looking at the 
# results that the model has been overtrained. The RMSE is quite high for the first 
# test set where the values not included in training the model while the remaining 
# 9, which probably are comprised signifantly of training material, are all 
# considerably lower. All of this points to overtraining and the model likely became 
# too granular in an attempt to adjust on the user level with the information 
# available. 



## ------------------------------------------------------------------------------------------------------------------

# Simplified Model --------------------------------------------------------

# The model obviously requires simplification to correct for the overtraining created 
# from the previous method. To do so, accounting for variation at the user level has 
# been eliminated from the model. Genre and release year will still be taken into 
# effect, but only on the data set level. It will also include a regularization 
# parameter for each user to account for their positivity or negativity as a rater 
# overall.

# Create a new default training set to optimize the model on. 
edx_train_simp <- edx[-index[ , 2], ]

# The following function replicates the process performed in the previous tuning 
# function but does not concern itself with any corrections within the user level. 
# Once again, this function has been set up to allow for cross validation.

# INPUTS 
# - lambda_simp: lambda regularization value for testing
# - cross_val_train: training set from cross validation slice

# OUTPUTS 
# - RMSE for the given lambda and training set 
tuning_function_simp <- function(lambda_simp, cross_val_train){
  
  # Create movie specific regularization parameter
  reg_movie_simp <- cross_val_train %>% 
    group_by(movieId) %>% 
    summarize(movie_b = sum(rating - mu)/(lambda_simp + n())) 
  # Replace with lambda_simp[1] if testing unique lambdas
  
  # Add regularized movie parameter to training set
  cross_val_temp <- cross_val_train %>% 
    left_join(reg_movie_simp, by = 'movieId')
  
  # Create genre specific regularization parameter
  reg_genre_sum_simp <- cross_val_temp %>% 
    group_by(genres) %>% 
    summarize(genre_b = sum(rating - mu - movie_b)/(lambda_simp + n())) 
  # Replace with lambda_simp[2] if testing unique lambdas
  
  # Create user specific regularization parameter
  reg_user_sum_simp <- cross_val_temp %>% 
    group_by(userId) %>% 
    summarize(user_b = sum(rating - mu - movie_b)/(lambda_simp + n())) 
  # Replace with lambda_simp[3] if testing unique lambdas
  
  # Create release year specific regularization parameter
  reg_relyr_sum_simp <- cross_val_temp %>% 
    group_by(release_year) %>% 
    summarize(relyr_b = sum(rating - mu - movie_b)/(lambda_simp + n())) 
  # Replace with lambda_simp[4] if testing unique lambdas
  
  # Add new correction terms to training set and create prediction
  cross_val_temp <- cross_val_temp %>%
    left_join(reg_genre_sum_simp, by = 'genres') %>%
    left_join(reg_user_sum_simp, by = 'userId') %>%
    left_join(reg_relyr_sum_simp, by = 'release_year') %>%
    mutate(prediction = mu + movie_b + genre_b + relyr_b + user_b)
  
  # Keep track of progress during operation
  # print(paste0('Iteration complete with lambda iteration ', 
  #               lambda_simp[1], ', ', lambda_simp[2], ', ',
  #               lambda_simp[3], ', ', lambda_simp[4], '.'))
  # print(paste0('Iteration complete with lambda iteration ', lambda_simp, '.'))
  # Return RMSE for the current iteration
  RMSE(cross_val_temp$rating, cross_val_temp$prediction)
  
}


## ------------------------------------------------------------------------------------------------------------------

# Simplified Looping Function ---------------------------------------------

# The following function replicates the previous looping function from the updated the 
# testing process to be the modified and simplified version that accounts for movie, 
# user, genre, and release year on the data set level.

# INPUTS 
# - tuning_lambdas_simp: vector or dataframe of lambda(s) for testing
#                        function has been modified to allow for unique lambdas for 
#                        each variable
# - k: number of cross validation slices to test on 

# OUTPUTS (housed within data frame)
# - testing_RMSEs: vector with RMSEs from optimized settings for each cross validation 
#                 sample
# - testing_lambdas: final lambda parameter used to create final RMSE for each cross 
#                    validation sample

tuning_loop_simp <- function(tuning_lambdas_simp, k){
  
  # Create cross validation slices from training set
  cross_val_idx <- createDataPartition(y = edx_train_simp$rating, 
                                       times = k, 
                                       p = 0.90, 
                                       list = FALSE)
  # Create output data frame 
  testing_RMSEs <- vector(mode = 'numeric', length = k)
  testing_lambdas <- vector(mode = 'numeric', length = k)
  
  ## If testing for multiple unique lambdas, replace the previous line with the following one
  # testing_lambdas <- data.frame(Movie_Lambda = numeric(0), 
  #                               User_Lambda = numeric(0), 
  #                               Genre_Lambda = numeric(0), 
  #                               ReleaseYear = numeric(0))
  
  for (i in 1:k){
    # print(paste0('Beginning slice  ', i, '.')) # Keep track of progress during operation
    
    # Define training and test set for each cross validation slice
    cross_val_train <- edx_train_simp[cross_val_idx[ , i], ]
    cross_val_test <- edx_train_simp[-cross_val_idx[ , i], ]
    
    # Run tuning function 
    ## Use this line if testing different lambdas for each correction parameter
    # lambda_tune <- apply(tuning_lambdas_simp, 1, 
    #                     tuning_function_simp, 
    #                     cross_val_train = cross_val_train)
    
    ## Use this line if testing a single lambda across all correction parameters
    lambda_tune <- sapply(tuning_lambdas_simp, 
                          tuning_function_simp, 
                          cross_val_train = cross_val_train)
    
    # Extract the lambda(s) that created the smallest RMSE and apply to test set
    train_lambdas <- tuning_lambdas_simp[which.min(lambda_tune)]
    ## If training multiple unique lambdas, replace the previous line with the 
    # following one
    # train_lambdas <- tuning_lambdas_simp[which.min(lambda_tune), ]
    
    # Repeat process to evaluate model on cross validation test set
    
    # Movie correction parameter
    reg_movie_simp <- cross_val_test %>% 
      group_by(movieId) %>% 
      summarize(movie_b = sum(rating - mu)/(train_lambdas + n())) 
    # lambda from tuning, replace with train_lambdas[1] for 
    # testing multiple unique lambdas  
    
    # Add as a column to the test set
    cross_val_test <- cross_val_test %>% 
      left_join(reg_movie_simp, by = 'movieId')
    
    # Genre correction parameter
    reg_genre_simp <- cross_val_test %>% 
      group_by(genres) %>% 
      summarize(genre_b = sum(rating - mu - movie_b)/(train_lambdas + n())) 
    # lambda from tuning, replace with train_lambdas[2] 
    # for testing multiple unique lambdas
    
    # User correction parameter
    reg_user_simp <- cross_val_test %>% 
      group_by(userId) %>% 
      summarize(user_b = sum(rating - mu - movie_b)/(train_lambdas + n())) 
    # lambda from tuning, replace with train_lambdas[3] 
    # for testing multiple unique lambdas
    
    # Release year correction parameter
    reg_relyr_simp <- cross_val_test %>% 
      group_by(release_year) %>% 
      summarize(relyr_b = sum(rating - mu - movie_b)/(train_lambdas + n())) 
    # lambda from tuning, replace with train_lambdas[4] 
    # for testing multiple unique lambdas
    
    # Add correction parameters to test set and create predictions
    cross_val_final <- cross_val_test %>%
      left_join(reg_genre_simp, by = 'genres') %>%
      left_join(reg_user_simp, by = 'userId') %>%
      left_join(reg_relyr_simp, by = 'release_year') %>%
      mutate(prediction = mu + movie_b + genre_b + relyr_b + user_b)
    
    # Keep track of progress throughout operation
    # print(paste0('Slice ', i, ' is complete.')) 
    
    # Add results to the outputted data frame
    testing_RMSEs[i] <- RMSE(cross_val_final$rating, cross_val_final$prediction)
    testing_lambdas[i] <- train_lambdas  
    ## If tuning multiple unique lambdas, replace the previous line with the following one
    # testing_lambdas[i, 1:4] <- train_lambdas 
    
  }
  # Return final data frame with optimized results from each cross validation test set
  return(cbind(testing_RMSEs, testing_lambdas))
}



## ------------------------------------------------------------------------------------------------------------------
# Cross Validation on Various Test Sets -----------------------------------

# Prior to running the optimization functions, a cross validation testing function is
# created that can take the optimal lambda parameters and apply them across each of 
# the test sets for a final calculation on the accuracy of the model.

# INPUTS 
# - movie_lambda: optimized movie specific regularization value from tuning
# - genre_lambda: optimized genre specific regularization value from tuning
# - user_lambda: optimized user specific regularization value from tuning
# - relyr_lambda: optimized release year specific regularization value from tuning

# OUTPUTS
# RMSE_testSets: vector with RMSE from each of the cross validation slices from
# overall edx set

cross_validation_test <- function(movie_lambda, genre_lambda, user_lambda, relyr_lambda){
  
  # Create output vector
  RMSE_testSets <- vector(mode = 'numeric', length = ncol(index))
  for (i in 1:ncol(index)){
    
    # print(paste0('Starting loop #', i)) # Keep track of progress during operation
    
    # Create the correction parameters based on the training portion of 
    # cross validation slice
    
    # Movie specific correction 
    reg_movie_simp <- edx[-index[ , i], ]  %>% 
      group_by(movieId) %>%                    # 
      summarize(movie_b = sum(rating - mu)/(movie_lambda + n())) 
    # lambda optimized in tuning function
    
   
    edx_final <- edx[index[ , i], ] %>% 
      inner_join(reg_movie_simp, by = 'movieId')
    
    # Genre specific correction
    reg_genre_simp <- edx_final %>% 
      group_by(genres) %>% 
      summarize(genre_b = sum(rating - mu - movie_b)/(genre_lambda + n())) 
    # lambda optimized in tuning function
    
    # User specific correction
    reg_user_simp <- edx_final %>% 
      group_by(userId) %>% 
      summarize(user_b = sum(rating - mu - movie_b)/(user_lambda + n())) 
    # lambda optimized in tuning function
    
    # Release year specific correction
    reg_relyr_simp <- edx_final %>% 
      group_by(release_year) %>% 
      summarize(relyr_b = sum(rating - mu - movie_b)/(relyr_lambda + n())) 
    # lambda optimized in tuning function
    
    # Apply these parameters to the test set for each cross validation slice
    edx_test <- edx[index[ , i], ]  
    
    # Keep track of progress during operation
    # print(paste0('Progress at iteration ', i, '.')) 
    
    # Combine correction parameters into test set data frame
    edx_test <- edx_test %>%
      inner_join(reg_movie_simp, by = 'movieId') %>%
      inner_join(reg_genre_simp, by = 'genres') %>%
      inner_join(reg_relyr_simp, by = 'release_year') %>%
      inner_join(reg_user_simp, by = 'userId') 
    
    # Make predictions
    edx_test <- edx_test %>%
      mutate(prediction = mu + movie_b + genre_b + relyr_b + user_b)
    
    # Correct for predictions under/over the possible min/max 
    # allowed in the rating system
    edx_test$prediction[edx_test$prediction > 5] <- 5
    edx_test$prediction[edx_test$prediction < 0.5] <- 0.5
    
    # Add to results
    RMSE_testSets[i] <- RMSE(edx_test$rating, edx_test$prediction)
    
    # Keep track of progress throughout operation
    # print(paste0('Slice ', i, ' is complete.')) 
  }
  # Return complete results
  return(RMSE_testSets)
}



## ------------------------------------------------------------------------------------------------------------------
# Tuning Simplified Model with Single Lambda ------------------------------

# Run function with a single lambda
if (simplified_tuning){
  
  # Create vector of testing lambdas
  simplified_lambdas <- seq(0.1, 15, 0.1)
  # Run function with cross validation
  simplified_tune <- tuning_loop_simp(tuning_lambdas_simp = simplified_lambdas, k = k) 
  # Maintain 10 cross validation slices
  
  # Extract best lambda parameter
  finalSimp_lambda <- simplified_tune[which.min(simplified_tune[ , 1]), 2]
  
  movie_lambda <- finalSimp_lambda # tuned movie lambda
  genre_lambda <- finalSimp_lambda # tuned genre lambda
  user_lambda <- finalSimp_lambda  # tuned user lambba
  relyr_lambda <- finalSimp_lambda # tuned release year lambda
  
  # Cross validation on test set to determine RMSE
  simplified_test <- cross_validation_test(movie_lambda, 
                                           genre_lambda, 
                                           user_lambda, 
                                           relyr_lambda)
  
  # Find average RMSE
  simplified_RMSE <- mean(simplified_test)
  
  # Print RMSEs from cross validation test sets
  simplified_tune
  
} else if (noTuning_oneLambda){ 
  
  # 0.3 was determined to be the optimal lambda based on testing done prior to submission
  movie_lambda <- 0.3
  genre_lambda <- 0.3
  user_lambda <- 0.3
  relyr_lambda <- 0.3
  
  # Cross validation on test set to determine RMSE
  tuning_test <- cross_validation_test(movie_lambda, 
                                       genre_lambda, 
                                       user_lambda, 
                                       relyr_lambda)
  tuning_RMSE <- mean(tuning_test)
}

# Print results
print(tuning_test)
print(tuning_RMSE)



## ------------------------------------------------------------------------------------------------------------------
# Tuning Simplified Model with Multiple Lambdas ---------------------------


# Run function with two lambdas
if (modified_tuning){
  # It still stands to reason that each of the four variables should have a unique 
  # regularization lambda in an optimized model. 

  multi_lambdas <- seq(0.5, 6.5, 1.5)
  
  # Create lambdas permutations to test
  multi_lambdas_perm <- permutations(n = length(multi_lambdas), 
                                     r = 4, 
                                     v = multi_lambdas, 
                                     repeats.allowed = TRUE)
  
  # Run updated model with unique lambdas for each regularization

  modified_tune <- tuning_loop_simp(tuning_lambdas_simp = multi_lambdas_perm, k = k)
  
  # tuned movie lambda
  movie_lambda <- modified_tune$Movie_Lambda[which.min(modified_tune$testing_RMSEs)] 
  # tuned genre lambda
  genre_lambda <- modified_tune$Genre_Lambda[which.min(modified_tune$testing_RMSEs)]
  # tuned user lambda
  user_lambda <- modified_tune$User_Lambda [which.min(modified_tune$testing_RMSEs)]
  # tuned release year lambda
  relyr_lambda <- modified_tune$ReleaseYear[which.min(modified_tune$testing_RMSEs)]
  
  tuning_test <- cross_validation_test(movie_lambda, 
                                       genre_lambda, 
                                       user_lambda, 
                                       relyr_lambda)
  tuning_RMSE <- mean(modified_test)
  
} else if (noTuning_multipleLambdas){ 
  
  # Optimal lambda settings for this particular unique multi lambda model
  movie_lambda <- 0.5 
  genre_lambda <- 0.5
  user_lambda <- 6.5
  relyr_lambda <- 6.5
  
  # Cross validation on test set to get results
  tuning_test <- cross_validation_test(movie_lambda, 
                                       genre_lambda,
                                       user_lambda, 
                                       relyr_lambda)
  tuning_RMSE <- mean(tuning_test)
  # Results were consistently higher than the single lambda model
}


#####################################################################


## ------------------------------------------------------------------------------------------------------------------

### RESULTS 

# This section will create the optimized and regularized correction parameters based on 
# the entire data set so those values can be applied to make predictions on the edx data 
# set, validation data set, and future recommendations that arise. 

# The inner join function has been used in combining the correction parameters with each 
# data set in order to ensure that no NaNs enter the data set and nullify the RMSE 
# calculations.


# Final Model -------------------------------------------------------------

# The four variables created in this section will correspond to predictions made based 
# on overall rating's average in addition to movie, genre, user, and release year 
# adjusted terms. A prediction can be made on any rating given that those four criteria 
# are available. 

# Final movie correction term
movie_correction <- edx  %>% 
  group_by(movieId) %>%                    
  summarize(movie_b = sum(rating - mu)/(movie_lambda + n())) 
# lambda optimized in tuning function

# Add movie specific correction test set
edx <- edx %>% 
  inner_join(movie_correction, by = 'movieId')

# Final genre specific correction
genre_correction <- edx %>% 
  group_by(genres) %>% 
  summarize(genre_b = sum(rating - mu - movie_b)/(genre_lambda + n())) 
# lambda optimized in tuning function

# Final user specific correction
user_correction <- edx %>% 
  group_by(userId) %>% 
  summarize(user_b = sum(rating - mu - movie_b)/(user_lambda + n())) 
# lambda optimized in tuning function

# Final release year specific correction
releaseYear_correction <- edx %>% 
  group_by(release_year) %>% 
  summarize(relyr_b = sum(rating - mu - movie_b)/(relyr_lambda + n())) 
# lambda optimized in tuning function



## ----Apply completed model to entire edx dataset to determine final RMSE-------------------------------------------

# edx Data Set ------------------------------------------------------------

# Add correction parameters to entire edx data set
edx_final <- edx  %>%
  inner_join(genre_correction, by = 'genres') %>%
  inner_join(releaseYear_correction, by = 'release_year') %>%
  inner_join(user_correction, by = 'userId')

# Create predictions
edx_final <- edx_final %>% mutate(prediction = mu + movie_b + genre_b + relyr_b + user_b)

# Confine low/high predictions within the limits of the recommendation system
edx_final$prediction[edx_final$prediction > 5] = 5
edx_final$prediction[edx_final$prediction < 0.50] = 0.5

# Calculate RMSE of model on edx data set
edx_RMSE <- RMSE(edx_final$rating, edx_final$prediction)
print(paste0('RMSE on the edx data set is ', edx_RMSE, '.'))


## ----Add parameters to validation set and prepare for validation testing-------------------------------------------

# Validation Data Set -----------------------------------------------------

# Create the necessary release year column in the validation data set
validation <- validation %>% 
  mutate(release_year = str_extract(title, "(?<=\\()\\d{4}?(?=\\))"))

# Add correction parameters to entire validation data set
validation <- validation %>%
  inner_join(movie_correction, by = 'movieId') %>%
  inner_join(genre_correction, by = 'genres') %>%
  inner_join(releaseYear_correction, by = 'release_year') %>%
  inner_join(user_correction, by = 'userId') 

# Create predictions
validation_final <- validation %>% 
  mutate(prediction = mu + movie_b + genre_b + relyr_b + user_b)

# Confine low/high predictions within the limits of the recommendation system
validation_final$prediction[validation_final$prediction > 5] = 5
validation_final$prediction[validation_final$prediction < 0.50] = 0.5

# Calculate RMSE of model on validation data set
validation_RMSE <- RMSE(validation_final$rating, validation_final$prediction)
print(paste0('RMSE on the validation data set is ', validation_RMSE, '.'))

#################################################################

### DISCUSSION


