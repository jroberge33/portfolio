# Personal Portfolio 
## About Me
#### My name is `Jack Roberge`; I am a third year studying `Mathematics and Data Science` at `Northeastern University`, graduating in `December of 2025`. See a few samples of my work so far below! 



## Political Polarization NLP Analysis 

Folder: https://github.com/jroberge33/portfolio/tree/main/[_________]

#### **Question**: Can we use publicly available daily congressional record transcripts to analyze key political themes and political polarization trends over time?

#### **Method**: 
- Perform `web scraping` on publicly available Library of Congress congressional record data, iterating through hundreds of daily PDFs and aggregating transcript text
- Clean text data, removing `stopwords` and ensuring word validity
- 
- Perform monthly `sentiment analysis` 
- 

#### **Analysis**: 

- Linear regression was too simplistic of a model in using weather metrics to effectively predict the number of delayed flights
<img src="https://i.ibb.co/cQ4CB5S/Screenshot-2024-07-15-at-1-58-25-PM.png">

<img src="https://i.ibb.co/gzZVjPj/Screenshot-2024-07-15-at-2-14-11-PM.png">

<img src="https://i.ibb.co/mtfNwdq/Screenshot-2024-07-15-at-2-15-56-PM.png">

- However, the second decision tree iteration made clear that weather metrics were unable to accurately predict more granularity in the number of flight delays (groupings of low, moderate, high, very high monthly flight delays)
<img src="https://i.ibb.co/MVG9mpm/Decision-Tree-v2.png">

<img src="https://i.ibb.co/ncZ6PDz/Decision-Tree-v2-Confusion-Matrix.png">

- Certain weather metrics stood out as more impactful than others on flight delays - all making strong conceptual sense 
<img src="https://i.ibb.co/x2Jfbmm/Decision-Tree-v2-Feature-Importance.png">






## Climate Data and Flight Delay Prediction Model 

Folder: https://github.com/jroberge33/portfolio/tree/main/weather_metrics_flight_delay_analysis

#### **Question**: Can we use weather metrics from the NOAA to predict flight delays at Logan Airport in Boston, MA?  

#### **Method**: 
- Perform `multivariate linear regression` with continuous quantitative delay data to see whether or not monthly average flight metrics can predict number of flight delays
- Perform multiple iterations of a `decision tree` model using categorical delay data, to determine if weather metrics can predict higher/lower volumes of delays
- Perform `knn classification` to predict high/low monthly flight delay volumes to determine if weather metrics can produce more granular predictions of flight delays

#### **Analysis**: 

- Linear regression was too simplistic of a model in using weather metrics to effectively predict the number of delayed flights
<img src="https://i.ibb.co/BCBYY45/All-Weather-Metrics-Linear-Regression-Actual-vs-Predicted-Monthly-Flight-Delays.png">

- First decision tree iteration was relatively successful in predicting above or below average monthly flight delays  
<img src="https://i.ibb.co/hMK3rQF/decision-tree-v1.png" alt="decision-tree-v1">

<img src="https://i.ibb.co/7N0Dkvb/Decision-Tree-v1-Confusion-Matrix.png">

- However, the second decision tree iteration made clear that weather metrics were unable to accurately predict more granularity in the number of flight delays (groupings of low, moderate, high, very high monthly flight delays)
<img src="https://i.ibb.co/MVG9mpm/Decision-Tree-v2.png">

<img src="https://i.ibb.co/ncZ6PDz/Decision-Tree-v2-Confusion-Matrix.png">

- Certain weather metrics stood out as more impactful than others on flight delays - all making strong conceptual sense 
<img src="https://i.ibb.co/x2Jfbmm/Decision-Tree-v2-Feature-Importance.png">

## Catan Game Theory and Outcomes Analysis 

Folder: https://github.com/jroberge33/portfolio/tree/main/catan_analysis

#### **Question**: Can we gain insights into the outcomes and dynamics of the game of Catan from data on game starting positions and outcomes?

#### **Method**: 
- Analyze and visualize key game metrics, including roll probabilities, frequency of starting placement, and resource gains + losses
- Utilize multiple ML models to determine whether victory can be predicted based on starting resource and number placements, including `knn Classification` and `logistic regression`
- Employ `smote oversampling` to address a low count of wins in data as well as `principal component analysis` to address the high number of variables

#### **Analysis**: 

- Underwent multiple iterations in an attempt to increase model performance and refine the model to this dataset and situation
  
<img src="https://i.ibb.co/VHFvqDr/Screenshot-2024-01-30-at-11-23-54-PM.png">

- KNN classification model was unable to predict wins given their underrepresentation in the training data -> SMOTE synthetic oversampling rebalanced data set, improved precision and recall

<img src="https://i.ibb.co/hyt3Qy4/Screenshot-2024-01-30-at-11-31-02-PM.png">

## KNN Prediction Model for Bank Failures

Folder: https://github.com/jroberge33/portfolio/tree/main/bank_failure_analysis

#### **Question**: Can a KNN classification model be used to predict whether or not a bank will fail based on the bank's financial data and metrics?

#### **Method**: 
- Normalize and wrangle bank financial data to merge it with bank failure data
- Perform `knn classification` and `cross validation` across `accuracy, recall, and precision scores`
- Choose an ideal k value based on the desired/appropriate scoring metric
- Produce `confusion matrix heatmap` and visualize performance metrics

#### **Analysis**: 

- Accuracy, precision and recall metrics each highlight optimal value of k -> model trained on accuracy's optimal k value
  
<img src="https://i.ibb.co/TRFJ7Np/bank-failure-precision-lineplot.png">
<img src="https://i.ibb.co/6Nf3ddT/bank-failure-recall-lineplot.png">
<img src="https://i.ibb.co/wgx8WZm/bank-failure-accuracy-lineplot.png">

- Model is very high-performing in predicting bank failures based on key industry financial metrics
  
<img src="https://i.ibb.co/9gp5YpS/bank-failure-heatmap.png">

## Boston Marathon Linear Regression and Correlation Analysis 

Folder: https://github.com/jroberge33/portfolio/tree/main/boston_marathon_analysis

#### **Question**:  

#### **Method**: 
- Take in data from multiple yearly Boston Marathon data files to identify key race outcomes and metrics
- Calculate the `correlation` between various variables in the marathon
- Produce `linear regression model` of average finish time vs years, using said model to interpolate missing data (2020 due to COVID pandemic)

#### **Analysis**: 

<img src="https://i.ibb.co/kK1qwc0/boston-marathon-normalized-lineplot.png">

<img src="https://i.ibb.co/1Q8pTpW/boston-marathon-linear-regression.png">

## Educational Outcomes and Attainment by Demographic Analysis through the COVID-19 Pandemic

Folder: https://github.com/jroberge33/portfolio/tree/main/pandemic_educational_outcomes_analysis

#### **Question**: Can we produce analytics and visualizations regarding the effect of the COVID-19 pandemic on educational metrics, through the lens of racial groups and states? 

#### **Method**: 
- Visualize and analyze school enrollment, educational attainment, and test scores to determine how the pandemic impacted educational outcomes
- Segment data by race, age group, and attainment level to determine whether the pandemic had outsized impacts on certain groups
- Allow data to be presented for specific states and/or national level 

#### **Analysis**: 

- Significantly higher attainment in white and Asian populations relative to Latino and black populations in both high school and bachelor's degrees

<img src="https://i.ibb.co/RBwKk5p/race-highschool-degree.png">
<img src="https://i.ibb.co/qJjBHHn/race-bachelors-degree.png">

- Decline in overall school enrollments, with sharp declines in undergraduate and preschool age groups

<img src="https://i.ibb.co/gvXDgts/national-enrollment-change.png">
  
- Sharp drop in test scores across all subjects and age groups studied

<img src="https://i.ibb.co/6H4pprM/national-test-scores.png">

## Delta Sigma Pi Sigma Omega Application Interview Scheduling Code

Folder: https://github.com/jroberge33/portfolio/tree/main/interview_scheduler

#### **Question**: Can a program be used to schedule interviews for the fraternity's rush process, taking into account diversity in the interviewers; schedules of both interviewers and interviewees; and conflicts of interest between interviewers and interviewees?

#### **Method**: 
- Read in data from `SQL` and `Excel`, using `numerous complex data structures` and processes to schedule hundreds of interviews involving over 150 individual parties
- Effectively control for conflicts of interest, schedule conflicts, diversity of interviewer groups, and time constraints 
- Output polished and user-friendly schedules to Excel

## Markov Chain Tennis Simulation

Folder: https://github.com/jroberge33/portfolio/tree/main/markov_chains_tennis_analysis

#### **Question**: Can we use stochastic processes to simulate a tennis match's outcome based on starting serve and win probabilities?

#### **Method**:
- Model out probability outcomes based on each serving possibility for each game, set, and overall match using `markov chains` and other `stochastic modeling` concepts 
- Produce a flexible model that can be easily updated with different probabilities and starting conditions
