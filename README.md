# Personal Portfolio 
## About Me
#### My name is `Jack Roberge`; I am a fourth year studying `Mathematics and Data Science` at `Northeastern University`, graduating in `December of 2025`. See a few samples of my personal projects thus far below! 


## LLM NLP Political Bias Analysis 

Folder: https://github.com/jroberge33/portfolio/tree/main/nlp_ai_polarization

#### **Question**: Can I use NLP to discern trends, predominant sentiment, and political biases in various LLMs by analyzing AI-generated output for a controlled set of queries, and does the style and bias of the given query affect the model's output? 

#### **Method**: 
- Simulate AI-generated responses for 6 different AI models, including Grok, DeepSeek, Claude, ChatGPT, Perplexity, and Gemini for numerous query cases (simplistic versus complex, biased versus unbiased, right-leaning political slant versus left-leaning political slant)
- Develop `NLP` text analysis `class`  ComparativeTextAnalysis to ensure modularity and possible application to alternate corpuses
- Produce sentiment comparisons, word length average charts, response complexity and word frequency `sankey` diagrams 

#### **Analysis**: 

- Most models' output was notably more complex (gauged by the Flesch Grade Score) and had higher average word lengths when fed a more complexly structured query (notably Grok and Gemini). DeepSeek was a clear exception, where the model's output complexity and word length are largely independent of the query's complexity.  
<img src="https://i.ibb.co/5gLvzk78/Screenshot-2025-07-06-at-4-05-44-PM.png">

- The output content of certain models (DeepSeek, Claude, and Grok) was more easily swayed politically than for other models based on the political slant of the prompt. With the prompt being one a conservative would respond positively to and a liberal negatively to, the sentiment was used to gauge political slant. Here, the bias of the prompt has a stark effect on the sentiment of the output. 
<img src="https://i.ibb.co/R47cSKbV/Screenshot-2025-07-06-at-4-18-49-PM.png">

- Meanwhile, for other models (notably Gemini), the sentiment of the response changed very little in the face of political bias in the prompt.
<img src="https://i.ibb.co/hP2KghM/Screenshot-2025-07-06-at-4-23-06-PM.png">

- Politically-biased model inputs produced outputs with word frequencies consistent with the views associated with that political slant.
<img src="https://i.ibb.co/cXr6dRTF/Screenshot-2025-07-06-at-4-24-18-PM.png">


## Evolutionary Computing Framework and Analysis 

Folder: https://github.com/jroberge33/portfolio/tree/main/evolutionary_computing

#### **Question**: Can I use randomized iterative evolutionary-style computing to better allocate teaching assistants to sections and office hours for an upcoming semester? 

#### **Method**: 
- Create a generalized `evolutionary computing` class that can perform 'evolution' for any dataset with the given set of 'agents of change' to produce random evolutionary change in the dataset towards an optimized solution
- Develop `profiler` wrapper to track function calls and timing of evolutionary agents in order to improve efficiency 
- Create defined evolutionary objectives, including minimal undersupported sections, minimal overscheduled teaching assistants, and minimal unpreferred scheduled sections as well as specific evolutionary randomizers ('agents of change')
- Apply class and evolutionary computing methods in coordination with a Northeastern professor who I supported as a teaching assistant in the spring of 2025. 


## Flight Traffic Dashboard 

Folder: https://github.com/jroberge33/portfolio/tree/main/flight_traffic_dashboard

#### **Question**: Can I use publicly available Department of Transportation flight traffic data to develop a multi-tab dashboard analyzing multiple dimensions of flight traffic trends? 

#### **Method**: 
- Obtain bi-annual data from the Department of Transportation on a host of metrics from hundreds of thousands of flights since 2018
- Create a flight frequency tab for any desired period(s) and states that produces a sankey diagram of states of origin and destination flight traffic during that period
- Develop a geographic visualization highlighting cities most frequently experiencing a specified type of delay
- Produce box plots analyzing average minutes of departure and arrival delays by type of delays for designated states and a specified time period 

#### **Analysis**: 

- Higher movements during summer months to warmer states and the inverse is true during colder months 
<img src="https://i.ibb.co/007cg4B/Screenshot-2025-07-06-at-5-28-56-PM.png">

- Weather delays are most common across colder states, as anticipated
<img src="https://i.ibb.co/663SGjQ/Screenshot-2025-07-06-at-5-28-46-PM.png">

- Weather delays on average cause the longest delays for both arrivals and departures, with security delays generally resolved much faster
<img src="https://i.ibb.co/KcpZZMLJ/Screenshot-2025-07-06-at-5-28-31-PM.png">


## Political Polarization NLP Analysis 

Folder: https://github.com/jroberge33/portfolio/tree/main/political_sentiment_analysis

#### **Question**: Can I use publicly available daily congressional record transcripts to analyze key political themes and political polarization trends over time?

#### **Method**: 
- Perform `web scraping` on publicly available Library of Congress congressional record data, iterating through hundreds of daily PDFs and aggregating transcript text
- Clean text data, removing `stopwords` and ensuring word validity
- Produce frequency and rolling avg frequency graphs for key words/trends of interest to highlight usage and importance over time
- Perform monthly `sentiment analysis` to determine overall political sentiment trends over time across the entire corpus of congressional transcripts and texts

#### **Analysis**: 

- Key trends' frequency and 12-month rolling avg frequency across period of analysis ('ukraine', 'pandemic', 'inflation') showcases trends consistent with current events and documented political trends 
<img src="https://i.ibb.co/cQ4CB5S/Screenshot-2024-07-15-at-1-58-25-PM.png">

<img src="https://i.ibb.co/gzZVjPj/Screenshot-2024-07-15-at-2-14-11-PM.png">

<img src="https://i.ibb.co/mtfNwdq/Screenshot-2024-07-15-at-2-15-56-PM.png">

- Sentiment analysis showcases a pointedly downward trend in political sentiment over time, a trend that markedly increased following the 2016 election
<img src="https://i.ibb.co/fNCDVwP/Screenshot-2024-07-15-at-2-18-23-PM.png">


## Discrete Random Variable Modeling and the Drake Equation 

Folder: https://github.com/jroberge33/portfolio/tree/main/discrete_random_variable_drake_eq

#### **Question**: Can I develop a discrete random variable modeling class to create an estimate of the number of intelligent and communicating civilizations in our universe using the Drake Equation?  

#### **Method**: 
- Develop a discrete random variable class that can perform mathematical calculations and manipulation involving normal, uniform, and other discrete random variables 
- Estimate key scientific values, including the % of planets with life, the chance of a population developing intelligence, the average lifespan of a civilization and more in order to effectively estimate the # of intelligent civilizations in our galaxy via the Drake Equation

#### **Analysis**: 

- Using educated guesses (emphasis on the guessing), my model produced a prediction of just under 9000 communicating extraterrestrial civilizations in our galaxy. 
<img src="https://i.ibb.co/xSspJKG8/Screenshot-2025-07-06-at-5-41-42-PM.png">


## Climate Data and Flight Delay Prediction Model 

Folder: https://github.com/jroberge33/portfolio/tree/main/weather_metrics_flight_delay_analysis

#### **Question**: Can I use weather metrics from the NOAA to predict flight delays at Logan Airport in Boston, MA?  

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

#### **Question**: Can I gain insights into the outcomes and dynamics of the game of Catan from data on game starting positions and outcomes?

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

#### **Question**: Can I identify key relationships between finish time and years and employ the resultant model to interpolate missing data and extrapolate future outcomes?

#### **Method**: 
- Take in data from multiple yearly Boston Marathon data files to identify key race outcomes and metrics
- Calculate the `correlation` between various variables in the marathon
- Produce `linear regression model` of average finish time vs years, using said model to interpolate missing data (2020 due to COVID pandemic)

#### **Analysis**: 

<img src="https://i.ibb.co/kK1qwc0/boston-marathon-normalized-lineplot.png">

<img src="https://i.ibb.co/1Q8pTpW/boston-marathon-linear-regression.png">

## Educational Outcomes and Attainment by Demographic Analysis through the COVID-19 Pandemic

Folder: https://github.com/jroberge33/portfolio/tree/main/pandemic_educational_outcomes_analysis

#### **Question**: Can I produce analytics and visualizations regarding the effect of the COVID-19 pandemic on educational metrics, through the lens of racial groups and states? 

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

#### **Question**: Can I use stochastic processes to simulate a tennis match's outcome based on starting serve and win probabilities?

#### **Method**:
- Model out probability outcomes based on each serving possibility for each game, set, and overall match using `markov chains` and other `stochastic modeling` concepts 
- Produce a flexible model that can be easily updated with different probabilities and starting conditions
