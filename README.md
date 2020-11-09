# Job-Skill-and-Title-Prediction
This project uses deep learning to predict the 'job title' and 'skill-set' from given 'description'

## Datasets were a mixture of features which could be broadly classified into:
1. Job_description → Skill 
   - Abilities to Work Activities
   - Abilities to Work Context
   - Skills to Work Context
   - Skills to Work Activities

2. Job description → job Title
   - Emerging Tasks
   - Occupation Data
   - Technology Skills
   - Tools
   - Work context

3. Job title → Skill
   - Abilities
   - Skills
   - Work Style

## Methodology :
Input: Job Description (Raw Text)

Output:
    -Job Title – (Single label, Multi Class)
    -Skill Set- (Multi label)
    
Datasets Under header 1 are merged as ‘jd_to_skill_final’
Datasets under header 2 are merged as ‘jd_title_final’.
I have built two independent Models, trained on the two datasets, predicting Job_title and Skill set
respectively.
Since the Title and Output both had to be predicted, I did not use the 3rd Dataset.

## Part – 1 Job title prediction:
  - Created a dictionary of unique job titles-(1110 unique values, large number of classes
    therefore very high accuracy is not expected)
  - Some labels had fewer than 100 examples, such labels have less chances of getting predicted
    and increase the number of classes, so I dropped those rows.
  - Generated integer encoding for the remaining classes
  - Performed pre-processing on the job_description, like stemming, lemmatization stopword
    removal.
  - Frequently occurring words which do not contribute to classification were also removed.
  
  ### Models:
Two types of models were considered:
   - 1D conv. Net with Keras embedding
   - Twin Bi-LSTM with Keras embedding
 
## Part 2 – Job skill prediction:
   -Created a dictionary of unique skills (83 unique skills, total rows = 848)
   - Dataset was too small, over-fitting, models saturating early is expected
   - Performed EDA, like a distribution and frequency of classes and word-cloud of most frequent
     words. Distribution was fair.
   - Initially tried encoding with label encoder, but I found out it did not work well with
     multilabel classifications
   - Encoded with LabelBinarizer as they work well with string labels, whereas One-Hot encoding
     requires them to be converted to integers first (although newer version of scikitlearn OHE
     supports strings as well)
   - Usual pre-processing is done on job description

## Models Used
(Vector Embedding: Fast-text)
  - CNN + Fast-Text
   - RNN + Fast-Text
   - Bi-LSTM with Fast-Text
   - Bi-GRU with Fast-Text

## Observations
   - Train-Test is split as (90-10)% as number of training samples were low
   - Model Used for final Prediction: CNN + Fast-Text – Simplest model among all, very significant
     boost in accuracy was not observed by using other models.
   - Further, models tend to saturate early due to low number of training samples, results cannot
     be improved by improving the model, unless we have a bigger dataset.
   - In all Models, the final output layer is a ‘dense layer’ with nodes=no_of_unique_skills(83),
     which predicts the probability of each label
   - The activation used is ‘Sigmoid’ as ‘Softmax’ is specifically found to unsuitable for multi-label
     classification, as ‘Softmax’ evaluates the class probability against other class probabilities,
     which means that the probability of labels will depend on each other
   - As we needed to predict around 20-30 labels for skills, I took an average of 25 highest
     probabilities
   - **Although by using threshold on probability score – only top priority skills can also be**
     **predicted**



