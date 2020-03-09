## Project Title

Information Retrieval (Locally Sensitive Hashing)

### Goal and Process

The goal is to find similar pairs of reviews in the AmazonReviews dataset(108 Mb), also given any new review, this code will find the most similar one in the database.  

The process involves data preprocess (get rid of stopwords and punctuations), K-shingling of reviews (K=5), locally sensitive hashing (m times permutation & min-hashing & divided by b bands), compute Jaccard distance (1 - Jaccard similarity), and find similar pairs of reviews.

### Introduction

The 'lsh.py' takes 16 minutes to run in total (via Google Colab).  
After completion of each step (11 steps in total), there would be a reminder.  
In step 11, you can change the default input review to anything you want (string longer than 5) to search for similar review.

### Prerequisites

Put 'amazonReviews.json' in the same docment with 'lsh.py'.  
The dataset can be found here https://drive.google.com/file/d/1UMAL2OULAEpdhlSUSgtUMy7ErxVuYNdR/view?usp=sharing

### Data Visualization

For Step 4, I randomly picked 10000 pairs of reviews, and draw the distribution of their Jaccard distance, from which we can have a glimpse at how the whole AmazonReviews distinguish from each other.
![Image](https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20distance%20of%2010000%20pairs.png)  
For Step 5, The graph of probability of hit vs similarity with different parameters is plotted here so as to choose appropriate parameters, i.e. m permutations & b bands.  
![Image](https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png)  
For Step 9, I draw the distribution of Jaccard similarity in neareast duplicates, we can see that their Jaccard Similarity are very high since they are all similar pairs.  
![Image](https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20similarity%20distribution%20of%20nearest%20duplicates.png)  
For Step 11, if the input review is 'Love is the most important thing', the most similar review in the database would be found, together with ReviewerID and Jaccard similarity, which is listed below.

              ReviewerID: A3NTOYUJYVKOV7 
              ReviewText: "cats seem love important thing good sits dish" 
              Jaccard Similarity: 0.4117647058823529
