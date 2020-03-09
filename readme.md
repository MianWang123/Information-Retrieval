## Project Title

Information Retrieval (Locally Sensitive Hashing)

## Goal and Function

The goal is to find similar pairs of reviews in the AmazonReviews dataset(108 Mb), also given any new review, this code will find the most similar one in the database.  
The process involves data preprocess (get rid of stopwords and punctuations), K-shingles of reviews (K=5), locally sensitive hashing (m times permutation & min-hashing & divided by b bands), compute Jaccard distance (1 - Jaccard similarity), and find similar pairs of reviews.

## Getting Started

The 'lsh.py' takes 16 minutes to run in total (via Google Colab).  
After completion of each step (11 steps in total), there would be a reminder.  
In step 11, you can change the default input review to anything you want (string longer than 5) to search for similar review.

### Prerequisites

Put 'amazonReviews.json' in the same docment with 'lsh.py'.  
The dataset can be found here https://drive.google.com/file/d/1UMAL2OULAEpdhlSUSgtUMy7ErxVuYNdR/view?usp=sharing


## Data Visualization

For Step 4, I draw the distribution(histogram) of 10000 pairs of reviews' Jaccard distance.  
![Image](https://github.com/MianWang123/Information-Retrieval/tree/master/pics/Jaccard distance of 10000 pairs.png)  
For Step 5, I draw the graph of probability of hit vs similarity with different parameters(m permutations&b bands).  
![Image](https://github.com/MianWang123/Information-Retrieval/tree/master/pics/probability of hit.png)  
For Step 9, I draw the distribution of Jaccard similarity in neareast duplicates.  
![Image](https://github.com/MianWang123/Information-Retrieval/tree/master/pics/Jaccard similarity distribution of nearest duplicates.png)  
For Step 11, if I input review 'Love is the most important thing', the system will find the most similar review in the database.

              ReviewerID: A3NTOYUJYVKOV7 
              ReviewText: "cats seem love important thing good sits dish" 
              Jaccard Similarity: 0.4117647058823529
