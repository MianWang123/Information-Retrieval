# -*- coding: utf-8 -*-
"""
ESE545-Data Mining Project: Information Retrieval (Locally Sensitive Hashing)
Author: Mian Wang  
Time: 2/15/20
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import itertools
from collections import Counter

def preprocess(s):
  # get rid of punctuations and stopwords
  new_s = ''
  for c in s:    
    new_s += c if c not in punctuations else ''
  s1 = ' '.join(word for word in new_s.split() if word not in stopwords)   
  return s1

def Kshingle(s, k):
  # divide the string s into k-shingles, return the list of shingles
  s_nospace = s.replace(' ','')
  n = len(s_nospace)
  if n<k: return None
  res = []
  for i in range(n-k+1):
    res.append(s_nospace[i:i+k])
  if res: return res
  else: return None
  
def indices(shingles):
  # convert the shingles to their indices  
  res = []
  for sh in shingles:
    res.append(all_shingles[sh])
  return res

def J_dist(indices, n):
  # calculate Jaccard distance of n pairs reviews, return their distances
  dis = []
  for i in range(n):
    a, b = indices.sample(n=2, replace=False)
    intersection = len(set(a).intersection(b))
    union = len(set(a).union(b))
    dis.append(1-intersection/union)
  return dis

def is_prime(number):
  # determine whether a number is prime or not
  if number==1 or number==2: return True
  for i in range(2, int(np.sqrt(number)+1)):
    if number%i == 0: return False
  return True

def m_hashes(indices, a, b, prime):
  # transform the indices to m hash values (m times permutation + min-hashing)
  r = np.array(list(indices)).reshape(1,len(indices))
  temp = (np.matmul(a,r) + b) % prime
  res = np.min(temp,axis=1)
  return res

def lsh(sig_matrix, b):
  # divide m columns(signature matrix) into b bands, calculate the sum of each band
  # return the index of reviews that have the same band sum (i.e. band[i][j]=band[i][k])
  r = sig_matrix.shape[1]//b
  res = []
  for i in range(b):
    band_i = np.sum(sig_matrix[:,i*r:(i+1)*r], axis=1)
    count = Counter(band_i)
    dup = [sum_i for sum_i in count if count[sum_i]>=2]
    for sum_i in dup:
      res.append(list(np.where(band_i==sum_i)[0]))
  return res

def J_sim(a, b):
  # calculate Jaccard similarity between 2 reviews
  intersection = len(set(a).intersection(b))
  union = len(set(a).union(b))
  sim = intersection/union
  return sim

def find_similar(review):
  # preprocess and do K shingles for the queried review,
  # store the indices of queried review's shingles
  review = preprocess(review.lower())
  shingles = Kshingle(review,5)
  idx = []
  for sh in shingles:
    if sh not in all_shingles: idx.append(0)
    else: idx.append(all_shingles[sh])

  # compute jaccard similarity between queried review and each review in the given set
  jaccard_sim = np.zeros(len(reviews))
  for i in range(len(reviews)):
    jaccard_sim[i] = J_sim(idx, set(reviews.loc[i]['indices']))

  # return the reviewerID with the largest jaccard similarity and its reviewText
  similar_idx = np.argmax(jaccard_sim)
  return reviews.loc[similar_idx]['reviewerID'], reviews.loc[similar_idx]['reviewText'], jaccard_sim[similar_idx]

if __name__=='__main__':
    reviews_dataset = pd.read_json('amazonReviews.json',lines=True)
    reviews = reviews_dataset[['reviewerID','reviewText']]
    
    stopwords_str = 'i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its \
    itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having \
    do does did doing a an the and but if or because as until while of at by for with about against between into through during before after \
    above below to up down in out on off over under again further then once here there when where why how all any both each few more most \
    other some such from no nor not only own same so than too very s t can will just don should now'
    stopwords = set(stopwords_str.split())
    punctuations = set([',','.','/','\'',';','[',']','(',')','~','<','>','?',':','\"','{','}','\\','|','`','-','!','@','#','$','%','^','&','*','=','+','-'])
    
    # Step 1: convert the text to lower case, and remove the punctuations & stopwords with self-defined function
    reviews.loc[:]['reviewText'] = reviews.loc[:]['reviewText'].apply(lambda x: str.lower(x))
    reviews.loc[:]['reviewText'] = reviews.loc[:]['reviewText'].apply(lambda x: preprocess(x))
    print('Step 1 done!')           
    
    # Step 2: do K-shingling of all Amazon reviews with self-defined function (K=5)
    shingles = []
    reviews['shingles'] = None
    for i in range(len(reviews)):
      reviews.loc[i]['shingles'] = Kshingle(reviews.loc[i]['reviewText'], 5)
      if not reviews.loc[i]['shingles']: continue
      shingles += reviews.loc[i]['shingles']
      
    # store all shingles together via hash table (key:shingles; value:index)
    all_shingles = {}
    for i,sh in enumerate(set(shingles)):
      all_shingles[sh] = i
    print('Step 2 done!')

    # Step 3: remove the reviews with empty shingles and reset the index
    reviews = reviews.dropna(subset=['shingles']).reset_index(drop=True)
    reviews['indices'] = reviews.loc[:]['shingles'].apply(lambda x: indices(x))
    print('Step 3 done!')
    
    # Step 4: randomly pick 10000 pairs of reviews, plot histgram of their Jaccard distance
    dis = J_dist(reviews['indices'], 10000)
    ave_dis = np.mean(dis)
    low_dis = min(dis)
    print(f'average J-dis:{ave_dis}, lowest J-dis:{low_dis}')
    
    plt.figure(0)
    plt.hist(dis,bins=15,rwidth=0.5,color='g')
    plt.title('histogram of 10000 Jaccard distance')
    plt.savefig('Jaccard distance of 10000 pairs.png')
    plt.show()
    print('Step 4 done!')
    
    # Step 5: plot the probability curve of lsh with different parameters (m permutations, b bands)
    m_candidates = {200,400,600}
    b_candidates = {20,40}
    s = np.arange(0,1,0.001)
    
    plt.figure(1)
    for m in m_candidates:
      for b in b_candidates:
        r = m//b
        plt.plot(s, 1-(1-s**r)**b, label={m,b})
    plt.vlines(0.8,0,1,linestyles='dashed')
    plt.legend()
    plt.title('m hashes divided by b bands')
    plt.xlabel('similarity')
    plt.ylabel('probability of hit')
    plt.savefig('probability of hit.png')
    plt.show()
    print('Step 5 done!')
    
    # Step 6: choose appropriate parameters from the probability curve, and the prime to do lsh
    m, b = 200, 20
    prime = len(all_shingles)
    while not is_prime(prime):
      prime += 1
    print('Step 6 done!')
    
    # Step 7: compute the signature matrix
    a_matrix = np.random.choice(range(1,prime+1),m).reshape(m,1) 
    b_matrix = np.random.choice(range(1,prime+1),m).reshape(m,1)
    sig_matrix = np.zeros((len(reviews),m), dtype=int)
    for i in range(len(reviews)):
      sig_matrix[i,:] = m_hashes(reviews.loc[i]['indices'], a_matrix, b_matrix, prime)
    print('Step 7 done!')
    
    # Step 8: do lsh to the signature matrix, and find duplicates with Jaccard similarity >= 0.8
    similar_pairs = lsh(sig_matrix, b)
    sim = {}
    nearest_neigh = []
    nearest_dup = []
    for i,idx in enumerate(similar_pairs):
      for (x,y) in itertools.combinations(set(idx),2):
        if (x,y) not in sim:
          jaccard_sim = J_sim(reviews.loc[x]['indices'], reviews.loc[y]['indices'])
          if jaccard_sim >= 0.8:
            nearest_neigh.append([x,y])
            nearest_dup.append([reviews.loc[x]['reviewText'], reviews.loc[y]['reviewText']])
          sim[(x,y)] = jaccard_sim
    print(f'The number of nearest duplicates: {len(nearest_dup)}')
    print('Step 8 done!')
    
    # Step 9: plot the distribution of Jaccard similarity of nearest duplicates
    l = []
    for (x,y) in nearest_neigh:
      l.append(sim[(x,y)])
    plt.figure(2)
    plt.hist(l,bins=10,rwidth=0.5,color='g')
    plt.title('distribution of jaccard similarity in nearest duplicates')
    plt.xlabel('similarity')
    plt.ylabel('frequency')
    plt.savefig('Jaccard similarity distribution of nearest duplicates.png')
    plt.show()
    print('Step 9 done!')
    
    # Step 10: store all the nearest duplicates as csv file
    with open('nearest_duplicates.csv','w') as f:
      file = csv.writer(f)
      file.writerow(['review1','review2'])
      file.writerows(nearest_dup)
    print('Step 10 done!')
      
    # Step 11: input a review, find the most similar review in the database
    review = input('Please Input a Review: ')
    #review = 'Love is the most important thing.'
    reviewer_id, review_text, similarity = find_similar(review)
    print(f'ReviewerID: {reviewer_id} \nReviewText: "{review_text}" \nJaccard Similarity: {similarity}')
    print('Step 11 done!')
