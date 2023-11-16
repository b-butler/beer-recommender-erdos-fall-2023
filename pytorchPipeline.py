#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import clean
import process
import pandas as pd
import numpy as np


# In[2]:


def get_ratings(style = 'range', threshold = 2.5):
    """Gets the ratings/interaction/user-item matrix
    in the form we need to apply the model.

    Parameters
    ----------
    style: str, optional
        Options: 'range', 'pos_neg', 'zero_one'
        The format of the ratings matrix (see process.py)
        Defaults to 'range'
    threshold: int, optional
        The cutoff threshold for "liking" a beer; only
        matters if style = 'pos_neg' or style = 'zero_one'
        (see process.py)
        Defaults to 2.5
    """
    if style not in ['range', 'pos_neg', 'zero_one']:
        raise Exception('Not a valid choice for style.')
        
    reviews = pd.read_csv('https://query.data.world/s/55cb4g2ccy2sbat45jzrwmfjfkp2d5?dws=00000')
    reviews.review_time = pd.to_datetime(reviews.review_time,unit = 's')
    clean_reviews = clean.remove_null_rows(reviews)
    clean_reviews = clean.remove_dup_beer_rows(clean_reviews)
    clean_reviews = clean.merge_similar_name_breweries(clean_reviews)
    clean_reviews = clean.merge_brewery_ids(clean_reviews)
    
    if style == 'range':
        return process.InteractionMatrixTransformer(clean_reviews).to_range().tocsr()
    if style == 'pos_neg':
        return process.InteractionMatrixTransformer(clean_reviews).to_positive_negative(threshold).tocsr()
    if style == 'zero_one':
        return process.InteractionMatrixTransformer(clean_reviews).to_zero_one(threshold).tocsr()


# In[3]:


class MatrixFactorization(torch.nn.Module):
    """Matrix factorization model.

    - Initializes user and item embeddings (default to dimension 10).
    - Forward pass is just a dot product between user and item latent vectors.
    """
    def __init__(self, num_users, num_items, emb_size = 10):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, emb_size)
        self.item_emb = torch.nn.Embedding(num_items, emb_size)
        # initializing weights
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)

    def forward(self, user, item):
        return (self.user_emb(user) * self.item_emb(item)).sum(1)


# In[4]:


def train_epochs(model, ratings, epochs = 10, lr = 0.001, wd = 0.0):
    """Trains the matrix factorization model.

    Parameters
    ----------
    model: MatrixFactorization
        The instance of a matrix factorization model to train
    ratings: scipy.sparse._csr.csr_matrix
        The ratings/interaction/user-item matrix
    epochs: int, optional
        Number of epochs in training
        Default is 10
    lr: float, optional
        Learning rate
        Default is 0.001
    wd: float, optional
        weight decay
        Default is 0.0
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)  # learning rate
    for i in range(epochs):
        model.train() # put model in training mode (?)
        
        # Set gradients to zero
        optimizer.zero_grad()
        
        # Shuffle our data (Why?)
        rows, cols = ratings.nonzero() # (rows[i], cols[i]) across all i will be the coordinates of all nonzero entries of ratings.
        #p = np.random.permutation(len(rows))
        #rows, cols = rows[p], cols[p] # this just shuffles the data

        pred = model(torch.tensor(rows), torch.tensor(cols))
        actual = torch.tensor(ratings.tocsr()[rows, cols], dtype = torch.float32).squeeze() # squeeze just reshapes to the appropriate dim
        
        loss = F.mse_loss(pred, actual)

        # Backpropagate
        loss.backward()
        #print(model.user_emb.weight[0,0])

        # Update the parameters
        optimizer.step()
            
        print("train loss %.3f" % loss.item())
    print('Final RMSE: %.4f' % loss.item()**(1/2))


# In[5]:


def recall_at_k_range(model, ratings, user, threshold = 2.5, k = 10):
    """Compute recall at k for a given user in the case
    where the ratings matrix was built using to_range.
    This is the ratio of number of items the user 
    actually liked (determined by threshold) in the 
    top k predictions made by the model over the total 
    number of items the user actually liked.

    Parameters
    ----------
    ratings: scipy.sparse._csr.csr_matrix
        The ratings/interaction/user-item matrix
    user: int
        The index of the user for whom you want to compute recall at k
    threshold: float, optional
        The threshold used to determine if a user actually likes a beer
        Defaults to 2.5
    k: int, optional
        The k parameter
        Defaults to 10
    """
    num_items = ratings.shape[1]
    preds = model(torch.tensor(user), torch.arange(num_items)) # predicated ratings of all beers
    topk_inds = torch.topk(preds, k).indices # indices of the top k predictions
    
    # get indices of the liked beers
    x = torch.tensor(ratings[user].toarray())
    inds_of_liked_beers = torch.where(x > threshold, 1, 0).squeeze().nonzero() # puts 1 in indices where x > thresh, 0 else then gets the nonzero indices
    
    if len(inds_of_liked_beers) == 0:
        return 'no liked beers'
    
    intersect = np.intersect1d(topk_inds, inds_of_liked_beers)
    
    return len(intersect)/len(inds_of_liked_beers)


# In[6]:


def recall_at_k_nonrange(model, ratings, user, k = 10):
    """Compute recall at k for a given user in the case
    where the ratings matrix was built using to_zero_one
    or to_positive_negative. This is the ratio of number
    of items the user actually liked (appear with 1 for
    this user in the ratings matrix) in the top k predictions
    made by the model over the total number of items the user
    actually liked.

    Parameters
    ----------
    model: MatrixFactorization
        The instance of a matrix factorization model to be evaluated
    ratings: scipy.sparse._csr.csr_matrix
        The ratings/interaction/user-item matrix
    user: int
        The index of the user for whom you want to compute recall at k
    k: int, optional
        The k parameter
        Defaults to 10
    """
    num_items = ratings.shape[1]
    preds = model(torch.tensor(user), torch.arange(num_items)) # predicated ratings of all beers
    topk_inds = torch.topk(preds, k).indices # indices of the top k predictions
    
    # get indices of the liked beers
    x = torch.tensor(ratings[user].toarray())
    inds_of_liked_beers = torch.where(x == 1, 1, 0).squeeze().nonzero() # puts 1 in indices where x == 1, 0 else then gets the nonzero indices
    
    if len(inds_of_liked_beers) == 0:
        return 'no liked beers'
    
    intersect = np.intersect1d(topk_inds, inds_of_liked_beers)
    
    return len(intersect)/len(inds_of_liked_beers)


# In[ ]:




