# Beer Recommender

The Github repository uses the Beer Advocate training set (https://data.world/socialmediadata/beeradvocate) to train a matrix factorization model to recommend new beers to users.
We use two models:
1. A standard matrix factorization model written in PyTorch
2. The matrix factorization model provided by LightFM.

The model trained in `train-final-model.ipynb` can be used to recommend beers either from the buyer or seller side.

## Interaction Matrix
The dataset gives reviews from 1 to 5 stars.
We either use this directly in the factorization (for the PyTorch model) or convert this to a -1, 0, 1 matrix or 0 and 1 matrix.
The -1, 0, 1 matrix represents bad reviews as a -1 making them worse than not trying a beer at all while the 0 and 1 matrix sets bad reviews and no reviews to be equivalent.

## Hyperparameter Optimization
We performed hyperparameter grid searches on parameters for creating the interaction matrix and the models themselves.
For the LightFM models we varied the number of latent dimensions, the loss function used, k for the Warp-KOS loss, whether a -1, 0, 1 matrix or 0, 1 matrix was used, the threshold to convert a review to a 1 rather than -1 or 0.

## Notebook toc

- light-fm-models.ipynb: Performs a hyperparameter grid search over LightFM models.
- train-final-model.ipynb: Trains, evaluations, and saves the final model with optimal hyperparameters.
