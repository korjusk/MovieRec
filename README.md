# MovieRec

##### Goal 
Personal movie recommendations system.

##### Hypothesis
My past movie ratings and MovieLens Datasets are an informative signal.  

##### Data
My IMDb [ratings](https://www.imdb.com/user/ur15834927/ratings) and
MovieLens 20m [dataset](http://files.grouplens.org/datasets/movielens/ml-20m.zip) 
and dataset [readme.](http://files.grouplens.org/datasets/movielens/ml-20m-README.html)  

##### Hypothesis test
[FastaAi's](https://docs.fast.ai/collab.html) collaborative filtering model.  

##### Success
Final test is with 100 randomly selected Never-Before-Seen movies 
that I have rated. The model is going predict the order of these movies from worst to best.  
In the top 10 of the most promising movies there should be:  
* 8+ good movies
* 0 bad movies

##### Terms
![](images/green.png) Good movie - 9 stars or more out of 10. (my personal rating)  
![](images/yellow.png) Average movie  
![](images/red.png) Bad movie =< 4 stars  

##### Example
100 randomly sorted movies  
![random.png](images/random.png)  

Ideal output from model
![ideal.png](images/ideal.png)  

Only the last 10 (highly recommended) movies have to be correctly predicted.  
There is 2% change that randomly sorted movies are correctly predicted. (8+ good, 0 bad)

##### Code
Pyton [notebook](https://github.com/korjusk/MovieRec/blob/master/MovieRec.ipynb) in 
[nbviewer](https://nbviewer.jupyter.org/github/korjusk/MovieRec/blob/master/MovieRec.ipynb)

##### Authors
* Kaur Korjus
