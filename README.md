# MovieRec

##### Goal 
I want to find good movies to watch.

##### The problem 
The problem is best framed as unidimensional regression, which predicts how much I'm going to like a certain movie.

##### Hypothesis
My past movie ratings and MovieLens Datasets are an informative signal.  

##### Data
My IMDb [ratings](https://www.imdb.com/user/ur15834927/ratings) and
MovieLens 20m [dataset.](http://files.grouplens.org/datasets/movielens/ml-20m.zip) 
([readme](http://files.grouplens.org/datasets/movielens/ml-20m-README.html))  

##### Success definition
Final test is with 100 randomly selected Never-Before-Seen movies 
that I have rated. The model is going predict the order of these movies from worst to best.  
In the top 10 of the most promising movies there should be:  
* 8+ good movies
* 0 bad movies

##### Terms
![](images/green.png) Good movie - movies that I rated 9 stars or more out of 10.  
![](images/yellow.png) Average movie  
![](images/red.png) Bad movie =< 4 stars  

##### Example
100 randomly sorted movies  

![list_random.png](images/list_random.png)  

Ideal output from model: movies sorted from worst to best.  

![list_ideal.png](images/list_ideal.png)  

Only the last 10 (highly recommended) movies have to be correctly predicted.  
There is 2% probability that randomly sorted movies are correctly predicted. (8+ good, 0 bad)

##### Baseline prediction
Hypothesis: My taste in movies is same as general population taste. So I should watch movies with highest ratings.  
Here are randomly selected 100 movies ordered from lowest IMDb rating to the best.  
![list_imdb.png](images/list_imdb.png)  

Better than random but not good enough.

##### Possible complication
* Diversity  
Example: Model will reccomend only 'war' movies.  

* Changing preferences  
The first movie I rated was from 2007. My taste in movies might now be different.  

* Old dataset  
The newest movie in dataset is from 2015. The model is going to reccomend only old movies.  

&nbsp;
## Data analysis

In between 2007 and 2015 I rated 497 movies in IMDb.  

MovieLens dataset contains 20000263 ratings across 27278 movies. These data were created by 138493 users between 1995 and 2015.
Users were selected at random and all of them had rated at least 20 movies.


To speed up the training I could use MovieLens ratings from the 2007-2015 period only.

#### Ratings

IMDb ratings are made on a 10-star scale (1 stars - 10 stars).
MovieLens ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars) but the half-star ratings are less frequently used.

To compare my ratings with MovieLens ratings I used 5-star scale, with one-star increments (1 stars - 5 stars)

![plt_rating_bar.png](images/plt_rating_bar.png)![plt_rating_pie.png](images/plt_rating_pie.png)

#### Most popular movies
Not to be confused with best/highest rated movies.

| Movie                    | Year | View count |
|--------------------------|:----:|-----------:|
| Pulp Fiction             | 1994 |      67310 |
| Forrest Gump             | 1994 |      66172 |
| The Shawshank Redemption | 1994 |      63366 |
| The Silence of the Lambs | 1991 |      63299 |
| Jurassic Park            | 1994 |      59715 |

#### Genres
Plotting 500 most popular movie genres and genres from the movies that I have rated.  
Keep in mind that one movie could have multiple different genres.

![plt_genre_barh.png](images/plt_genre_barh.png)

* Over half of the movies that I have rated are 'Drama'
* I have avoided 'Musicals'
* I don't like 'Film-Noir' (film marked by a mood of pessimism, fatalism, and menace). 
* I should watch more 'War' movies.

#### Movie Age
Do I prefer old movies or new?  
'Movie age' 0 means that I rated the movie the same year it was released.

![plt_age_plot.png](images/plt_age_plot.png)
* 1/3 movies are up to 2 years old
* There are no clear correlation between move age and average rating

&nbsp;
## Collaborative filtering
&nbsp;
To solve the problem I'm going to try [FastaAi's](https://docs.fast.ai/collab.html) collaborative filtering which is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). For user taste information I'm only going to use ratings. Later I might use genres, movie age and realease date aswell.  
...

### Code
Pyton [notebook](https://github.com/korjusk/MovieRec/blob/master/MovieRec.ipynb) in 
[nbviewer](https://nbviewer.jupyter.org/github/korjusk/MovieRec/blob/master/MovieRec.ipynb)

##### Authors
* Kaur Korjus
