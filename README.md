# MovieRec

##### Goal 
I want to find good movies to watch.


##### Problem 
The problem is best framed as a unidimensional regression, which predicts how much I'm going to like a certain movie.


##### Hypothesis
My past movie ratings and MovieLens Datasets are an informative signal.  


##### Data
My IMDb [ratings](https://www.imdb.com/user/ur15834927/ratings) and
MovieLens 20m [dataset.](http://files.grouplens.org/datasets/movielens/ml-20m.zip) 
([readme](http://files.grouplens.org/datasets/movielens/ml-20m-README.html))  


##### Success Definition
The final test is with 100 randomly selected Never-Before-Seen movies 
that I have rated. The model is going to predict the order of these movies from worst to best.  
In the top 10 of the most promising movies there should be:  
* 8+ good movies
* 0 bad movies


##### Terms
![](images/green.png) Good movie - movies that I rated 9 stars or more out of 10.  
![](images/yellow.png) Average movie  
![](images/red.png) Bad movie =< 4 stars  


##### Final Test
I have 100 randomly selected movies that I've rated. One circle represents one movie:
![list_random.png](images/list_random.png)  
Each movie is color-coded for easier visualizing. Green = good movie, Red = Bad movie.

For the final test, the model is going to predict how much I'm going to like these 100 movies.  
If the model is successful then the movies that got good predictions also got good ratings from me.  

So the ideal output from the model,  
sorted from lowest predicted likeability to highest and  
color-coded with my real ratings:  
![list_ideal.png](images/list_ideal.png)  

Only the last 10 (highly recommended) movies have to be correctly predicted.  
There is a 2% probability that randomly sorted movies are correctly predicted. (8+ good, 0 bad)


##### Baseline Prediction
Hypothesis: My taste in movies is the same as the general population taste. So I should watch movies with the highest ratings. 

Here are randomly selected 100 movies ordered from lowest IMDb rating to the best.  
![list_imdb.png](images/list_imdb.png)  

There are 6 good and 4 average movies in the last 10 movies. Better than random but not good enough.


##### Possible Complication
* Lack of diversity  
For example, the model might only recommend 'war' movies.  

* Changing preferences  
The first movie I rated was from 2007. My taste in movies might now be different.  

* Old dataset  
The newest movie in the dataset is from 2015. The model is going to recommend only old movies.  


&nbsp;
## Data Analysis

In between 2007 and 2015, I rated 497 movies in IMDb.  

The MovieLens dataset contains 20000263 ratings across 27278 movies. These data were created by 138493 users between 1995 and 2015.
Users were selected at random and all of them had rated at least 20 movies.


To speed up the training I could use MovieLens ratings from the 2007-2015 period only.


#### Distribution of Ratings

IMDb ratings are made on a 10-star scale (1 star - 10 stars).
MovieLens ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars) but the half-star ratings are less frequently used.

To compare my ratings with MovieLens ratings I normalized them to 5-star scale, with one-star increments (1 star - 5 stars)

![plt_rating_bar.png](images/plt_rating_bar.png)![plt_rating_pie.png](images/plt_rating_pie.png)  
  

#### Most Popular Movies
Not to be confused with the best/highest rated movies.

Movie|Mean Rating|Number Of Votes|My Rating
---|---|---|---
Pulp Fiction (1994)|4.2|67310|4.0
Forrest Gump (1994)|4.0|66172|5.0
Shawshank Redemption, The (1994)|4.4|63366|5.0
Silence of the Lambs, The (1991)|4.2|63299|3.5
Jurassic Park (1993)|3.7|59715|n/a
  
  
#### Rare Movies
The least often rated movies that I've rated.  

Movie|Mean Rating|Number Of Votes|My Rating
---|---|---|---
Rolling (2007)|3.0|2|5.0
Justin Bieber: Never Say Never (2011)|1.5|2|0.5
  
  
#### Unusual Likes
The movies where my rating is higher than the average rating.  

Movie|Mean Rating|Number Of Votes|My Rating
---|---|---|---
Year One (2009)|2.5|605|5.0
Twilight (2008)|2.7|2156|5.0
  
  
#### Unusual Dislikes
The movies where my rating is lower than the average rating.  

Movie|Mean Rating|Number Of Votes|My Rating
---|---|---|---
Seventh Seal, The (Sjunde inseglet, Det) (1957)|4.1|4142|2.0
No Country for Old Men (2007)|4.0|10248|1.0
  
  
#### Genres
Plotting 500 most popular movie genres and genres from the movies that I have rated.  
Keep in mind that one movie could have multiple different genres.

![plt_genre_barh.png](images/plt_genre_barh.png)  
In the top 500 movies, there were 32 movies with 'War' genre.  
I've rated 29 movies with 'War' genre, 16 of them as good and 13 of them as average.  
The 29 movies mean rating is 4.3 which makes 'War' highest-rated genre.  
But most of the movies I've watched are 'Drama' (297 out of 497).  
  
I have avoided 'Musicals' and I don't like 'Film-Noir' (film marked by a mood of pessimism, fatalism, and menace).


#### Movie Age
Do I prefer old movies or new?  
'Movie age' 0 means that I rated the movie the same year it was released.

![plt_age_plot.png](images/plt_age_plot.png)
* 1/3 movies are up to 2 years old
* There is no clear correlation between move age and the average rating

&nbsp;
## Collaborative Filtering
&nbsp;

To solve the problem I'm going to try [FastaAi's](https://docs.fast.ai/collab.html) collaborative filtering which is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). For user taste information I'm only going to use ratings.  


The algorithm has a very interesting property of being able to do feature learning on its own, which means that it can start to learn for itself what features to use. Because of that, I don't have to use genres to train the algorithm.  

The model:
 ```python
class EmbeddingDotBias(nn.Module):
    def __init__(self, n_factors:int, n_users:int, n_items:int, y_range:Tuple[float,float]):
        super().__init__()
        self.y_range = y_range
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)]]

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        dot = self.u_weight(users) * self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]

n_factors = 30
y_range = (0.0, 5.5)
```

To speed up the training I'm going to filter my data. I take  
19% (5004 out of 26744) of most popular movies and  
63% (87099 out of 138494) of the most active user ratings. That's  
75% (14959080 out of 20000760) of the whole data. Now my movies x users matrix is about  
10x smaller (5004×87099=0.4e9 vs 26744×138494=4e9) which means that the training is about  
10x faster.  

Now I have a 15m dataset where every user has rated at least 100 movies and every movie has been rated at least ~400 times.  

### Result
After xxx epoch and xxx hours of training:  
Mean Absolute Error(MAE) = 0.xxx  
Mean Squared Error (MSE) = 0.xxx  

If the model predicts that I would rate the movie 7 stars out of 10, in reality, I would give xx-xx stars to the movie.






...

### Code
Pyton [notebook](https://github.com/korjusk/MovieRec/blob/master/MovieRec.ipynb) in 
[nbviewer](https://nbviewer.jupyter.org/github/korjusk/MovieRec/blob/master/MovieRec.ipynb)


##### Authors
* Kaur Korjus
