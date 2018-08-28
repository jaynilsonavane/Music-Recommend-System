# Music-Recommend-System

##Dependencies

numpy (http://www.numpy.org/)

scipy (https://www.scipy.org/)

pandas (https://wwww.pandas.org)

While googling around for a good dataset, I stumbled upon a page from 2011 with a bunch of cool datasets. Since I use Spotify and Pandora all the time, I figured I’d choose a music dataset.

The Last.fm data are from the Music Technology Group at the Universitat Pompeu Fabra in Barcelona, Spain. The data were scraped by Òscar Celma using the Last.fm API, and they are available free of charge for non-commercial use. So, thank you Òscar!

The Last.fm data are broken into two parts: the activity data and the profile data. The activity data comprises about 360,000 individual users’s Last.fm artist listening information. It details how many times a Last.fm user played songs by various artists. The profile data contains each user’s country of residence. We’ll use read.table from pandas to read in the tab-delimited files

To find out which artists are popular, we need to know the total play count of every artist. Since our user play count data has one row per artist per user, we need to aggregate it up to the artist level. With pandas, we can group by the artist name and then calculate the sum of the plays column for every artist. If the artist-name variable is missing, our future reshaping and analysis won’t work. So I’ll start by removing rows where the artist name is missing just to be safe.

Reshaping the Data

For K-Nearest Neighbors, we want the data to be in an m x n array, where m is the number of artists and n is the number of users. To reshape the dataframe, we’ll pivot the dataframe to the wide format with artists as rows and users as columns. Then we’ll fill the missing observations with 0s since we’re going to be performing linear algebra operations (calculating distances between vectors). Finally, we transform the values of the dataframe into a scipy sparse matrix for more efficient calculations

Fitting the Model

Time to implement the model. We’ll initialize the NearestNeighbors class as model_knn and fit our sparse matrix to the instance. By specifying the metric = cosine, the model will measure similarity bectween artist vectors by using cosine similarity.

