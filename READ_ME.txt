### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There is no installation required to complete the project.

This Project has been completed in Jupytor NoteBook.

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using AirBnB open data for Seattle(One of the US state) to better understand:
1) On which factor price is dependent?
2) What should be no. of amenities if someone wants to set up a new business in this?
3) Which are the important amenities and what causes the more price?
3) Building of predictive model for price.

I have built a price model where in I have selected features by experimenting with different methods.


## File Descriptions <a name="files"></a>

There are 2 notebooks available here to showcase work related to the above questions.  
1)Utilities.py >> Contains all Function used in inference and building of predictive model.
2)Airbnd>> for analyzing the data.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@josh_2774/how-do-you-become-a-developer-5ef1c1c68711).

The Price model however does not have a decent accuracy. Partly due to the fact, that I have neglected text based columns which require Sentiment based Analysis.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to AirBnb for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/airbnb/seattle/data).

Also, while building the model, the methods I have used for Feature Extraction were inspired from [here](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b) and [here](https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf)
Otherwise, feel free to use the code here as you would like! 