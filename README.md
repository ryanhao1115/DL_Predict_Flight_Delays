# Machine Learning Walk Through: Predicting Flight Delay
The last time you booked a flight, what were your key considerations? Price? Airline? Departure and arrival time? I think flight delays will definately be one in your mind. But how can we know the flight delay when we book the flight?
Bringing the question, this article describes the full process of building a Machine Learning solution. In this project, we only work with one month US domestic flights data.
## 1.Preparation and planning
Before we jump into the data pool, it is better to study the key reasons for flight delay. After searching in internet, I summarize top ten reasons:
![](https://github.com/ryanhao1115/DL_Predict_Flight_Delays/blob/main/p1.png)
The next step is to brainstorm where can we find the data that predicts the situation above. I use the below table to map the reasons and potential data.

According table above, I decided to use flight data and weather data for this solution.

## 2. Data Preparation and Exploratory Analysis.
Feature selection is one of the most important tasks in an ML project. The data downloaded from Flights API has over 50 data fields. According to the above mapping table, we can choose only 9 related fields, including carrier, flight number, flight date, departure airport, departure time, arrival airport, arrival time, flight distance, and delay in minutes.

Counting the number of flights per month per airport, we ranked the top 10 airports. The Origin Top 10 are the same as Destination Top 10( chart above). However, if we compare the top10 average delay(chart below), lists of destination airports and origin airports are different.

Although the top10 average delay looks horrible( according to chart of distribution of average delays below), they are just outliners.

From the Weather API, I choose 6 weather features — Temperature , Wind Speed, Visibility, Wind Gust, Cloud, Ice, plus airport, date and hour.
The next step is to combine flight data with weather data. The joining keys are aireport, date and hour. The merge operation needs to be done twice. One is to merge origin airports and departure time with weather data; the other is for destination and arrival. After merge, each flight record has two sets of weather features.

## 3. Feature Engineering
In the data science field, the data features determine the upper limit of machine learning, and the algorithm only approaches this upper limit as much as possible.
There are two main parts of feature engineering in the project. One is Categorical fields encoding; the other is target fields engineering.
The categorical fields, such as flight number, airports, have hundreds, and even thousands of classes. We can’t use the normal One-hot-encoding for them. I chose the Target Encoding, which is replacing a categorical value with the mean of the target variable.

The target field for this project is arr_dealy. It is in minutes. Hence, this is a regression problem, which means we try to predict the delay in minutes.

As shown in the above diagram, delay in minutes has a long tail, and a lot of data falls into negative values. This is because these numbers are differences between plan arrival time and actual arrival time. On one hand, the algorithm is very hard to predict an accurate delay in minutes with such data. On the other hand, customers might not care how many exactly minutes of delay. Therefore, I mapped the minutes into 5 categories:

This project became predicting a classification problem. It is also more useful for customers.
## 4. Deep Learning Model Training
I used Tensorflow Keras to build Deep Learning for this project.

During the training, I set an early-stop to save time.

The accuracy score is 75.77% in training data, and 72.06% in validation data. It is not high, cause I only used one month data for training. If I put more data, the model should perform better.

In conclusion, during a machine learning project, most of time and effort are spent on planning, data preparation and feature engineering. Building model and model training just consumed a small part. Again, quality of data decides the ceiling of a machine learning project’s result.

