# Databricks notebook source

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import math

ratings_data=sc.textFile('/FileStore/tables/ratings.dat')
ratings_data = ratings_data.map(lambda x:x.split("::"))
ratings_data = ratings_data.map(lambda tokens: (tokens[0],tokens[1],tokens[2]))
train_rdd,test_rdd = ratings_data.randomSplit([6,4],seed = 0)
test_for_predict = test_rdd.map(lambda x:(x[0],x[1]))


minimum_error = float('inf')

seed = 5
iterations = 25
ranks = [10,16,20]
errors = [0,0,0,0]
err = 0
tolerance = 0.02
lmda = 0.1

for rank in ranks:
    model = ALS.train(train_rdd , rank , seed = seed, iterations = iterations , lambda_=lmda)
    predictions = model.predictAll(test_for_predict).map(lambda r: ( ((r[0],r[1]),r[2])   )  )
    test_pred_join=test_rdd.map(lambda x: ( (( int(x[0]),int(x[1])),float(x[2]))   ) ).join(predictions)
    error = math.sqrt(test_pred_join.map(lambda x: (x[1][0] - x[1][1])**2).mean())
    errors[err] =error
    err +=1    
    print("For rank %s the MSE is %s" %(rank,error))  
    if(error < minimum_error):
        minimum_error = error
        highest = rank
        
        
print("The best trained model : Rank %s" %highest)

