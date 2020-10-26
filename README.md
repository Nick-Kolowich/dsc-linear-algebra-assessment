# Assessment


```python
def mean_center(lst1):
    
    # calculate mean of the existing list
    mean = float((sum(lst1) / len(lst1)))
    
    # initialize an empty list
    new_list = []
    
    # subtract each item in the existing list from the mean
    # append to the new list
    for i in lst1:
        new_item = (i - mean)
        new_list.append(new_item)
    
    return print("mean centered:{}".format(new_list))
```


```python
def dot_product(lst1, lst2):
    
    # create a zipped version of the two input lists
    zipped = zip(lst1, lst2)
    zipped_list = list(zipped)
    
    # return the sum of the zipped list products
    return print("dot product = {}".format(sum([x*y for (x, y) in zipped_list])))
```


```python
list1 = [1, 2, 3, 4, 5]
list2 = [2, 4, 8, 16, 32]

mean_center(list1)
dot_product(list1, list2)
```

    mean centered:[-2.0, -1.0, 0.0, 1.0, 2.0]
    dot product = 258
    

# Standard Deviation Formula
![](images/standard-deviation.png)


```python
def standard_deviation(lst1):
    
    # calculate mean of the existing list
    mean = (sum(lst1) / len(lst1))
    
    # calculates the variance of the population by taking the mean of the (x-x_mu)^2 values
    variance = (sum([(x - mean) ** 2 for x in lst1])) / (len(lst1))
    
    # takes the square root of the variance to find the std. dev
    standard_deviation = variance ** 0.5
    
    return standard_deviation
```


```python
list3 = [1, 10, 100, 1000, 10000]

round(standard_deviation(list3), 4)
```




    3906.8974



# Covariance Formula
![](images/covariance.png)


```python
def covariance(lst1, lst2):
    
    # calculate the x mean of existing lst1
    x_mu = (sum(lst1) / len(lst1))
    
    # calculate the y mean of existing lst2
    y_mu = (sum(lst2) / len(lst2))
    
    # create a zipped version of the two input lists
    zipped = zip(lst1, lst2)
    zipped_list = list(zipped)
    
    cov = sum((x - x_mu) * (y - y_mu) for (x, y) in zipped_list) / len(lst1)
    
    return cov
    
```


```python
covariance(list1,list2)
```




    14.4



# Correlation Formula
![](images/correlation.png)


```python
def correlation(lst1, lst2):
    
    # employs two previous functions to calculate correlation
    corr = (covariance(lst1, lst2)) / ((standard_deviation(lst1)) * standard_deviation(lst2))
    
    return print("correlation = {:0.4f}".format(corr))
```


```python
correlation(list1, list2)
```

    correlation = 0.9333
    

# RMSE Formula
![](images/rmse.png)


```python
def rmse(ytrue, ypred):
    import numpy as np
    
    # finds the difference between y-pred and y-true and squares the values
    diff = np.array(ypred) - np.array(ytrue)
    diff_sq = diff ** 2

    # calculate the mean of squared differences
    diff_sq_mean = sum(diff_sq) / len(diff_sq)

    # finds rmse by taking the square root of the mean of the squared differences
    rmse = np.sqrt(diff_sq_mean)

    return print("RMSE = {:0.4f}".format(rmse))    
```


```python
rmse(list1, list2)
```

    RMSE = 13.4387
    

# RSS Formula 
![](images/rss.png)


```python
def rss(ytrue, ypred):
    import numpy as np
    
    rss = sum((np.array(ytrue) - np.array(ypred)) ** 2)

    return print("RSS = {}".format(rss))
```


```python
rss(list1, list2)
```

    RSS = 903
    
