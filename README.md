# Kaggle---Google-Local-visit-prediction
## Predict whether users will visit a business using BPR
* Who should read this: People want to know how to implement BPR *(Bayesian Personalized Ranking from Implicit Feedback)* 
https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf
* Competition description: Predict given a (user,business) pair from 'pairs Visit.txt' whether the user visited the business (really, whether it was one of the business they reviewed). Accuracy will be measured in terms of the categorization accuracy (fraction of predictions you got right).
* Data description: 
    - **businessID**: The ID of the business. This is a hashed product identifier from Google.
    - **userID**: The ID of the reviewer. This is a hashed user identifier from Google.
    - **rating**: The star rating of the user’s review.
    - **reviewText**: The text of the review. It should be possible to successfully complete this assignment without making use of the review data, though an effective solution to the category prediction task will presumably make use of it.
    - **reviewHash**: Hash of the review (essentially a unique identifier for the review).
    - **unixReviewTime**: Time of the review in seconds since 1970.
    - **reviewTime**: Plain-text representation of the review time.
    - **categories**: Category labels of the product being reviewed.
* Basic Concept of BPR:
    - Maximize the differences bewteen items that users prefer and not prefer, with a Gaussian distribution prior
    - Decompose the high dimensional users vectors into low dimensional ones, and still remain such differenes.
    - Regularization parameters on User, Positive item (prefered), Negative item (not prefered)
    - Stochastic gradient descent
* Functions in the codes:
    - Sample function: each time sample a pair consists of user, positive item and negative item.
    - Update function: each time calculate the gradient of the three object and update their values simultaneously.
    - Train function: within given iterations, sample and update repeatedly.
    - Test function: test the cost function, to see whether to change learning rate.
    - To get the better result, I set the lambda to -0.1,-0.1,-0.2 for user, positive items, negative items respectively.
    - For stochastic gradient descent, I stop it when it vibrates when alpha is 0.01. Actually, it can be better, but I am not that patient to wait.
#### ( In the paper, it is said lambda should be positive, but this will lead to overflow.)
