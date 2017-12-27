
## Xiangyuan Ren （Shawn Ren）
###  Department of Electrical and Computer Engineering 
### Email: xir010@eng.ucsd.edu 
### Kaggle: https://www.kaggle.com/vertago
![image.png](http://upload-images.jianshu.io/upload_images/9147346-c4f92a52bd03acf4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
---
## Tasks (Visit prediction)
## Rank: 7

### For this task, I used two model, i.e.  One Class (BPR) and Cosine Similarity, which can separately get 90% and 87% accuracy respectively. 
### Then I ensemble them to get a 92% accuracy.

### BPR:
* Sample function: each time sample a pair consists of user, positive item and negative item.
* Update function: each time calculate the gradient of the three object and update their values simultaneously.
* Train function: within given iterations, sample and update repeatedly.
* Test function: test the cost function, to see whether to change learning rate.
* To get the better result, I set the lambda to -0.1,-0.1,-0.2 for user, positive items, negative items respectively.
* For stochastic gradient descent, I stop it when it vibrates when alpha is 0.01. Actually, it can be better, but I am not that patient to wait.
#### ( In the paper, it is said lambda should be positive, but this will lead to overflow.)

### Cosine Similarity: (Most students used Jaccard, but I get inspired from the BPR paper that cosine is better for this task)
* For each items b_i a user visit, build a set of the users that buy that item.
* For the target item b2, build the set for it as well.
* Use two set to compute the cosine similarity and add them up for all b_i.
* Test the threshold on validation set to find that it should be 1/7.

### Ensemble:
* Use the BPR model to get a score and find a good threshold (i.e. median score) by using the validation set 
* For those lower than threshold, change their status to positive if the Cosine similarity is larger than threshold.

---
## Tasks (Rating prediction)
## Rank: 23

### For this task, I used two models, i.e. models in homework3 and Latent Factor Model with Bias. 
Actually, I have found that when using the latter, though it is more complex and "seems" better, it will lead to worse performance due to too sparse dataset.
Here is the hint I posted on Piazza 10 days ago: (but looking at the leaderboard, not many students understand what I mean.)
![image.png](http://upload-images.jianshu.io/upload_images/9147346-c0457e5cda6df2c8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### For model in Homework3
* it is too easy for me to illustrate how to build the model.
* Build I notice that we did not use lambdaU and lambdaI for different variables, so it helps me improve some. The lambda_U is 3.7 and Lambda_I is 4.5
### *I do notice that the leaderboard kind of weird, lots of students improved their score simultaneously and got the same score. I think maybe you should check the report.  ( You know what I mean)*
 
#### For model of LFM with Bias
* There are two ways, i.e. Stochastic Gradient Descent and Alternate Least Square. I used ALS, because I am too lazy to change the learning rate by myself. 
* Besides, if calculating the gradient using Numpy instead of for-loop, it can accelerate a lot.
* Here is the code for LFM:

```python
def pro(u,b,K,j):
    ans=float(gu[u]*gi[b].T)
    if j==-1:
        return ans
    else:
        return ans-gu[u,j]*gi[b,j]

def ALS(lam):#输入alpha_T,train
    global alpha_T
    alphaold1=0
    count1=0
    while (abs(alpha_T-alphaold1)>1e-6):
        print '!';count1+=1; alphaold1=alpha_T; alphaold2=0;count2=0
        while (abs(alpha_T-alphaold2)>1e-4):
            if count2>3:
                break
            alphaold2=alpha_T
            alpha_T=sum([ sum(r-bu[u]-bi[b]-pro(u,b,K,-1) for (b,r) in ubr[u]) for u in ubr])/len(data)
            for u in ubr:
                bu[u]=sum( r-alpha_T-bi[b] -pro(u,b,K,-1) for (b,r) in ubr[u] )/(lam+len(ubr[u]))
            for b in bur:
                bi[b]=sum( r-alpha_T-bu[u] -pro(u,b,K,-1) for (u,r) in bur[b] )/(lam+len(bur[b]))
            for u in ubr:
                tmp0=np.zeros((1,K)) ;tmp1=np.ones((1,K))*lam
                for (b,r) in ubr[u]:
                    tmp0+=(np.array(r-alpha_T-bi[b]-bu[u]-pro(u,b,K,-1)) + np.array(gu[u])*np.array(gi[b]) ) * np.array(gi[b]) 
                    tmp1+=np.array(gi[b])**2
                gu[u]=np.array(tmp0)/np.array(tmp1)
#                 for k in range(K):
#                     tmp0=sum( ( r-alpha_T-bi[b]-bu[u]-pro(u,b,K,k) )*gi[b,k] for (b,r) in ubr[u])
#                     tmp1=lam+sum(gi[b,k]**2 for (b,r) in ubr[u])
#                     gu[u,k]=tmp0/tmp1
            count2+=1;print "first: ",alpha_T-alphaold2
        
        count3=0;alphaold2=0
        while (abs(alpha_T-alphaold2)>1e-4):
            if count3>4:
                break
            alphaold2=alpha_T
            alpha_T=sum([ sum(r-bu[u]-bi[b]-pro(u,b,K,-1) for (b,r) in ubr[u]) for u in ubr])/len(data)
            for u in ubr:
                bu[u]=sum( r-alpha_T-bi[b] -pro(u,b,K,-1) for (b,r) in ubr[u] )/(lam+len(ubr[u]))
            for b in bur:
                bi[b]=sum( r-alpha_T-bu[u] -pro(u,b,K,-1) for (u,r) in bur[b] )/(lam+len(bur[b]))
            for b in bur:
                tmp0=np.zeros((1,K)) ;tmp1=np.ones((1,K))*lam
                #tmp0=np.zeros(K) ;tmp1=np.ones(K)*lam
                for (u,r) in bur[b]:
                    tmp0+=(np.array(r-alpha_T-bi[b]-bu[u]-pro(u,b,K,-1)) + np.array(gu[u])*np.array(gi[b])) * np.array(gu[u])
                    tmp1+=np.array(gu[u])**2
                gi[b]=np.array(tmp0)/np.array(tmp1)                
#                 for k in range(K):
#                     tmp0=sum( ( r-alpha_T-bi[b]-bu[u]-pro(u,b,K,k) )*gu[u,k] for (u,r) in bur[b])
#                     tmp1=lam+sum(gu[u,k]**2 for (u,r) in bur[b])
#                     gi[b,k]=tmp0/tmp1
            count3+=1;print "second: ",alpha_T-alphaold2
        print "inner loop:",count2,count3
        testmse()
    print "outter loop: ",count1," alpha_diff: ",alpha_T-alphaold1

```
Hope it will help








