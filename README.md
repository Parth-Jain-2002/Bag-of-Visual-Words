# Bag-of-Visual-Words

## Functions created:-
1. K Means class 
2. CreateVisualDictionary() 
3. ClosetVisualWord()
4. getOptimumK() to get the optimum K 
5. ComputeHistogram() 
6. MatchHistogram() 
7. PredImage() to get the label for one image 
8. Predict_test() to test the BOVW on the test images

### In the project, 
-> Used the fashion_mnist dataset from the tensorflow <br/>
-> Used the skimage SIFT for extraction of features <br/>
-> Wrote the k-means clustering code for generating codebook <br/>
-> Used k-NN and SVM classifier for predicting labels <br/>
-> Used elbow method to find the optimum cluster number <br/>
-> Stored the closet visual words in the closetVisualWords directory <br/>
-> Got 60-65% accuracy on multiple runs (Variation due to randomness in k-means) <br/>

![image](https://user-images.githubusercontent.com/72060359/204859919-07fb510c-3998-4505-9d84-305d4f9861d1.png)

![image](https://user-images.githubusercontent.com/72060359/204859134-fdcc157d-0d16-4580-9810-770e1299b44a.png)

![image](https://user-images.githubusercontent.com/72060359/204858468-c6c255d9-0ddd-4977-aeac-51b644d38b4c.png)

![image](https://user-images.githubusercontent.com/72060359/204858360-8cef0681-f981-45fe-85a8-7072ffbc8e8f.png)






