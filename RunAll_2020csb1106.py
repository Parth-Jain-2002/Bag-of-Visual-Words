# Name : Parth Jain
# Entry No: 2020csb1106
# Assignment 2

# The visual words are stored in the directory closet Visual Words for later inspection
# Cluster.txt saves the cluster centroids after k-means clustering

import cv2
import numpy as np
import tensorflow as tf
from sklearn import svm
import matplotlib.pyplot as plt
from skimage.feature import SIFT
from scipy.cluster.vq import vq
from numpy.linalg import norm
from sklearn.metrics import classification_report

np.random.seed(42)

def euclidean_distance(x, y):
    '''
    Function to compute the euclidean distance 
    between two same dimensional objects
    '''
    return np.sqrt(np.sum((x-y)**2))

class KMeans:
    ''' 
    Class for k-means code
    '''
    def __init__(self,num_clusters=10,max_iterations=300,show_progress=False):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.show_progress = show_progress

        # Create a 2D-empty list of size of cluster number
        self.cluster_ids = []
        for _ in range(self.num_clusters):
            self.cluster_ids.append([])
        # List for storing the centroids
        self.centroids = []

    def fit(self,dataset):
        '''
        Fitting the K-means on the dataset
        '''
        self.dataset = dataset
        self.sample_count, self.feature_count = dataset.shape
        
        # Initialize centroids
        # Assign random points to be centroids at first
        random_samples_index = np.random.choice(self.sample_count,self.num_clusters,replace=False)
        self.centroids = []
        for idx in random_samples_index:
            self.centroids.append(self.dataset[idx])

        # Running through all the iterations
        for i in range(self.max_iterations):
                # To see the progress of the code
                if self.show_progress:
                    print(f"Iteration {i} completed")
                # Generating cluster membership based on the centroids
                self.cluster_ids = self.generate_clusters(self.centroids)

                # updating centroids based on the cluster membership
                # computing them to the mean of all the centroids
                centroids_old = self.centroids
                self.centroids = self.update_centroids(self.cluster_ids)

                # check if the old and new centroids converged or not
                if self.check_covergence(centroids_old,self.centroids):
                    break

        # return cluster labels
        return self.generate_labels(self.cluster_ids)

    def generate_labels(self,clusters):
        '''
        Assign all the points in the dataset their nearest centroid
        '''
        labels = np.empty(self.sample_count)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx   
        return labels

    def generate_clusters(self,centroids):
        '''
        Get all the cluster members assigned to the nearest centroid
        '''
        clusters = []
        for _ in range(self.num_clusters):
            clusters.append([])
        for index,feature in enumerate(self.dataset):
            centroid_index = self.compute_closet_centroid(feature,centroids)
            clusters[centroid_index].append(index)
        return clusters

    def compute_closet_centroid(self,feature,centroids):
        '''
        Computing the closet centroid to a given feature
        '''
        distances = norm(feature - centroids, axis=1)
        closet_index = np.argmin(distances)
        return closet_index

    def update_centroids(self,clusters):
        '''
        Updating the centroids based on the cluster membership
        '''
        centroids = np.zeros((self.num_clusters,self.feature_count))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.dataset[cluster],axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def check_covergence(self,centroids_old,centroids_new):
        '''
        Check if there is any difference between old and new centroids or not
        '''
        dists = norm(centroids_old - centroids_new)
        return np.sum(dists) == 0

def CreateVisualDictionary():
    '''
    Creating the visual dictionary (codebook) from the descriptors from all the train images
    '''
    k = 57
    # Can increase the number of iterations to 300 for better quality
    # Change the show progress to false to stop the iteration getting printed
    kmeans = KMeans(num_clusters = k,max_iterations = 100,show_progress = True)
    kmeans.fit(global_descriptors)
    codebook = kmeans.centroids

    # Saving the codebook(centroids) in the text file
    np.savetxt("cluster.txt",np.array(codebook))
    return codebook

def ClosetVisualWord(save=True,show=False):
    '''
    Saving the closet visual word to the centroid with their image and keypoint that represent it
    All the visual words are stored in the directory closet visual words
    '''

    keypoints_cv2 = {}
    descriptors_cv2 = {}
    extractor = cv2.SIFT_create()

    # Using the cv2 extractor for this, since the skimage keypoint didn't have size factor in it
    for index, img in enumerate(train_images):
        img_keypoints, img_descriptors = extractor.detectAndCompute(img,None);
        keypoints_cv2[index] = img_keypoints
        descriptors_cv2[index] = img_descriptors
    
    global_ind_cv2 = []
    global_keypoints_cv2 = []
    global_descriptors_cv2 = []

    for key,value in descriptors_cv2.items():
        if value is None:
            continue
        for i in value:
            global_ind_cv2.append(key)
            global_descriptors_cv2.append(i)

    for value in keypoints_cv2.values():
        if value is None:
            continue
        for i in value:
            global_keypoints_cv2.append(i)

    global_ind_cv2 = np.stack(global_ind_cv2)
    global_keypoints_cv2 = np.stack(global_keypoints_cv2)
    global_descriptors_cv2 = np.stack(global_descriptors_cv2)

    closet_word_index = {}

    for ind, cl in enumerate(codebook):
        # Euclidean distance between each centroid and entire global descriptors
        dist = np.linalg.norm(cl - global_descriptors_cv2, axis=1) 
        min_distance_index = np.argmin(dist)

        closet_word_index[ind] = {}
        closet_word_index[ind]["index"] = global_ind_cv2[min_distance_index]
        closet_word_index[ind]["keypoint"] = global_keypoints_cv2[min_distance_index]
        closet_word_index[ind]["descriptor"] = global_descriptors_cv2[min_distance_index]
        output_image = cv2.drawKeypoints(train_images[closet_word_index[ind]["index"]], [closet_word_index[ind]["keypoint"]], 0, (255, 0, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Saving the visual word for later inspection
        plt.imshow(output_image)
        if save:
            plt.title(f"Visual word corresponding to cluster {ind+1}")
            plt.savefig(f"./closetVisualWords/closet_word_{ind+1}")
        if show:
            plt.show()

def getOptimumK(start_cluster,end_cluster,step = 1):
    '''
    Function to get the optimum k using the elbow method
    '''
    keypoints = {}
    descriptors = {}
    extractor = SIFT()

    for index, img in enumerate(train_images):
        try:
            extractor.detect_and_extract(img)
            keypoints[index] = extractor.keypoints
            descriptors[index] = extractor.descriptors
        except:
            pass
    
    global_descriptors = []
    for key,value in descriptors.items():
        if value is None:
            continue
        for i in value:
            global_descriptors.append(i)

    global_descriptors = np.stack(global_descriptors)

    cluster_no = []
    WSS = []

    # Evaluting the k means on variable no of clusters and comparing them based on WSS
    for k in range(start_cluster,end_cluster,step):
        kmeans = KMeans(n_clusters = k,max_iter = 500)
        kmeans.fit(global_descriptors)
        cluster_no.append(k)
        WSS.append(kmeans.inertia_)

    plt.plot(cluster_no,WSS)
    plt.title(f"Number of clustes vs WSS from {start_cluster} to {end_cluster} with step{step}")
    plt.savefig("Elbow.png")
    plt.show()
        
    data = []
    data.append(cluster_no)
    data.append(WSS)
    np.savetxt("elbow.txt",data)   

def ComputeHistogram(feature_vector,visual_dict_matrix,plot=False):
    '''
    Computing the histogram from the feature vector representation and codebook
    '''

    # Using vector quantization to map the feature vector to the codebook
    visual_word, distance = vq(feature_vector,visual_dict_matrix)
    frequency_vector = np.zeros(num_clusters)

    # Converting this visual words to histogram
    for img_visual_word in visual_word:
        frequency_vector[img_visual_word]+=1
    
    # To plot the histogram
    if plot == True:
        plt.bar(list(range(10)),frequency_vector)
        plt.show()

    return frequency_vector

def MatchHistogram(tfidf_img):
    '''
    To match the test histogram with other histograms in the train set
    '''
    # Matching the test image histogram with train test images (60000) using cosine similarity
    # Multiply the array with matrix to speed up the operation
    matching = np.dot(tfidf_img, tfidf.T)/(norm(tfidf_img) * norm(tfidf, axis=1))

    # Using the top 25 most similar matching results
    top_results = 25
    idx = np.argsort(-matching)[:top_results]
    class_vector = np.zeros(10)
    for i in idx:
      class_vector[train_labels[fv_ind[i]]]+=1

    # Returning the class having the most matchings
    class_pred = np.argmax(class_vector)
    return class_pred

def PredImage(image,image_label,plot=False):
    # Extracting keypoints and descriptors on the test image
    extractor = SIFT()
    try:
        extractor.detect_and_extract(image)
        img_key = extractor.keypoints
        img_desc = extractor.descriptors
    except:
        img_desc = None

    # If the SIFT doesn't generate any keypoint on the image
    if img_desc is None:
      return 0,None,None

    # Computing the histogram for the test image descriptors
    histogram = ComputeHistogram(img_desc,codebook,plot=False)
    tfidf_img = histogram * idf
    # Using the match histogram to generate the label for the image
    class_pred = MatchHistogram(tfidf_img)

    if plot == True:
      plt.imshow(image)
      plt.show()
    
    # Returning the result according to the label predicted correct or not
    if(class_pred==image_label):
      return 1,class_pred,tfidf_img
    else:
      return 0,class_pred,tfidf_img

def Predict_test():
    '''
    Testing the code on the train images and 
    displaying overall classification accuracy, class wise accuracy, precision and recall
    '''
    y_test_label = []
    y_feature_vector = []
    kNN_pred = []
    correct_pred = 0

    for i in range(len(test_images)):
        # Calling pred image on each of the image to generate the label on it
        pred_res, pred_label, test_feature_vector = PredImage(test_images[i],test_labels[i])
        if pred_label is not None:
            correct_pred += pred_res
            kNN_pred.append(pred_label)
            y_test_label.append(test_labels[i])
            y_feature_vector.append(test_feature_vector)

    # Using the Support Vector Machine to generate the labels
    clf = svm.SVC(kernel='linear')
    clf.fit(tfidf,fv_label)
    
    # Classification report using the k-NN (k=25)
    report = classification_report(y_test_label, kNN_pred)
    
    print(f"Clusters-no: {num_clusters}")
    print(f"Classification report after doing k-NN (k=25)")
    print(report)

    SVM_pred = clf.predict(y_feature_vector)
    # Classificaton report using SVM (Support vector machine)
    report1 = classification_report(y_test_label,SVM_pred)
    print("Classification report after using SVM (Support Vector Machines)")
    print(report1)
     

if __name__=="__main__":
    # Functions created
    # 1. K Means class
    # 2. CreateVisualDictionary()
    # 3. ClosetVisualWord()
    # 4. getOptimumK() to get the optimum K
    # 5. ComputeHistogram()
    # 6. MatchHistogram()
    # 7. PredImage() to get the label for one image
    # 8. Predict_test() to test the bovw on the test images

    # Used the fashion_mnist dataset from the tensorflow
    # Used the skimage SIFT for extraction of features
    # Wrote the k-means clustering code for generating codebook
    # Used k-NN and SVM classifier for predicting labels
    # Used elbow method to find the optimum cluster number
    # Got 63% accuracy on one of the runs

    # Loading the dataset
    image_data = tf.keras.datasets.fashion_mnist
    (train_images,train_labels),(test_images,test_labels) = image_data.load_data()
    class_names_mapping = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Extracting the keypoints and the descriptors from the train images
    keypoints = {}
    descriptors = {}
    extractor = SIFT()

    for index, img in enumerate(train_images):
        try:
            # Using the skimage SIFT to extract the features
            extractor.detect_and_extract(img)
            keypoints[index] = extractor.keypoints
            descriptors[index] = extractor.descriptors
        except:
            pass
    
    # Creating a list of all descriptors for k-means clustering
    global_descriptors = []
    for key,value in descriptors.items():
        if value is None:
            continue
        for i in value:
            global_descriptors.append(i)

    print("Number of total descriptors of train images: ",len(global_descriptors))
    global_descriptors = np.stack(global_descriptors)
    #print(global_descriptors.shape)
    print("Features are extracted and saved")

    # Using the create visual dictionary function
    codebook = CreateVisualDictionary()
    print("Visual Dictionary is created")

    # Using the closet visual word to visualize what each centroid represent
    ClosetVisualWord()
    print("The Closet Visual Word has been stored in the directory closetVisualWords")
    
    num_clusters = 57
    frequency_vectors = []
    fv_ind = []
    fv_label = []
    for img_ind, img_feature_vector in descriptors.items():
        if img_feature_vector is None:
            continue
        # Using compute histogram to generate histogram from the image feature vector
        frequency_vectors.append(ComputeHistogram(img_feature_vector,codebook))
        fv_ind.append(img_ind)
        fv_label.append(train_labels[img_ind])

    frequency_vectors = np.stack(frequency_vectors)

    # Using the tf-idf to assign the weightage to each feature
    df = np.sum(frequency_vectors > 0, axis=0)
    N = 60000
    idf = np.log(N/ df)
    tfidf = frequency_vectors * idf

    # Using this Bag of Words, predicting the labels on the test images
    Predict_test()
    print(f"Prediction using both classifiers has been printed")
        
    



    