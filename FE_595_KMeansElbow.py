import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# We can use the elbow method to determine the number of clusters/classes
iris = load_iris()

# Convert to pandas dataset
data1 = pd.DataFrame(data= np.c_[iris['data'],iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(data1.head(5))

x = data1.iloc[:, [1, 2, 3, 4]].values
WithinClusterSumofSquares = []

print(x)
print(WithinClusterSumofSquares)

# testing a range 1-21 eventhough we know it should be 3 (i is range 1-21)
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i,
                    #we know this is 3 but i will give us range to test
                    init = 'k-means++',
                    max_iter = 300,
                    n_jobs=1,
                    precompute_distances='auto',
                    n_init = 10,
                    random_state = None,
                    tol=0.0001,
                    verbose=0)
    kmeans.fit(x)
    WithinClusterSumofSquares.append(kmeans.inertia_)

arrow_x = 3 #defining location of arrow
arrow_y = 50
label_x = 6 #defining location of text
label_y = 150

arrow_properties = dict(
    facecolor="black",
    width= 1.0,
    headwidth=4,
    shrink=1) #defining size of arrow

plt.plot(range(1, 21), WithinClusterSumofSquares,
         label = 'cluster',
         marker = 'o',
         linestyle = '--',
         color = 'r') #red
plt.xlabel('# of Clusters') # adding x label
plt.xticks(range(0,21,3)) # in intervals of 3 bc we know class is 3
plt.ylabel('Within Cluster Sum of Squares') # adding y label
plt.title('The elbow method for Iris DataSet') #adding title
plt.annotate("Elbow Occurs at 3 Clusters: 3 Classes", #add arrow
             xy=(arrow_x, arrow_y),
             xytext=(label_x, label_y),
             arrowprops=arrow_properties)
plt.legend() #adding legend to the top right of graph
plt.savefig("ElbowMethod_Iris.png", dpi = 100) #exporting png of graph
plt.show() #final graphical depiction output