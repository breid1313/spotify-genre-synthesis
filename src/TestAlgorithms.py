from ClusteringAlgorithms import KMeans

import matplotlib.pyplot as plt
import random


testData = []
attributes = ['Attribute1', 'Attribute2']
for i in range(1000):
    testData.append({
        'Attribute1': random.randint(0,1000),
        'Attribute2': random.randint(0,1000)
    })

clusters = KMeans(3, 5, attributes, testData)

colors = ['red', 'green', 'blue']
for i in range(len(clusters)):
    xValues = list(map(lambda data: data['Attribute1'], clusters[i]['data']))
    yValues = list(map(lambda data: data['Attribute2'], clusters[i]['data']))

    plt.scatter(xValues, yValues, c=colors[i], alpha=0.8)



plt.title('Test K-Means')
plt.xlabel('Attribute1')
plt.ylabel('Attribute2')
plt.show()