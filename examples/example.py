import torch
import numpy as np
from timefly.kmeans import TimeSeriesKMeans
from timefly.soft_dtw import SoftDTW

if not torch.cuda.is_available():
    print('CUDA is not available. do u have a GPU?')
    exit()

device = 'cuda'

n_clusters = r
n_features = 1
time_series_len = 1000
num_series = 40000

X_train = torch.rand(num_series, time_series_len, n_features)

# 1. create an instance of the TimeSeriesKMeans class
k_means = TimeSeriesKMeans(
    max_iter=10,
    random_state=np.random.RandomState(0),
)

# 2. fit the model
k_means.fit(X_train)

# 3. predict the labels
prediction = k_means.predict(X_train[0].unsqueeze(0))
print(f"Predicted cluster: {prediction}")