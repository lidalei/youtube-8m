# This project was done to participate in the [YouTube-8M Video Understanding Competition on Kaggle](https://www.kaggle.com/c/youtube8m).
## Work
* Implemented ML-kNN that supports tuning k (passing a list of k), File [ml-knn.py](../master/ml-knn.py)
* Implemented k-means that supports removing empty or small clusters during each iteration and supports Cosine and Euclidean distance, File [kmeans.py](../master/kmeans.py)
* Implemented linear regression that does not require data to be shifted and supports tuning l2 regularization (passing a list of l2 reg. rates), Class LinearClassifier in [linear_model.py](../master/linear_model.py)
* Implemented one-vs-rest logistic regression that supports class weights and instance weights. More importantly, it supports any feature transformation implemented in Tensorflow, Class LogisticRegression in [linear_model.py](../master/linear_model.py)
* Implemented stable standard scale transformation that supports any feature transformation, function compute_data_mean_var in utils.py and example in [rbf_network.py](../master/rbf_network.py)
* Implemented multi-label RBF network that supports three-phase learning and experimented with the hyper-parameters, such as the number of centers and scaling factors, [rbf_network.py](../master/rbf_network.py)
* Implemented multi-layer neural networks and experimented with architecture, batch normalization and dropout and bagging, [mlp_fuse_rgb_audio.py](../master/mlp_fuse_rgb_audio.py)
* Implemented inference that supports bagging many models, Class BootstrapInference in [bootstrap_inference.py](../master/bootstrap_inference.py)
* Finally, the implementations can be easily adapted to other large datasets.
## Presentation
See [presentation.pdf](../master/presentation.pdf)

## Paper
See [http://oa.upm.es/55867/](http://oa.upm.es/55867/)
