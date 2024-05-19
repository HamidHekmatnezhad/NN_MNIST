# Neural Net - MNIST
It is a university project and an exercise for neural networks.

Data is obtained from this [website](https://pjreddie.com/projects/mnist-in-csv/) in CSV format. 
The activation function is **sigmoid**, and the weight distribution has been derived from the following relationships:
$\pm1/\sqrt{incomin Links}$ A normal distribution where the mean is zero.

## Model Accuracy and Optimization
The model has achieved an accuracy of **97.53%** with a learning rate of **0.07**, over **5 epochs**, and **600 nodes**. It exhibits the lowest accuracy of approximately **95%** on digits 7 and 2, while the highest accuracy is around **99%** on digits 1 and 0. 

For new data, the model requires optimization as it has been trained. It currently struggles with noisy images.

### Note on Optimization
To maintain the integrity of the model's performance on new datasets, it is crucial to optimize it following the established training protocols.


## Requirement
- **matplotlib**==3.8.4
- **imageio**==2.34.1
- **numpy**==1.26.4
- **scipy**==1.13.0

**[Artikel - in persian language](https://docs.google.com/document/d/1_BfeoZNyo_W1c6rmdG1JzUO3GuJJeaw0KLrDfq47dwc/edit?usp=sharing)**

## Note
You can test the network with new data, but because the network is trained with clean data and the number recognition algorithm is not implemented, it cannot recognize noisy data, and gives wrong output.

**[site on Streamlit cload](https://nnmnist.streamlit.app/)**
