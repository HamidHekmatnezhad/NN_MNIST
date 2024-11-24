
# Neural Net - MNIST

This university project focuses on the study and implementation of neural networks for handwritten digit recognition using the MNIST dataset. The data is obtained from [this website](https://pjreddie.com/projects/mnist-in-csv/) in CSV format.

## Model Details
- **Activation Function**: **sigmoid**
- **Weight Distribution**: Normal distribution with a mean of zero, derived from the following relationships: $\pm1/\sqrt{incomin Links}$
                
## Model Accuracy and Optimization
The model has achieved an accuracy of **97.53%** with a learning rate of **0.07**, over **5 epochs**, and **600 nodes**. The model exhibits an accuracy of approximately **95%** on digits 7 and 2, while achieving around **99%** accuracy on digits 1 and 0.

The model requires optimization for new data, as it currently struggles with noisy images.

### Note on Optimization
To maintain the integrity of the model's performance on new datasets, it is crucial to optimize it following the established training protocols.

## Algorithms Used
In this project, several different neural network algorithms have been utilized:

- **MLP (Multi-Layer Perceptron)**: For learning complex, non-linear features from the data.
- **CNN (Convolutional Neural Network)**: For processing images and extracting features using convolutional and pooling layers. The concept of the kernel in CNN allows us to identify important features in images.
- **RNN (Recurrent Neural Network)**: For processing sequential data and recognizing temporal patterns.
- **LSTM (Long Short-Term Memory)**: An improvement over RNN that can capture long-term dependencies in the data.

## Requirements
- **matplotlib**==3.8.4
- **tensorflow**==2.18.0
- **imageio**==2.34.1
- **numpy**==1.26.4
- **scipy**==1.13.0

**[Article - in Persian](https://docs.google.com/document/d/1_BfeoZNyo_W1c6rmdG1JzUO3GuJJeaw0KLrDfq47dwc/edit?usp=sharing)**

## Note
You can test the network with new data, but because the network was trained with clean data and the digit recognition algorithm is not implemented, it cannot recognize noisy data and produces incorrect outputs.

**[Site on Streamlit Cloud](https://nnmnist.streamlit.app/)**
                