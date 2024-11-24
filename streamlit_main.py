import streamlit as st
from neural_networks_model import CNN, LSTM, GRU, RNN, nn_MLP, nn_basic, kernel_conv
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def camera_data(pic):
    # Get camera data
    pic = Image.open(pic, mode='r').convert('L')
    pic = pic.resize((28, 28))
    pic = np.array(pic)
    pic = pic.astype('float32') / 255
    return pic


st.title('Neural Networks - Islamic Azad University of Ramsar')

options = ["Test NN", "Infos", "Links", "Brain Scan", "Optimization", ]
slctbx_1 = options[0]
slctbx_1 = st.segmented_control('Switch Slide:', options, selection_mode='single')

if slctbx_1:
    st.title(slctbx_1.upper())

if slctbx_1 == options[0]: # Test NN
    list_of_nn = ['NN Basic', 'MLP', 'CNN', 'RNN', 'LSTM', 'GRU', 'Kernel Convolution']
    sgn = st.segmented_control(
        "Function",
        list_of_nn,
        selection_mode='single',
        help="select one of the functions"
        )
    st.write('-------')

    if sgn == list_of_nn[0]: # NN Basic
        nn = nn_basic()

        rad = st.radio('Select Data: ', ['Test Data', 'Camera'])

        if rad == 'Camera':
            img = st.camera_input("Take a picture from digit") # image 28x28
            if img:
                img = Image.open(img, 'r').convert('L')
                img = img.resize((28, 28))
                img = np.array(img)
                img = img.reshape(784)
                result = nn.query(img)

                st.write(f"### Answer of Predition: **{np.argmax(result)}**")
            
        elif rad == 'Test Data':
            number = st.number_input('Enter n of example: ', min_value=1, max_value=5000, value=12, )
            btn = st.button('start', use_container_width=True)

            if btn:
                result, msgs, ims_idx = nn.test_data(number)
                score = 0
                for r in result:
                    if r == 'T':
                        score += 1
                score = score / number * 100
                st.write(f"#### Answer of Predition: {result},     score: {score}%")
                col = st.columns(3)
                j = 0
                while j < len(msgs):                  
                    for i in range(3):
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        if 'Correct' in  msgs[j]:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'green', 'pad': 4, 'edgecolor':'green', 'boxstyle':'round'})
                        else:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'red', 'pad': 4, 'edgecolor':'red', 'boxstyle':'round'})

                        ax.imshow(nn.x_test[ims_idx[j]], cmap='binary')
                        
                        col[i].pyplot(fig)
                        j += 1
    
    elif sgn == list_of_nn[1]: # MLP
        nn = nn_MLP()
        mdl = ['4 layers', ]
        col = st.columns(2)
        sgn_model = col[0].segmented_control('select model: ', mdl, selection_mode='single')
        rad = col[1].radio('Select Data:', ['Test Data', 'Camera'], index=0)

        if sgn_model == mdl[0]: # 4l
            nn.build_model('4l')
        
        if sgn_model and rad == 'Camera':
            img = st.camera_input("Take a picture from digit") # image 28x28
            if img:
                img = camera_data(img) # image 28x28
                result = nn.predict_model(img.reshape(1, 28, 28, 1))
                st.write(f"### Answer of Predition: **{np.argmax(result)}**")

        elif sgn_model and rad == 'Test Data':
            number = st.number_input('Enter n of example: ', min_value=1, max_value=5000, value=12, )
            btn = st.button('start', use_container_width=True)

            if btn:
                result, msgs, ims_idx = nn.predict_of_test_data(number)
                score = 0
                for r in result:
                    if r == 'T':
                        score += 1
                score = score / number * 100
                st.write(f"#### Answer of Predition: {result},     score: {score}%")
                col = st.columns(3)
                j = 0
                while j < len(msgs):                  
                    for i in range(3):
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        if 'Correct' in  msgs[j]:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'green', 'pad': 4, 'edgecolor':'green', 'boxstyle':'round'})
                        else:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'red', 'pad': 4, 'edgecolor':'red', 'boxstyle':'round'})

                        ax.imshow(nn.x_test[ims_idx[j]], cmap='binary')
                        
                        col[i].pyplot(fig)
                        j += 1
    
    elif sgn == list_of_nn[2]: # CNN
        nn = CNN()
        mdl = ['simple(1 layer)', '4 layers', '4 layer with dropout']
        col = st.columns(2)
        sgn_model = col[0].segmented_control('select model: ', mdl, selection_mode='single')
        rad = col[1].radio('Select Data:', ['Test Data', 'Camera'], index=0)

        if sgn_model == mdl[0]: # smpl
            nn.build_model('smpl')

        elif sgn_model == mdl[1]: # 4l
            nn.build_model('4l')

        elif sgn_model == mdl[2]: # dropout
            nn.build_model('dropout')

        if sgn_model and rad == 'Camera':
            img = st.camera_input("Take a picture from digit") # image 28x28
            if img:
                img = camera_data(img) 
                result = nn.predict_model(img.reshape(1,28,28,1))

                st.write(f"#### Answer of Predition: **{np.argmax(result)}**")

        elif sgn_model and rad == 'Test Data':
            number = st.number_input('Enter n of example: ', min_value=1, max_value=5000, value=12, )
            btn = st.button('start', use_container_width=True)

            if btn:
                result, msgs, ims_idx = nn.predict_of_test_data(number)
                score = 0
                for r in result:
                    if r == 'T':
                        score += 1
                score = score / number * 100
                st.write(f"#### Answer of Predition: {result},     score: {score}%")
                col = st.columns(3)
                j = 0
                while j < len(msgs):                  
                    for i in range(3):
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        if 'Correct' in  msgs[j]:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'green', 'pad': 4, 'edgecolor':'green', 'boxstyle':'round'})
                        else:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'red', 'pad': 4, 'edgecolor':'red', 'boxstyle':'round'})

                        ax.imshow(nn.x_test[ims_idx[j]], cmap='binary')
                        
                        col[i].pyplot(fig)
                        j += 1

    elif sgn == list_of_nn[3]: # RNN
        nn = RNN()
        mdl = ['simple(1 layer)', '4 layers']
        col = st.columns(2)
        sgn_model = col[0].segmented_control('select model: ', mdl, selection_mode='single')
        rad = col[1].radio('Select Data:', ['Test Data', 'Camera'], index=0)

        if sgn_model == mdl[0]: # smpl
            nn.build_model('smpl')

        elif sgn_model == mdl[1]: # 4l
            nn.build_model('4l')

        
        if sgn_model and rad == 'Camera':
            img = st.camera_input("Take a picture from digit") # image 28x28
            if img:
                img = camera_data(img) 
                result = nn.predict_model(img.reshape(1,28,28,1))

                st.write(f"#### Answer of Predition: **{np.argmax(result)}**")

        elif sgn_model and rad == 'Test Data':
            number = st.number_input('Enter n of example: ', min_value=1, max_value=5000, value=12, )
            btn = st.button('start', use_container_width=True)

            if btn:
                result, msgs, ims_idx = nn.predict_of_test_data(number)
                score = 0
                for r in result:
                    if r == 'T':
                        score += 1
                score = score / number * 100
                st.write(f"#### Answer of Predition: {result},     score: {score}%")
                col = st.columns(3)
                j = 0
                while j < len(msgs):                  
                    for i in range(3):
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        if 'Correct' in  msgs[j]:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'green', 'pad': 4, 'edgecolor':'green', 'boxstyle':'round'})
                        else:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'red', 'pad': 4, 'edgecolor':'red', 'boxstyle':'round'})

                        ax.imshow(nn.x_test[ims_idx[j]], cmap='binary')
                        
                        col[i].pyplot(fig)
                        j += 1

    elif sgn == list_of_nn[4]: # LSTM
        nn = LSTM()
        mdl = ['simple(1 layer)', ]
        col = st.columns(2)
        sgn_model = col[0].segmented_control('select model: ', mdl, selection_mode='single')
        rad = col[1].radio('Select Data:', ['Test Data', 'Camera'], index=0)

        if sgn_model == mdl[0]: # smpl
            nn.build_model('smpl')

        if sgn_model and rad == 'Camera':
            img = st.camera_input("Take a picture from digit") # image 28x28
            if img:
                img = camera_data(img)
                result = nn.predict_model(img.reshape(1,28,28,1))

                st.write(f"#### Answer of Predition: **{np.argmax(result)}**")

        elif sgn_model and rad == 'Test Data':
            number = st.number_input('Enter n of example: ', min_value=1, max_value=5000, value=12, )
            btn = st.button('start', use_container_width=True)

            if btn:
                result, msgs, ims_idx = nn.predict_of_test_data(number)
                score = 0
                for r in result:
                    if r == 'T':
                        score += 1
                score = score / number * 100
                st.write(f"#### Answer of Predition: {result},     score: {score}%")
                col = st.columns(3)
                j = 0
                while j < len(msgs):                  
                    for i in range(3):
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        if 'Correct' in  msgs[j]:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'green', 'pad': 4, 'edgecolor':'green', 'boxstyle':'round'})
                        else:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'red', 'pad': 4, 'edgecolor':'red', 'boxstyle':'round'})

                        ax.imshow(nn.x_test[ims_idx[j]], cmap='binary')
                        
                        col[i].pyplot(fig)
                        j += 1

    elif sgn == list_of_nn[5]: # GRU
        nn = GRU()
        mdl = ['simple(1 layer)', ]
        col = st.columns(2)
        sgn_model = col[0].segmented_control('select model: ', mdl, selection_mode='single')
        rad = col[1].radio('Select Data:', ['Test Data', 'Camera'], index=0)

        if sgn_model == mdl[0]: # smpl
            nn.build_model('smpl')

        if sgn_model and rad == 'Camera':
            img = st.camera_input("Take a picture from digit") # image 28x28
            if img:
                img = camera_data(img)
                result = nn.predict_model(img.reshape(1,28,28,1))

                st.write(f"#### Answer of Predition: **{np.argmax(result)}**")
        
        elif sgn_model and rad == 'Test Data':
            number = st.number_input('Enter n of example: ', min_value=1, max_value=5000, value=12, )
            btn = st.button('start', use_container_width=True)

            if btn:
                result, msgs, ims_idx = nn.predict_of_test_data(number)
                score = 0
                for r in result:
                    if r == 'T':
                        score += 1
                score = score / number * 100
                st.write(f"#### Answer of Predition: {result},     score: {score}%")
                col = st.columns(3)
                j = 0
                while j < len(msgs):                  
                    for i in range(3):
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        if 'Correct' in  msgs[j]:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'green', 'pad': 4, 'edgecolor':'green', 'boxstyle':'round'})
                        else:
                            ax.text(0, 0, msgs[j], bbox={'facecolor': 'red', 'pad': 4, 'edgecolor':'red', 'boxstyle':'round'})

                        ax.imshow(nn.x_test[ims_idx[j]], cmap='binary')
                        
                        col[i].pyplot(fig)
                        j += 1

    elif sgn == list_of_nn[6]: # Kernel Convolution
        conv = kernel_conv() # 478
        knls = conv.knls
        knls_name = ['Box Blur', 'Identity', 'Edge Detect', 'Edge Detect 2', 'Sharpen', 'Emboss', 'Gaussian Blur']
        pic = st.camera_input("Take a picture from digit") # image 28x28
        if pic:
            pic = Image.open(pic, mode='r')#.convert('L')
            pic = np.array(pic)
            st.write(pic.shape)

            imgs =[]
            pp = 0
            for knl in knls:
                    prgss = st.progress(pp, text=f"Working... {pp}%")
                    p = conv.convolution(pic, knl, dim='3d')
                    imgs.append(p)
                    pp += 15
                    if pp > 100:
                        pp = 100
                    prgss.empty()
        
            col = st.columns(3)
            i = 0
            while i < len(imgs):
                for j in range(3):
                    col[j].image(imgs[i].astype('uint8'), caption=f'{knls_name[i]}')
                    i += 1
                    if i >= 7:
                        break




elif slctbx_1 == options[1]: # Infos
    st.write('-------')
    
    st.markdown("""
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
                """)

elif slctbx_1 == options[2]: # Links
    st.write('-------')

    col1, col2 = st.columns(2)

    col1.markdown("""
    # [Github](https://github.com/HamidHekmatnezhad/NN_MNIST)
                  """)
    
    col2.markdown("""
    # [Artikel](https://docs.google.com/document/d/1_BfeoZNyo_W1c6rmdG1JzUO3GuJJeaw0KLrDfq47dwc/edit?usp=sharing)
                  """)

elif slctbx_1 == options[3]: # Brain Scan
    st.write('The image that the network has in relation to numbers')
    st.write('-------')
    
    col1, col2, col3 = st.columns(3)
   
    col1.image('img/brain_scan/BS_0.png')
    col1.image('img/brain_scan/BS_3.png')
    col1.image('img/brain_scan/BS_6.png')
    col1.image('img/brain_scan/BS_9.png')
    col2.image('img/brain_scan/BS_1.png')
    col2.image('img/brain_scan/BS_4.png')
    col2.image('img/brain_scan/BS_7.png')
    col3.image('img/brain_scan/BS_2.png')
    col3.image('img/brain_scan/BS_5.png')
    col3.image('img/brain_scan/BS_8.png')

elif slctbx_1 == options[4]: # Optimization
    st.write('-------')

    learning_rate = [0.01, 0.1, 0.2, 0.3, 0.6]
    performance_lr = [94.78, 97.21, 96.86, 96.24000000000001, 91.36]
    epoch = [1, 2, 3, 4, 5, 7, 10, 20]
    performance_e = [95.72, 96.61999999999999, 97.03, 97.42, 97.34, 97.52, 97.53, 97.21]
    nodes = [10, 100, 200, 250, 500, 1000]
    performance_n = [89.3, 96.76, 97.24000000000001, 97.28, 97.5, 97.55]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(3,1)
    ax[0].plot(learning_rate, performance_lr, linestyle='--', marker='o', color='y')
    ax[0].set_ylabel('Learning Rates')
    ax[1].plot(epoch, performance_e, marker='o', linestyle='--', color='r')
    ax[1].set_ylabel('Epochs')
    ax[2].plot(nodes, performance_n, marker='o', linestyle='--', color='b')
    ax[2].set_ylabel('Nodes')

    st.pyplot(fig)

    st.markdown("""
                ### Variations of network accuracy with respect to changes in _learning rate_, _epochs_ and number of _nodes_
                """)
