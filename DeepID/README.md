## DeepID (Deep Learning face representation from predicting 10,000 classes)

Herein, China involved in the face recognition competition with its prestigious academic institution as well. Researchers of the Chinese University of Hong Kong announced two different version of DeepID model for face recognition tasks.

It is referred as Deep Hidden Identity feature(DeepID) for face verification. It is a face verification algorithm.

It extract high level identity features from the face image.

### Model Architecture

The both 1st and 2nd generation of DeepID models are almost same as seen. The 1st generation expect 39×31 sized 1 channel input whereas 2nd generation expects 55×47 sized 3 channel (RGB) input images. The 2nd generation is named DeepID2 as well.

There are 4 convolution layers and one fully connected layer in DeepID models. Researchers trained the model as a regular classification task to classify n identities initially. Then, they removed the final classification softmax layer when training is over and they use an early fully connected layer to represent inputs as 160 dimensional vectors. In this way, the model can represent faces it haven’t seen before.

Below is the diagram of model structure of DeepID1 and DeepID2.
![deepid-model-structures](https://user-images.githubusercontent.com/50628520/88133078-f161b480-cc00-11ea-9e75-3e9a6db3b59d.png)

As a state-of-the-art design 3rd convolution layer is connected to the both 4th convolution layer and fully connected layer whereas 4th convolution layer is connected to fully connected layer as well. Fully connected layer adds the receiving signal from 3rd and 4th convolution layers in DeepID2 whereas 1st generation DeepID appends receiving signals from those layers.

![Link to the DeepID paper](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)

I am going to implement DeepID2 to in Tensorflow.
