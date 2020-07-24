## VGGFace(Deep Face Recognition)

Oxford visual geometry group announced its deep face recognition architecture. We have been familiar with VGG in imagenet challenge. We can recognize hundreds of images just applying transfer learning. Also, we have used same model for style transfer.
Face recognition either from a single photograph or from a set of faces tracked in video. It Focuses on face recognition in images and videos.

## Architecture:

Even though research paper is named Deep Face, researchers give VGG-Face name to the model. This might be because Facebook researchers also called their face recognition system DeepFace – without blank. VGG-Face is deeper than Facebook’s Deep Face, it has 22 layers and 37 deep units.

The structure of the VGG-Face model is demonstrated below. Only output layer is different than the imagenet version – you might compare.

This is VGGFace model.
![vgg-face-model](https://user-images.githubusercontent.com/50628520/88368108-14c96280-cdad-11ea-8b96-f93d0a892a64.png)

The layer stucture is given below in the diagram.
![layer-details-in-vgg-face](https://user-images.githubusercontent.com/50628520/88368196-43dfd400-cdad-11ea-99ba-206836dd6e46.png)

![vgg-face-architecture (1)](https://user-images.githubusercontent.com/50628520/88368248-5d811b80-cdad-11ea-8981-761325cd3ecb.jpg)

### 1) Learning a face classifier:

It recognized N = 2,622 unique individuals, setup as a N-ways classification problem. A final FC layer contained N linear predictor one per identity. These scores are compared to ground turh class identity c={1, ..., N} by using empirical softmax log loss.

After learning, the classifier layer (w, b) can be removed and the score vectors can be used for face verification using the Euclidean distance to compare them. However, the score can be significantly improved by tuning them for verification in Euclidean space using triplet loss training process.

### 2) Learning face embedding using a triplet loss:

Triplet loss training aims at learning score vectors that perform well in the final application, i.e identity verification by comparing face descriptors in Euclidean space(metric learning).

A triplet (a, p, n) contains an anchor face image 'a' as well as a positive 'p' and negative 'n'.

![triplet_loss](https://user-images.githubusercontent.com/50628520/88369268-5ce98480-cdaf-11ea-9f46-21cbdce3b978.png)

### 3) Training:

For learning the embedding using triplet loss, the network is frozen except the last FC layer implementing the discriminative projection.

### 4) Predictions:

#### 1) VGG16 backbone:

![Screenshot from 2020-07-24 18-22-51](https://user-images.githubusercontent.com/50628520/88392502-d303e080-cddb-11ea-8a40-d5dacb4b3289.png)

![Screenshot from 2020-07-24 18-22-28](https://user-images.githubusercontent.com/50628520/88392531-df883900-cddb-11ea-83da-a0124c76c90f.png)

![Screenshot from 2020-07-24 18-22-16](https://user-images.githubusercontent.com/50628520/88392552-eb73fb00-cddb-11ea-85b5-2906dadf0159.png)

![Screenshot from 2020-07-24 18-22-04](https://user-images.githubusercontent.com/50628520/88392596-f9298080-cddb-11ea-92ae-8019988238e9.png)
