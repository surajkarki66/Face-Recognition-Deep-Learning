# Face-Recognition-Deep-Learning

## Deep learning approach:
It all begun in 2014, when Facebook introduced DeepFace algorithm.

### Evolution of face recoginition:
#### 1) DeepFace
#### 2) DeepID Series
#### 3) VGGFace
#### 4) FaceNet
#### 5) VGGFace2 , etc

![Screenshot from 2020-07-20 13-19-27](https://user-images.githubusercontent.com/50628520/87911911-1da7f480-ca8c-11ea-982e-87123415452a.png)


### FR Module:
FR can be categorized  as face verification and face identification. Face verification computes  one to one similarity between gallery and probe to determine whether the two images are of the same object. Face identification compute one to many similarity to determine the specific identitis of the probe face.

When the probe appears in the gallery identity, this is reffered  to as closed-set identification where as when th probes include those who are not in the gallery, this is open-set identification.

FR module consists of  face proccessing, deep feature extraction and face matching. Which is shown by below expression.

#### M[F(Pi(Ii)), F(Pj(Ij))]

Where Ii and Ij are two face images. P stands for face proccessing to handle intra-personal variations (poses, illuminations, expressions, etc). F denotes feature extraction, which encodes the identity information. M means a face matching algorithm used to compute similarity scores.

![Screenshot from 2020-07-20 13-20-05](https://user-images.githubusercontent.com/50628520/87911951-30222e00-ca8c-11ea-83ad-3facbbc3b158.png)



#### 1) Face Proccessing:
The face proccessing methods are categorized as "one to many" augmentation and "many to one" normalization.


##### a) One to many augmentation:
Generating many patches or images of pose variability from a single image to enable deep networks to learn pose-invariant representation.


##### b) Many to one normalization:
Recovering the canonical view of face images from one or many images of nonfrontal view; then, FR can be performed as if it were under controlled conditions.


![processing](https://user-images.githubusercontent.com/50628520/87911123-d2d9ad00-ca8a-11ea-8289-f71e70fe8f99.png)


#### 2) Deep Feature Extraction:
Using CNN architecture as backbone. Such as AlexNet, VGGNet, GoogleNet, ResNet and SENet.

![Screenshot from 2020-07-20 12-34-36](https://user-images.githubusercontent.com/50628520/87911450-601d0180-ca8b-11ea-8fd2-2be06d0bdd59.png)


##### Loss Function:
Many works focus on creating novel loss functions to make features not only more separable but also discriminative.

###### i) Euclidean Distance Based loss:
Compressing  intra-variance and enlarging inter-variance based on Euclidean distance.

###### ii) Angular/Cosine-margin based loss:
Learning discriminative face features in terms of angular similarity, leading to potentially larger angular/cosine separability between learned feature.

###### iii) Softmax loss and its variations:
Directly using softmax loss or modifying it to improve performance. eg. L2 normalization.

![Screenshot from 2020-07-20 13-20-59](https://user-images.githubusercontent.com/50628520/87911871-0a952480-ca8c-11ea-9f1b-c6b0ecf69c78.png)



#### 3) Face Matching By Deep Features:
After the deep networks are trained with massive data and an appropriate loss function, each of the test images is passed through the networks to obtain a deep feature representation.

Once the deep features are extracted most methods directly calculate the similarity between two features using cosine distance or L2 distance.

Then the nearest neighbors & threhold comparision are used for both identification and verification tasks.


All Steps is cleared from the below image.

![Screenshot from 2020-07-20 13-20-41](https://user-images.githubusercontent.com/50628520/87912004-4b8d3900-ca8c-11ea-9b41-79285a6277d6.png)

