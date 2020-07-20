## DeepFace
In modern face recognition, the conventional pipeline consists of four stages: detect ⇒ align ⇒ represent ⇒ classify
Facebook researchers announced its face recognition model DeepFace. It shows a very close performance to human level. Humans have 97.53% score whereas DeepFace model has 97.35% ± 0.25%. This means that the model can get higher score than human beings sometimes

### Architecture
DeepFace model is a 8 layered convolutional neural networks. Each layer is named with a letter and number as seen. Here, number refers to the index from 1 to 8 and letter states the type of layer. C refers to convolutional layer, M refers to max pooling, L refers to locally connected layer and F refers to fully connected layer.

![deepface-model](https://user-images.githubusercontent.com/50628520/87941595-19470000-cabb-11ea-88cb-9ba805451a24.png)

