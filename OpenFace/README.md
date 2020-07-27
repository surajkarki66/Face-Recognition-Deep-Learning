## OpenFace (A general purpose face recognition library with mobile application)

OpenFace is a lightweight and minimalist model for face recognition. Similar to Facenet, its license is free and allowing commercial purposes.
It provides near human accuracy on the LFW benchmark and present a new classification benchmark for mobile scenarios.

It is used as general purpose library for face recognition. It is well suited for mobile scenarios.

## Architecture

OpenFace model expect (96x96) RGB images as input. It has a 128 dimensional output. The model is built on Inception ResNet V1.Even though the model seems complex, number of parameters are much less than VGG-Face.

![Copy-of-Racial-Bias-in-Facial-Recognition-Software-8-4](https://user-images.githubusercontent.com/50628520/88500983-dbca0180-cfe9-11ea-8d28-e218dfa20981.jpg)

### InceptionResNet V1

![Inception-resnet-v1-network-architecture-with-SPP](https://user-images.githubusercontent.com/50628520/88501257-a671e380-cfea-11ea-956f-23f85a48505b.png)
