from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D
from tensorflow.keras.models import Model

def deepface(input_shape=None):
    input_data = Input(shape=input_shape)
    
    conv1 = Conv2D(32, (11, 11), activation='relu', name='C1')(input_data)
    maxpool1 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2')(conv1)
    conv2 = Conv2D(16, (9, 9), activation='relu', name='C3')(maxpool1)
    lconv1 = LocallyConnected2D(16, (9, 9), activation='relu', name='L4')(conv2)
    lconv2 = LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5')(lconv1)
    lconv3 = LocallyConnected2D(16, (5, 5), activation='relu', name='L6')(lconv2)
    flat = Flatten(name='F0')(lconv3)
    fc1 = Dense(4096, activation='relu', name='F7')(flat)
    drop = Dropout(rate=0.5, name='D0')(fc1)
    fc2 = Dense(8631, activation='softmax', name='F8')(drop)
    
    model = Model(inputs=input_data, outputs=fc2, name="DeepFace")
    
    return model

def build_model():
    model = deepface(input_shape=(152, 152, 3))
    model.load_weights('../weights/VGGFace2_DeepFace_weights_val-0.9034.h5')

    new = Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)

    return new