# %% 5. Creación de un modelo de red neuronal adecuado para el problema a analizar
import numpy as np
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL
from keras.utils.np_utils import to_categorical


import imutils

os.chdir("C:/deep_learning/proyecto_retino_2")

def RecorteContorno(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)


    
    # Umbralizamos la imagen (erosiones/dilataciones) para eliminar regiones
    # de ruido 
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh , None , iterations = 2)
    thresh = cv2.dilate(thresh , None , iterations = 2)
    

    # Busqueda de los contornos 
    cnts = cv2.findContours(thresh.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    
    # Usamos imutils 
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)

    # Obtener los puntos/pixeles extremos 
    extLeft = tuple(c[c[:,:,0].argmin()][0]) 
    extRight = tuple(c[c[:,:,0].argmax()][0]) 
    extTop = tuple(c[c[:,:,1].argmin()][0]) 
    extBot = tuple(c[c[:,:,1].argmax()][0]) 
    
    NuevoCerebro = image[ extTop[1]: extBot[1], extLeft[0]: extRight[0]]
    return NuevoCerebro

# Probemos nuestra funcion RecorteContorno

''' 
Prueba1 = cv2.imread("439_left.jpeg")
plt.imshow( Prueba1)


Prueba1Transform = RecorteContorno(Prueba1)

plt.subplot(1,2,1)
plt.imshow(Prueba1)
plt.subplot(1,2,2)
plt.imshow(Prueba1Transform)

'''

# %% Data Augmatation
# Realiza operaciones del algebra lineal sobre las imagenes
from keras.preprocessing.image import ImageDataGenerator

# help(ImageDataGenerator)

def CrecimientoData(file_dir,
                    n_generated_samples,
                    save_to_dir,cant_img):
    """
    

    Parameters
    ----------
    file_dir : str
        Directorio donde se encuentran las imagenes a realizar el 
        data augmentation.
    n_generated_samples : int
        Numero de muestras generadas usando  la imagen data.
    save_to_dir : str
        Directorio donde se almacenan las imagenes aumentadas

    Returns
    -------
    None.

    """

    # Configuremos nuestro generador de datos
    data_gen = ImageDataGenerator(rescale=1./255,
                                  #rotation_range= 10,
                                  #(mover horiz)width_shift_range=0.1,
                                  #(mover vert)height_shift_range=0.1,
                                  #(cortar la imagen)shear_range=0.1,
                                  #(brillo aleatorio)brightness_range = (0.3,1.0),
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  #(espejo vertical)vertical_flip=True,
                                  #fill_mode = "nearest"
                                  )
    
    # Necesito que cada una de mis imagenes pase por ese 
    # generador 
    #for filename in os.listdir(file_dir):
    cant = 0
    for filename in os.listdir(file_dir):
        # Cargamos la imagen 
        image = cv2.imread(file_dir + '/' + filename)
        
        # Redimensionamiento de imagenes para el batch 
        image = image.reshape((1,) + image.shape)
        
        
        # Crear un nuevo  nombre en la salida para cada imagen 
        ImgGuardada = 'aug_' + filename[:-4]
        
        # Flujo de procesamiento 
        i = 0
        for batch in data_gen.flow(x  = image,
                                   batch_size = 4,
                                   save_to_dir=save_to_dir,
                                   save_prefix= ImgGuardada,
                                   save_format= 'jpg'):
            i = i+1
            if i> n_generated_samples:
                break

        cant = cant+1
        if cant> cant_img:
            break
#  PRobemos la funcion CrecimientoData

# Directorio para almacenar las imagenes procesadas con el 
# data augmentation

augmented_data_path = os.getcwd()  + "/retino_dataset_f1"+ "/retino" + "/augmented_data"


# Medimo el tiempo para el data augmentation 

import time 
inicio = time.time()

#  Realizamos el data augmentation : yes
CrecimientoData(file_dir = os.getcwd() + "/retino_dataset_f1"  + "/archive" +"/0"
                , n_generated_samples =  2
                , save_to_dir = augmented_data_path + "/0",cant_img=250)


#  Realizamos el data augmentation : no
CrecimientoData(file_dir = os.getcwd() + "/retino_dataset_f1"  + "/archive" +"/1"
                , n_generated_samples =  2
                , save_to_dir = augmented_data_path + "/1",cant_img=250)

#  Realizamos el data augmentation : no
CrecimientoData(file_dir = os.getcwd() + "/retino_dataset_f1"  + "/archive" +"/2"
                , n_generated_samples =  2
                , save_to_dir = augmented_data_path + "/2",cant_img=250)

#  Realizamos el data augmentation : no
CrecimientoData(file_dir = os.getcwd() + "/retino_dataset_f1"  + "/archive" +"/3"
                , n_generated_samples =  2
                , save_to_dir = augmented_data_path + "/3",cant_img=250)
#  Realizamos el data augmentation : no
CrecimientoData(file_dir = os.getcwd() + "/retino_dataset_f1"  + "/archive" +"/4"
                , n_generated_samples =  2
                , save_to_dir = augmented_data_path + "/4",cant_img=250)

fin = time.time()

print("Tiempo de procesamiento [Data Augmentation]", fin-inicio)
# Tiempo de procesamiento [Data Augmentation] 64.03230118751526

# %% Cargar los datos aumentados (crear objetos de tipo ndarray)
import numpy as np 
def jpg2Array(dir_list, image_size):
    
    """
    dir_list : lista de directorios : yes / no
    image_size = dimension de la imagen que va alimentar a mi red neuronal 
    
    """
    
    # Almacenar las variables independientes : X
    # Almacenar la variables dependiente : y
    
    # Creamos listas en blanco para almacenar las imagenes (ndarray)
    dataX = []
    datay = []
    
    # A partir de image_size debo obtener ancho y alto de las images que 
    # van a alimentar a mi red neuronal 
    image_width , image_height = image_size 
    
    # Usamos una estructura repetitiva para barrer los directorio 0/1/2/3/4
    for directory in dir_list:

        i=0
        for filename in os.listdir(directory):
            # Cargamos la imagen 
            
            #if (tipo_data == 2 & i == cant_img) :
             #   break

            image = cv2.imread(directory + "/" + filename)
                
                # En este punto se necesita que la imagen almacena solo la informacion
                # de interes para alimentar el fit de mi red neuronal 
            image = RecorteContorno(image)
                # Redimensionamos deacuerdo a lo especificado en el argumento image_size
            image = cv2.resize(image , dsize = (image_width, image_height),
                                interpolation = cv2.INTER_CUBIC)
             
                # Normalizamos los pixeles 
                #image  = image / 255
                
                # Almacenamos la imagen procesada
            imgNP_Red = np.array(image)[:,:,0] 
            imgNP_Green = np.array(image)[:,:,1]
            imgNP_Blue = np.array(image)[:,:,2]
            imgNP_Color = np.concatenate((imgNP_Red, imgNP_Green), axis=0)
            imgNP_Color = np.concatenate((imgNP_Color, imgNP_Blue), axis=0)

            dataX.append(imgNP_Color)

            ''' 
            if directory[-1:] == '0':
               datay.append([0])
            '''
            if directory[-1:] == '1':
                datay.append([0])
             
            if directory[-1:] == '2':
                datay.append([1])
            
            if directory[-1:] == '3':
                datay.append([2])
            if directory[-1:] == '4':
                datay.append([3])
            
            i=i+1
            
    
    # Transformamos las listas X e y a objetos de tipo ndrray
    X = np.array(dataX)

    X.shape
   
    y = np.array(datay)
    y.shape

   

    return X,y
    

# Probemos jpg2Array
# 
# Directorio donde estan las imagenes aumentadas 
augmented_data = os.getcwd()  + "/retino_dataset_f1"+ "/retino" + "/augmented_data"


# Creamos los nombres de directorios (imagenes aumentadas) para yes/no
augmented_0 = augmented_data + "/0"
augmented_1 = augmented_data + "/1"
augmented_2 = augmented_data + "/2"
augmented_3 = augmented_data + "/3"
augmented_4 = augmented_data + "/4"


# Definimos la dimension de las imagenes resultado 
IMG_WIDTH, IMG_HEIGHT = (350,350)

#IMG_WIDTH, IMG_HEIGHT = (512,512)

varX, vary = jpg2Array([augmented_1,augmented_2,augmented_3,augmented_4] , 
                       (IMG_WIDTH, IMG_HEIGHT))

                     



vary
# %% Particionamiento del conjunto de datos 
from sklearn.model_selection import train_test_split


train_X, test_X, train_y, test_y = model_selection.train_test_split(varX,vary,test_size=0.3)

# Visualicemos las imagenes
fig, ax =  plt.subplots(1,3, figsize = (10,10))
for i, axi in enumerate(ax.flat):
    axi.imshow(train_X[i])
    axi.set(xticks=[], yticks=[])
plt.show()

train_X  = train_X.reshape((train_X.shape[0], 350,350,3))
test_X  = test_X.reshape((test_X.shape[0], 350,350,3))
'''
train_X  = train_X.reshape((train_X.shape[0], 224,224,3))
test_X  = test_X.reshape((test_X.shape[0], 224,224,3))
train_X[0]
'''
# Convertimos a punto flotante: a las variables independientes
train_X = train_X.astype("float32")
test_X = test_X.astype("float32")

train_X.shape
#Normalizamos
train_X = train_X/255
test_X= test_X/255

train_y

train_y = to_categorical(train_y, 4)
test_y = to_categorical(test_y, 4)





# %% Definicion del modelo
# Tiene dos aspecots fundamentales:
    # FRONT-END: Extracción de características (capas conv. y capas pooling)
    # BACK-END: Clasificador

# Capa convulucional: filtro de (3,3) y 32 filtros y un capa pooling(max_pooling)
# Necesitamos aplanar los objetos bidimensionales (resultado de la conv. y del pooling)

# En vista que tenemos un problema de clasificación (2 clases)
# ==> necesitamos una capa de salida con 2 nodos a predecir y una función de activacion sigmoid


from keras.models import Sequential
from keras.layers import Conv2D,Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam


EPOCHS = 20
INIT_LR = 1e-3
NUM_CLASSES = 4
# we need images of same size so we convert them into the size
WIDTH = 350
HEIGHT = 350
DEPTH = 3
inputShape = (HEIGHT, WIDTH, DEPTH)

#from tensorflow.keras.applications import InceptionV3
def createModel():
    model = Sequential()
    # first set of CONV => RELU => MAX POOL layers
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # returns our fully constructed deep learning + Keras image classifier 
    
    # use binary_crossentropy if there are two classes
    return model

model = createModel()
model.summary()

# %% 3er Paso: Compilar el modelo

# Funcion de perdida : cross entropy (binary_crossentropy)
# Definir el optimizador : adam 
# metrica : accuracy

# Compilacion del modelo 

import tensorflow as tf

#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

help(tf.keras.optimizers.Adam)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
    loss = "categorical_crossentropy",
    optimizer =opt,
    metrics = 'accuracy'
)




# %% 4to Paso : Ajustemos el modelo 
# Este ajuste ocurre en una serie de epocas (epochs) y cada epoca se va dividir
# en lotes (batch_size)
# 
# epoch : una pasada a traves de todas las filas del conjunto de datos de train
# batch_size : UNa o mas muestras (filas) consideradas por el modelo dentro de una epoca
# antes de que se actualicen los pesos 

import numpy as np   

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

train_gen = DataGenerator(train_X, train_y, 150)

train_gen

inicio_CNN = time.time()
history = model.fit(train_gen,
                    epochs=2,
                    validation_data=test_gen)


inicio_CNN = time.time()
model.fit(train_gen, epochs= EPOCHS)
final_CNN = time.time()
print("Tiempo de Ajuste del modelo CNN:\t",final_CNN-inicio_CNN) 

model.save("50_CNN_350_350_3_nm.h5")

# %% 5to paso: Almacenamos el modelo y su arquitectura en un archivo 
dataX = []
Prueba1 = cv2.imread("51_left.jpeg")
plt.imshow( Prueba1)
                

image = RecorteContorno(Prueba1)
                
image = cv2.resize(image , dsize = (350, 350),
                                interpolation = cv2.INTER_CUBIC)

Prueba2 = np.array(image)

type(Prueba2)

dataX.append(Prueba2)

type(dataX)

train_X=np.array(dataX)
train_X.shape

train_X  = train_X.reshape((train_X.shape[0], 299,299,3))


train_X = train_X.astype("float32")


#Normalizamos
train_X = train_X/255



predicciones = model.predict(train_X)

print(predicciones)

# Displaying the image 
#plt.imshow( Prueba1)


#%%


from keras.models import load_model

#probando modelos con 150 - 125 -175 de batch size

#31_CNN_320_320_3.h5  /  150 batch
#32_CNN_320_320_3.h5  /  125 batch
#33_CNN_320_320_3.h5  /  175 batch

#model = load_model('33_CNN_320_320_3.h5')

model = load_model('full_retina_model.h5')




_, precision = model.evaluate(train_X,train_y)

print("Precion del modelo train: %.2f" %(precision*100)) # 100.00

_, precision = model.evaluate(test_X,test_y)

print("Precion del modelo test: %.2f" %(precision*100)) # 100.00


