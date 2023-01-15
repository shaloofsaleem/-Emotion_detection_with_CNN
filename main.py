from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import keras
import os

IMG_HEIGHT=48
IMG_WIEGHT= 48

batch_size = 32



train_data_dir='Data/train'
validation_data_dir= 'Data/test'

trin_dataGen = ImageDataGenerator(rescale=1./225,
                                  rotation_range=30,
                                  shear_range=0.3,
                                  zoom_range=0.3,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

validation_dataGen = ImageDataGenerator(rescale=1./255)


train_generator = trin_dataGen.flow_from_directory(
                                            train_data_dir,
                                            color_mode='grayscale',
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)

validation_generator = validation_dataGen.flow_from_directory(
                                            validation_data_dir,
                                            color_mode='grayscale',
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)
    
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral','sad','surprise']
img, lable = train_generator.__next__()

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer= keras.optimizers.Adam(), 
#               loss= keras.losses.SparseCategoricalCrossentropy(from_logits= True), 
#               metrics= ['accuracy'])
print(model.summary())


train_path='Data/train'
test_path= 'Data/test'

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)   
    
print(num_test_imgs)
print(num_train_imgs)
epochs=100

history= model.fit(train_generator,
                   steps_per_epoch =num_train_imgs//batch_size,
                   epochs = epochs,
                   validation_data= validation_generator,
                   validation_steps= num_test_imgs//batch_size)
 
model.save('/content/drive/MyDrive/Facial Emotions Recognition/model_file_30epoches.h5')   
    
