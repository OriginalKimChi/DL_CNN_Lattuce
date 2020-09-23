# DL_CNN_Lattuce

In this project, we are creating a lettuce pest diagnostic model.

It is a model that derives the results for a new lettuce image through learning the image of normal lettuce and lettuce with pests.

## Code

### 1. Load the required package

<pre>
<code>
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
</code>
</pre>

먼저 데이터 학습에 필요한 패키지를 불러옵니다.

### 2. Creating a dataset

<pre>
<code>
train_dataGenerator = test_dataGenerator= ImageDataGenerator(rescale=1./255)

train_generator = train_dataGenerator.flow_from_directory(
    '/gdrive/My Drive/Lattuce/train', 
    target_size=(24,24),
    batch_size=3,
    class_mode='binary')

test_generator = test_dataGenerator.flow_from_directory(
    '/gdrive/My Drive/Lattuce/test', 
    target_size=(24,24),
    batch_size=3,
    class_mode='binary')
</code>
</pre>

<img width="382" alt="img1" src="https://user-images.githubusercontent.com/48902646/94031959-6b2d3e80-fdfa-11ea-9a1e-3c9173fae2b3.png">

다음으로 Keras의 ImageDataGenerator를 통해서 객체를 생성한 뒤, flow_from_directory() 함수를 통해 generator를 생성합니다. 제너레이터는 train과 test용으로 두가지를 생성합니다.

flow_from_directory()의 인자는 다음과 같습니다.
-   첫번째 인자 : 이미지 경로
-   target_size : 패치 이미지 크기를 지정
-   batch_size : 배치 크기를 지정
-   class_mode : 분류 방식에 대한 지정('binary' : 1D 이진 라벨 반환)

### 3. Model configuration

<pre>
<code>
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
</code>
</pre>

다음으로 Convolution 신경망 모델을 구성하겠습니다.
아래과 같은 레이어로 모델을 구성합니다.
-   컨볼루션 레이어 : 필터 수 32개, 필터 크기 3X3, 활성화 함수 'relu', 입력 이미지 24X24, 이미지 채널 3개
-   컨볼루션 레이어 : 필터 수 64개, 필터 크기 3X3, 활성화 함수 'relu'
-   맥스풀링 레이어 : 풀 크기 2 x 2
-   댄스 레이어  : 출력 뉴런 수 128개, 활성화 함수 'relu'
-   댄스 레이어  : 출력 뉴런 수 1개, 활성화 함수 'sigmoid'

구성 모델은 다음과 같이 표현됩니다.

<img width="543" alt="스크린샷 2020-09-24 오전 12 42 16" src="https://user-images.githubusercontent.com/48902646/94036034-d11bc500-fdfe-11ea-96b6-54304a08013b.png">

<pre>
<code>
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
</code>
</pre>



### 4. Setting up the learning process

<pre>
<code>
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
</code>
</pre>

모델을 정의 한 후, loss 함수와 optimizer 알고리즘으로 컴파일을 합니다.
-   loss : 손실 함수는 단일 클래스 이므로 'binary_crossentropy'를 선택합니다.
-   optimizer : 최적화 알고리즘은 경사 하강법 알고리즘인 'adam'을 선택합니다.
-   metrics : 평가 척도를 나타내며, 분류를 하기 위한 모델이므로 'accuracy'를 선택합니다.

### 5. Model training

<pre>
<code>
hist = model.fit_generator(
        train_generator,
        steps_per_epoch=15,
        epochs=50,
        validation_data=test_generator,
        validation_steps=5)
</code>
</pre>

모델 학습은 generator로 생성한 배치 학습 인 fit_generator()라는 함수를 이용합니다.
-   train_generator : 훈련 데이터 셋을 앞서 생성한 'train_generator'로 설정합니다.
-   steps_per_epoch : 한 epoch에 15번의 스텝을 설정합니다.
-   epochs : 전체 훈련 데이터셋 반복 학습을 총 50번으로 설정합니다.
-   validation_data : 검증 데이터 셋을 앞서 생성한 'test_generator'로 설정합니다.
-   validation_steps : 한 epoch 마다 검증되는 스텝 수를 5로 설정합니다.

<pre>
<code>
15/15 [==============================] - 14s 921ms/step - loss: 0.7696 - acc: 0.4889 - val_loss: 0.6714 - val_acc: 0.5333
Epoch 2/50
15/15 [==============================] - 15s 988ms/step - loss: 0.6921 - acc: 0.4000 - val_loss: 0.6713 - val_acc: 0.5333
Epoch 3/50
15/15 [==============================] - 8s 536ms/step - loss: 0.6092 - acc: 0.5333 - val_loss: 0.5946 - val_acc: 0.6667
Epoch 4/50
15/15 [==============================] - 1s 88ms/step - loss: 0.4969 - acc: 0.8444 - val_loss: 0.4028 - val_acc: 0.8571
Epoch 5/50
15/15 [==============================] - 1s 64ms/step - loss: 0.4098 - acc: 0.8000 - val_loss: 0.7744 - val_acc: 0.5333
...
Epoch 46/50
15/15 [==============================] - 1s 87ms/step - loss: 7.3105e-04 - acc: 1.0000 - val_loss: 1.5714 - val_acc: 0.7333
Epoch 47/50
15/15 [==============================] - 1s 77ms/step - loss: 6.4181e-04 - acc: 1.0000 - val_loss: 1.7289 - val_acc: 0.6667
Epoch 48/50
15/15 [==============================] - 1s 96ms/step - loss: 3.5480e-04 - acc: 1.0000 - val_loss: 0.6887 - val_acc: 0.8571
Epoch 49/50
15/15 [==============================] - 2s 108ms/step - loss: 4.9159e-04 - acc: 1.0000 - val_loss: 1.5783 - val_acc: 0.7333
Epoch 50/50
15/15 [==============================] - 1s 80ms/step - loss: 3.9906e-04 - acc: 1.0000 - val_loss: 0.8824 - val_acc: 0.8667
</code>
</pre>

### 5. Model evaluation

<pre>
<code>
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
</code>
</pre>

모델애 대한 평가를 evaluate_generator()를 통해 진행합니다.

<pre>
<code>
-- Evaluate --
acc: 73.33%
</code>
</pre>

### 5. Use Model

<pre>
<code>
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
</code>
</pre>

제너레이터에서 제공되는 샘플을 사용하기 위해 predict_generator() 함수를 사용합니다.
예측 결과는 각 클래스 별로 확률 벡터로 표시됩니다.

<pre>
<code>
-- Predict --
{'cercospora': 0, 'normal': 1}
[[0.983]
 [0.221]
 [0.000]
 [1.000]
 [1.000]
 [0.001]
 [1.000]
 [0.012]
 [0.815]
 [0.000]
 [0.001]
 [0.000]
 [0.002]
 [0.137]
 [0.383]]
</code>
</pre>

### 5. Practical use

<pre>
<code>
from keras.preprocessing import image
test_image = image.load_img('/gdrive/My Drive/Lattuce/predict2.jpg', target_size = (24,24))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0] >= 0.5:
  prediction = 'normal' 
else:
  prediction = 'cercospora'
print(prediction)
</code>
</pre>

생성한 모델을 샘플 이미지가 아닌 실제 사진을 통해 결과를 확인합니다. predict()를 활용하면 해당 input 값에 대한 모델의 검증결과를 확인 할 수 있습니다.

<pre>
<code>
cercospora
</code>
</pre>

<img width="240" alt="스크린샷 2020-09-24 오전 12 34 23" src="https://user-images.githubusercontent.com/48902646/94035151-bbf26680-fdfd-11ea-902e-d643a711e2a0.png">


