import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2
import glob as gb

class AnnModel :
    s = 100
    output = 10
    DATADIR = "D://Ai//face recognition//dataset//training and test"
    code = {'Adriana Lima': 0, 'Ben Affleck': 1, 'Bill Gates': 2, 'Cristiano Ronaldo': 3, 'Emilia Clarke': 4,
            'Emma Watson': 5,
            'Lionel Messi': 6, 'Mark Zuckerberg': 7, 'Shakira Isabel Mebarak': 8, 'Tom Holland': 9}
    X = []
    y = []
    def __init__(self):
        for folder in os.listdir(self.DATADIR):
            files = gb.glob(pathname=str(self.DATADIR + '//' + folder + '/*.jpg'))
            for file in files:
                image = cv2.imread(file)
                image_array = cv2.resize(image, (self.s, self.s))
                self.X.append(list(image_array))
                self.y.append(self.code[folder])
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        X_train = X_train / 255.
        X_test = X_test / 255.
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(units=32, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(units=self.output, activation=tf.nn.softmax))
        self.model_compile()
        self.model_train(X_train, y_train)
        self.model.save('annimagee.model')

    def getcode(self, n):
        for x, y in self.code.items():
            if n == y:
                return x

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def model_train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=100)

    def train_new(self, file_path,name):
        x = []
        y = []
        self.code[name] = self.output
        self.output = self.output+1
        for folder in os.listdir(file_path):
            files = gb.glob(pathname=str(self.file_path + '/*.jpg'))
            for file in files:
                image = cv2.imread(file)
                image_array = cv2.resize(image, (self.s, self.s))
                x.append(list(image_array))
                y.append(self.code[folder])
        X_train = np.asarray(x)
        y_train = np.asarray(y)
        X_train = X_train / 255.
        self.model_train(X_train,y_train)

    def predectation(self,img_path):
        imge = []
        img_pred = cv2.imread(img_path)
        image_array = cv2.resize(img_pred, (self.s, self.s))
        imge.append(list(image_array))
        imge = np.array([imge])
        imge = imge / 255.
        image_pred = self.model.predict(imge)
        classes_x = np.argmax(image_pred, axis=1)
        class_pred = self.getcode(classes_x)
        return class_pred

obj1 = AnnModel()
img_path = "D:\\university\\LEVEL 3\\semester 1\\Ai project\\face recognition\\dataset\\predict\\Bill Gates5_583.jpg"
pred = obj1.predectation(img_path)
print(pred)
