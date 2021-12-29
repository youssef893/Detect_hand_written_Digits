import cv2
import numpy as np

cap = cv2.VideoCapture(0)

from keras.models import load_model

model = load_model('cnn.h5')


def predict_digit(img):
    # resize image to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    # predicting the class
    res = model.predict([img])

    return np.argmax(res, axis=1), np.max(res)


while True:
    success, img = cap.read()
    if not success:
        break
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    ret, mask = cv2.threshold(img_gray_blur, 70, 255, cv2.THRESH_BINARY_INV)
    text, prob = predict_digit(mask)
    cv2.putText(img, f'{text}, {prob * 100}%', (200, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('image', img)

    if cv2.waitKey(1) == 13:
        break
