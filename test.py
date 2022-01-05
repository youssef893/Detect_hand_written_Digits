import cv2
import numpy as np
from keras.models import load_model

model = load_model('cnn.h5')


def predict_digit(img):

    # resize image to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    # predicting the class
    res = model.predict([img])
    return np.argmax(res, axis=1), np.max(res)


def captch_ex(file_name):
    img = cv2.imread(file_name)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    ret, mask = cv2.threshold(img_gray_blur, 70, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(mask, kernel, iterations=10)  # dilate , more the iteration more the dilation

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours
    
    l = []
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        num = mask[y:y+h, x:x+w]

        text, prob = predict_digit(num)
        print(f'text is {text[0]}, acc={prob * 100}%')
        l.append(prob * 100)
        cv2.putText(img, f'text is {text[0]}', (x + w - 90, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    mean = np.mean(l)
    cv2.putText(img, f'acc={mean}%', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    # write original image with added contours to disk
    cv2.imwrite('teset.png', img)
    cv2.imshow('captcha_result', img)
    cv2.waitKey()


file_name = 'Screenshot (13).png'
captch_ex(file_name)
