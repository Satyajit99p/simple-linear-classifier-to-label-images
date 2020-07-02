import numpy as np
import cv2

labels = ["dogs", "cats", "pandas"]
np.random.seed(1)

W = np.random.randn(3, 3072)
B = np.random.randn(3)

orig = cv2.imread("C:/Users/SATYAJIT/Desktop/animals/dogs/dogs_00033.jpg")
image = cv2.resize(orig, (32,32)).flatten()

scores = W.dot(image) + B

for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

cv2.putText(orig, "labels : {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
            2)

cv2.imshow("image", orig)
cv2.waitKey(0)
