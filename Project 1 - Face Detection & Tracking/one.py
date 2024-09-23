import cv2

alg = "C:/Users/sen_s/Downloads/haarcascade_frontalcatface.xml"
haar = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)
while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

