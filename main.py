import fingertrackingmodule as ftm
import cv2
import os
import numpy as np

headers_List = []
FOLDER_HEADERS = 'Headers'
BRUSH_THICKNESS = 25
ERASE_THICKNESS = 100
draw = False
erase = False
draw_color = (0, 0, 0)


WIDTH = 1920
HEIGHT = 1080
imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
xp, yp = 0, 0
p1, p2 = 4, 8 # номера большего и указательных пальцев

myList = os.listdir(FOLDER_HEADERS)
for imgPath in myList:
    image = cv2.imread(FOLDER_HEADERS + '/' + imgPath)
    headers_List.append(image)
header = headers_List[-1]


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
detector = ftm.handDetector()
while cap.isOpened():  # пока камера "работает"
    success, image = cap.read()  # получение кадра с камеры
    if not success:  # если не удалось получить кадр
        print('Не удалось получить кадр с web-камеры')
        continue  # возвращаемся к ближайшему циклу
    image = cv2.flip(image, 1)  # зеркально отражаем изображение
    detector.findHands(image)
    detector.findFingersPositon(image)
    h, w, c = header.shape
    mhl = detector.result.multi_hand_landmarks
    if mhl:
        handCount = len(mhl)
        for i in range(handCount):
            x1, y1 = detector.pointPosition[i][p1][0], detector.pointPosition[i][p1][1]
            x2, y2 = detector.pointPosition[i][p2][0], detector.pointPosition[i][p2][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            distance = detector.findDistance(p1, p2, i)
            if distance < 50:
                if cy <= h:
                    if 300 <= cx <= 595:
                        header = headers_List[0]
                        draw = True
                        erase = False
                        draw_color = (0, 0, 255)
                    elif 709 <= cx <= 1006:
                        header = headers_List[1]
                        draw = True
                        erase = False
                        draw_color = (255, 0, 0)
                    elif 1103 <= cx <= 1409:
                        header = headers_List[2]
                        draw = True
                        erase = False
                        draw_color = (0, 255, 255)
                    elif 1525 <= cx <= 1652:
                        header = headers_List[3]
                        draw = False
                        erase = True
                        draw_color = (0, 0, 0)
                    elif 1765 <= cx <= 1884:
                        header = headers_List[4]
                        draw = False
                        erase = False
                        draw_color = (0, 0, 0)

                else:
                    if draw:
                        if xp == 0 and yp == 0:
                            xp, yp = cx, cy
                        cv2.line(image, (cx, cy), (xp, yp), draw_color, BRUSH_THICKNESS)
                        cv2.line(imgCanvas, (cx, cy), (xp, yp), draw_color, BRUSH_THICKNESS)
                    if erase:
                        if xp == 0 and yp == 0:
                            xp, yp = cx, cy
                        cv2.line(image, (cx, cy), (xp, yp), draw_color, ERASE_THICKNESS)
                        cv2.line(imgCanvas, (cx, cy), (xp, yp), draw_color, ERASE_THICKNESS)

                cv2.circle(image, (cx, cy), 15, draw_color, cv2.FILLED)
                xp, yp = cx, cy

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, imgCanvas)

    image[0:h, 0:w] = header
    cv2.imshow('window', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Ожидаем нажатие ESC
        break