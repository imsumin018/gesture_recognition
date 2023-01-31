import cv2

cap = cv2.VideoCapture('TRAIN_300.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
