import cv2 as cv

video = cv.VideoCapture('obama.mp4')
i = 0
while True:
    isTrue, frame = video.read()
    if isTrue == False:
        break
    resized_frame = cv.resize(frame,(256,256))
    cv.imwrite('Raghavendra/frames/'+str(i)+'.jpg', resized_frame)

    i += 1

video.release()
cv.destroyAllWindows()
