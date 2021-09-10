import cv2

def get_contours(im):
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    ret, bw_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        #roi = im[y:y + h, x:x + w]
        #cv2.imwrite(str(idx) + '.jpg', roi)
        cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
    cv2.imshow('img', im)
    cv2.waitKey(0)
    return im