import cv2


img_path = "D:/dataset/VOCdevkit/VOC2007/JPEGImages/001156.jpg"
im = cv2.imread(img_path)
cv2.imshow("input", im)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(im)
# ss.switchToSelectiveSearchFast()
ss.switchToSelectiveSearchQuality()
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

numShowRects = 2000
imOut = im.copy()

for i, rect in enumerate(rects):
    if (i < numShowRects):
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    else:
        break

cv2.imshow("output", imOut)
cv2.waitKey(0)