import cv2
vidcap = cv2.VideoCapture('3dpf_10X.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames_10X/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1