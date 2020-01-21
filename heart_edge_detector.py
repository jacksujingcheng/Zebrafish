import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

cap = cv2.VideoCapture('3dpf_40X.avi')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output9.mp4', fourcc, 4.0, (int(cap.get(3)), int(cap.get(4))))
while True:
    _, frame_raw = cap.read()
    height, width, layers = frame_raw.shape
    new_h = height//4
    new_w = width//4
    frame = cv2.resize(frame_raw, (new_w, new_h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    #    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #    frame = cv2.filter2D(frame, -1, kernel)

    #    sigma_est = np.mean(estimate_sigma(frame, multichannel=True))
    #
    #
    #    patch_kw = dict(patch_size=5,      # 5x5 patches
    #                patch_distance=6,  # 13x13 search area
    #                multichannel=True)
    #
    #    frame = cv2.medianBlur(frame,11)
    #    denoise2 = denoise_nl_means(frame, h=0.8 * sigma_est, sigma=sigma_est,
    #                            fast_mode=False, **patch_kw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frame = cv2.erode(frame, kernel, iterations=7)
    frame = cv2.dilate(frame, kernel, iterations=7)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    #    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #    frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    canny = cv2.Canny(frame, 50, 100)
    cv2.imshow("Frame", frame)
    cv2.imshow("Laplacian", laplacian)
    #    cv2.imshow("dst", dst)
    cv2.imshow("Canny", canny)
    video.write(canny)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()