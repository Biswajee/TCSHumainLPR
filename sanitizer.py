#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
processing() : accepts list of images and applies various image processing techniques
 + Maximizing contrast
 + Adaptive thresholding
 + Contouring 

 returns a list of sanitized images
'''
Sanitized_list = []     # list of sanitaized images
def processing(plate_list):
    '''
    # Code for debugging : test effects on a test image `test_img.jpg`
    # Read the image
    img_ori = cv2.imread('test_img.jpg')
    RGB_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    height, width, channel = RGB_img.shape
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)    # grayscaled image
    '''
    # height=1080       # $$$$$$$$$$$$$$$$$$$$$$$$$
    # width=1920        # $ TEST DATA | STAY AWAY $
    # channel=3         # $$$$$$$$$$$$$$$$$$$$$$$$$
    
    print('processing images...')
    for img in plate_list:
        try:
            height, width, channel = img.shape
            print('Original Image Shape',img.shape)
            
            # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print('Grayscale Image Shape',img.shape)

            # Maximize Contrast
            structElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            imgTopHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structElement)
            imgBlackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, structElement)
            imgGrayscalePlusTopHat = cv2.add(img, imgTopHat)
            img = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

            # Adaptive Thresholding
            img_blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
            img_thresh = cv2.adaptiveThreshold(
                img_blurred, 
                maxValue=255.0, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=19, 
                C=9
            )

            # Find Contours
            _, contours, _ = cv2.findContours(
                img_thresh, 
                mode=cv2.RETR_LIST, 
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            temp_result = np.zeros((height, width, channel), dtype=np.uint8)
            cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

            # Prepare Data
            temp_result = np.zeros((height, width, channel), dtype=np.uint8)
            contours_dict = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=1)
                # insert to dict
                contours_dict.append({
                    'contour': contour,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'cx': x + (w / 2),
                    'cy': y + (h / 2)
                })


            # Eliminating contours with irrelevant sizes 
            MIN_AREA = 80
            MIN_WIDTH, MIN_HEIGHT = 2, 8
            MIN_RATIO, MAX_RATIO = 0.25, 1.0

            possible_contours = []

            cnt = 0
            for d in contours_dict:
                area = d['w'] * d['h']
                ratio = d['w'] / d['h']
                
                if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
                    d['idx'] = cnt
                    cnt += 1
                    possible_contours.append(d)
                    
            # visualize possible contours
            temp_result = np.zeros((height, width, channel), dtype=np.uint8)

            for d in possible_contours:
            #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)



            Sanitized_list.append(temp_result)
        except:
            pass
    return Sanitized_list
        




'''
plot() : accepts an input image and displays plot on screen.
[pauses subsquent code executions] 
'''
def plot(segment_list, img_list):
    i = 0
    for img, ori in zip(segment_list, img_list):
        _ , ax = plt.subplots(1, 2) # horizontally stacked subplots
        plt.figure()
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(ori, cmap='gray')
        i=i+1
    plt.show()