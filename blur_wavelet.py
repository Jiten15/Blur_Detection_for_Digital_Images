
import pywt
import cv2
import numpy as np
import os


import json

#
# def convertScale(img, alpha, beta):
#     new_img = img * alpha + beta
#     new_img[new_img < 0] = 0
#     new_img[new_img > 255] = 255
#     return new_img.astype(np.uint8)
#
#
# # Automatic brightness and contrast optimization with optional histogram clipping
# def automatic_brightness_and_contrast(image, clip_hist_percent=8):
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray=image
#     # Calculate grayscale histogram
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     hist_size = len(hist)
#
#     # Calculate cumulative distribution from the histogram
#     accumulator = []
#     accumulator.append(float(hist[0]))
#     for index in range(1, hist_size):
#         accumulator.append(accumulator[index - 1] + float(hist[index]))
#
#     # Locate points to clip
#     maximum = accumulator[-1]
#     clip_hist_percent *= (maximum / 100.0)
#     clip_hist_percent /= 2.0
#
#     # Locate left cut
#     minimum_gray = 0
#     while accumulator[minimum_gray] < clip_hist_percent:
#         minimum_gray += 1
#
#     # Locate right cut
#     maximum_gray = hist_size - 1
#     while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
#         maximum_gray -= 1
#
#     # Calculate alpha and beta values
#     alpha = 255 / (maximum_gray - minimum_gray)
#     beta = -minimum_gray * alpha
#
#     auto_result = convertScale(image, alpha=alpha, beta=beta)
#     return (auto_result, alpha, beta)


def blur_detect(img, threshold):


    
    # Convert image to grayscale
    # Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Y=img
    h, w = Y.shape

    image = cv2.resize(Y, (160, 160))
    image = cv2.resize(image, (h, w))

    
    
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold;
    EdgePoint2 = Emax2 > threshold;
    EdgePoint3 = Emax3 > threshold;
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges));

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent





if __name__ == '__main__':

    # MinZero = 0.001
    # threshold = 35

    MinZero=0.1
    threshold=10
    results = []

    imgpath="input/out/"
    count=0

    for img in os.listdir("input/out/"):
        count=count+1
        print(count)
        # img="FIL220204101417166UO4FGZ16I4GZ3Z.jpeg"
        print(img)

        image2 = cv2.imread(imgpath+img)
        image=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


        clahe = cv2.createCLAHE(clipLimit=2,
                                tileGridSize=(8, 8))
        equalized = clahe.apply(image)



        per, blurext = blur_detect(equalized, threshold)

        font = cv2.FONT_HERSHEY_SIMPLEX

            # org
        org = (50, 50)

            # fontScale
        fontScale = 1

            # Blue color in BGR
        color = (255, 0, 0)

            # Line thickness of 2 px
        thickness = 2

            # Using cv2.putText() method
        image = cv2.putText(image2, str(blurext*100), org, font,
                             fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite("out2/"+img,image)
        # print(f"___{img}___  Image is blurred with blurriness : {blurext},per:{per}")

    # if blurext>0.87:
    #     print(f"___{img }___  Image is blurred with blurriness : {blurext},per:{per}")
    #     cv2.imwrite("blurred/"+img,image)
    # elif 0.78<blurext<0.87:
    #     print(f"___{img })___  This is image passed the blurriness check with blurriness :{blurext},per:{per}")
    #     cv2.imwrite("clear_with_high_blur/" + img, image)
    # else:
    #     cv2.imwrite("clear/" + img, image)
