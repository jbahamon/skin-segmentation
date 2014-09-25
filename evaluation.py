#!/usr/bin/python

import cv2
import os, sys
import numpy as np
import math
#Para exportar resultados a LaTeX

#from pylatex import Document, Section, Subsection, Table, Math, TikZ, Axis, \
#        Plot
#from pylatex.numpy import Matrix
#from pylatex.utils import italic, bold


N = 16 # derp

skin_w = [
        0.0294,
        0.0331,
        0.0654,
        0.0756,
        0.0554,
        0.0314,
        0.0454,
        0.0469,
        0.0956,
        0.0763,
        0.1100,
        0.0676,
        0.0755,
        0.0500,
        0.0667,
        0.0749 ]

skin_mu = [
        np.array([017.76, 029.94, 073.53]),
        np.array([217.49, 233.94, 249.71]),
        np.array([096.95, 116.25, 161.68]),
        np.array([114.40, 136.62, 186.07]),
        np.array([051.18, 098.37, 189.26]),
        np.array([090.84, 152.20, 247.00]),
        np.array([037.76, 072.66, 150.10]),
        np.array([156.34, 171.09, 206.85]),
        np.array([120.04, 152.82, 212.78]),
        np.array([138.94, 175.43, 234.87]),
        np.array([074.59, 097.74, 151.19]),
        np.array([059.82, 077.55, 120.52]),
        np.array([082.32, 119.62, 192.20]),
        np.array([087.24, 136.08, 214.29]),
        np.array([038.06, 054.33, 099.57]),
        np.array([176.91, 203.08, 238.88]) ]

skin_sigma = [
        np.array([112.80, 121.44, 765.40]),
        np.array([396.05, 154.44, 039.94]),
        np.array([162.85, 060.48, 291.03]),
        np.array([198.27, 064.60, 274.95]),
        np.array([250.69, 222.40, 633.18]),
        np.array([609.92, 691.53, 065.23]),
        np.array([257.57, 200.77, 408.63]),
        np.array([572.79, 155.08, 530.08]),
        np.array([243.90, 084.52, 160.57]),
        np.array([279.22, 121.57, 163.80]),
        np.array([175.11, 073.56, 425.40]),
        np.array([151.82, 070.34, 330.45]),
        np.array([259.15, 092.14, 152.76]),
        np.array([270.19, 140.17, 204.90]),
        np.array([151.29, 090.18, 448.13]),
        np.array([404.99, 156.27, 178.38]) ]

nonskin_w = [
    0.0637,
    0.0516,
    0.0864,
    0.0636,
    0.0747,
    0.0365,
    0.0349,
    0.0649,
    0.0656,
    0.1189,
    0.0362,
    0.0849,
    0.0368,
    0.0389,
    0.0943,
    0.0477 ]

nonskin_mu = [
    np.array([253.82, 254.41, 254.37]),
    np.array([008.52, 008.09, 009.39]),
    np.array([091.53, 096.95, 096.57]),
    np.array([159.06, 162.49, 160.44]),
    np.array([046.33, 063.23, 074.98]),
    np.array([018.31, 060.88, 121.83]),
    np.array([091.04, 154.88, 202.18]),
    np.array([206.55, 201.93, 193.06]),
    np.array([061.55, 057.14, 051.88]),
    np.array([025.32, 026.84, 030.88]),
    np.array([131.95, 085.96, 044.97]),
    np.array([230.70, 236.27, 236.02]),
    np.array([164.12, 191.20, 207.86]),
    np.array([188.17, 148.11, 099.83]),
    np.array([123.10, 131.92, 135.06]),
    np.array([066.88, 103.89, 135.96]) ]

nonskin_sigma = [
    np.array([0005.46, 0002.81, 0002.77]),
    np.array([0032.48, 0033.59, 0046.84]),
    np.array([0436.58, 0156.79, 0280.69]),
    np.array([0591.24, 0115.89, 0355.98]),
    np.array([0361.27, 0245.95, 0414.84]),
    np.array([0237.18, 1383.53, 2502.24]),
    np.array([1582.52, 1766.94, 0957.42]),
    np.array([0447.28, 0190.23, 0562.88]),
    np.array([0433.40, 0191.77, 0344.11]),
    np.array([0182.41, 0118.65, 0222.07]),
    np.array([0963.67, 0840.52, 0651.32]),
    np.array([0331.95, 0117.29, 0225.03]),
    np.array([0533.52, 0237.69, 0494.04]),
    np.array([0916.70, 0654.95, 0955.88]),
    np.array([0388.43, 0130.30, 0350.35]),
    np.array([0350.36, 0642.20, 0806.44]) ]


def mix_of_gaussians(image, theta):


    dest_image = np.zeros(image.shape, dtype = np.uint8)

    # Los factores que van con los sumandos son los mismos siempre asi que
    # mejor precalcular :)

    val = 1.0 / (2.0 * np.pi)** (2.0/3.0)

    skin_factors = [ val * skin_w[i] / np.sqrt(np.absolute(np.prod(skin_sigma[i])))  for i in xrange(N) ]

    nonskin_factors = [ val * nonskin_w[i] / np.sqrt(np.absolute(np.prod(nonskin_sigma[i])))  for i in xrange(N) ]

    #for each px
    (y, _x, fef) = image.shape

    for idx_i in xrange(_x):
        for idx_j in xrange(y):
            x = image[idx_j, idx_i].astype(np.float_)

            # Estos deberian ser, para cada i, los numeros que van en el
            # exponente :V
            skin_exponents = [ np.sum((x - skin_mu[i])**2) / skin_sigma[i] for i in xrange(N) ]


            skin_P = np.sum([ skin_factors[i] * np.exp( - skin_exponents[i]) for i
                in xrange(N) ])

            nonskin_exponents = [ np.sum((x - nonskin_mu[i])**2) / nonskin_sigma[i] for i in xrange(N) ]

            nonskin_P = np.sum([ nonskin_factors[i] *
                np.exp(- nonskin_exponents[i]) for i in xrange(N) ])

            dest_image[idx_j, idx_i] = 255 if skin_P/nonskin_P > theta else 0

    return dest_image

def roc_curve(num_images, thetas):

    # Los factores que van con los sumandos son los mismos siempre asi que
    # mejor precalcular :)

    val = 1.0 / (2.0 * np.pi)** (2.0/3.0)

    skin_factors = [ val * skin_w[i] / np.sqrt(np.absolute(np.prod(skin_sigma[i])))  for i in xrange(N) ]

    nonskin_factors = [ val * nonskin_w[i] / np.sqrt(np.absolute(np.prod(nonskin_sigma[i])))  for i in xrange(N) ]

    bin_folder = "./images/bin/"
    nonbin_folder = "./images/nonbin/"

    total_positives = 0.0
    total_negatives = 0.0
    TP = [0] * len(thetas)
    FP = [0] * len(thetas)

    for image_idx in xrange(1, num_images + 1):
        current_positives = 0
        print "Procesando imagen " + str(image_idx)
        if image_idx < 10:
            filename = "0" + str(image_idx) + ".png"
        else:
            filename = str(image_idx) + ".png"
        
        bin_img = cv2.imread(bin_folder + filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        bin_img = cv2.threshold(bin_img, 128, 255, cv2.THRESH_BINARY)[1]
        nonbin_img = cv2.imread(nonbin_folder + filename)
        (y, x, fef) = nonbin_img.shape

        for idx_i in xrange(x):
            for idx_j in xrange(y):
                px = nonbin_img[idx_j, idx_i].astype(np.float_)

            # Estos deberian ser, para cada i, los numeros que van en el
            # exponente :V
                skin_exponents = [ np.sum((px - skin_mu[i])**2) / skin_sigma[i]
                        for i in xrange(N) ]
                skin_P = np.sum([ skin_factors[i] * np.exp( - skin_exponents[i])
                    for i in xrange(N) ])

                nonskin_exponents = [ np.sum((px - nonskin_mu[i])**2) /
                        nonskin_sigma[i] for i in xrange(N) ]
                nonskin_P = np.sum([ nonskin_factors[i] * 
                    np.exp(- nonskin_exponents[i]) for i in xrange(N) ])

                true_value = bin_img.item(idx_j, idx_i)
                
                if bin_img.item(idx_j, idx_i) > 0:
                    current_positives += 1

                for i in xrange(len(thetas)):

                    # We have a positive
                    if skin_P/nonskin_P > thetas[i]:
                        if true_value > 0:
                            TP[i] +=1
                        else:
                            FP[i] += 1

        total_negatives += (x * y - current_positives)
        total_positives += current_positives

    TPR = [ positives / total_positives for positives in TP ]
    FPR = [ positives / total_negatives for positives in FP ]

    return zip(FPR, TPR)

def toBin(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (tresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def genROCPoints(nonbinfile, binfile):
    imgNB_Orig = cv2.imread(nonbinfile)
    imgB = toBin(cv2.imread(binfile))
    #Ambas imagenes DEBERIAN tener el mismo tamano
    (y, _x, fef) = imgB.shape
    puntos = []
    for theta in range(0, 1, 0.1):
        imgNB = toBin(mix_of_gaussians(imgNB_Orig), theta)
        classSkin = 0  # let's call this class A'
        classNoSkin = 0  # class B
        FP = 0
        TP = 0
        for idx_i in range(_x):
            for idx_j in range(y):
                pixNB = imgNB[idx_j, idx_i]
                pixB = imgB[idx_j, idx_i]
                #Clase real extrayendo de imagenes procesadas manualmente
                if(pixB == 255):
                    classSkin += 1
                    if (pixNB == 255):
                        TP += 1
                else:
                    classNoSkin += 1
                    if (pixNB == 255):
                        FP += 1
        #False Positive Rate (FPR)= FP/|B| (X axis)
        fpr = FP / classSkin
        #True Positive Rate (TPR)=TP/|A| (Y axis)
        tpr = TP / classNoSkin
        puntos.append((fpr, tpr))
    return puntos

if __name__ == '__main__':

    # img1 = cv2.imread(sys.argv[1])
    # img2 = mix_of_gaussians(img1, float(sys.argv[2]))
    # cv2.imshow("original", img1)
    # cv2.imshow("result", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #num_images = 30
    #thetas = np.concatenate((np.arange(0.1, 0.9, 0.1), np.arange(0.9, 1, 0.01)))

#    points = roc_curve(num_images, thetas)

 #   for pair in points:
  #      print pair

    cv2.imwrite("02.png",mix_of_gaussians(cv2.imread("images/nonbin/02.png"),0.2))
    cv2.imwrite("17.png",mix_of_gaussians(cv2.imread("images/nonbin/17.png"),0.9))
