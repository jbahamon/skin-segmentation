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
        0.11,
        0.0676,
        0.0755,
        0.05,
        0.0667,
        0.0749 ]

skin_mu = [
        np.array([73.53,  29.94,  17.76]),
        np.array([248.71, 233.94, 217.49]),
        np.array([161.68, 116.25, 96.95]),
        np.array([186.07, 136.62, 114.40]),
        np.array([189.26, 98.37,  51.18]),
        np.array([247.0,  152.2,  90.84]),
        np.array([150.1,  72.66,  37.76]),
        np.array([206.85, 171.09, 156.34]),
        np.array([212.78, 152.82, 120.04]),
        np.array([234.87, 175.43, 138.94]),
        np.array([151.19, 97.74,  74.59]),
        np.array([120.52, 77.55,  59.82]),
        np.array([192.2,  119.62, 82.32]),
        np.array([214.29, 136.08, 87.24]),
        np.array([99.57,  54.33,  38.06]),
        np.array([231.88, 203.08, 176.91]) ]

skin_sigma = [
        np.array([765.40, 121.44, 112.8]),
        np.array([39.94,  154.44, 396.05]),
        np.array([291.03, 60.48,  162.85]),
        np.array([274.95, 64.60,  192.27]),
        np.array([633.18, 222.40, 250.69]),
        np.array([65.23,  691.53, 609.92]),
        np.array([408.63, 200.77, 257.57]),
        np.array([530.08, 155.08, 572.79]),
        np.array([160.57, 84.52,  243.9]),
        np.array([163.8,  121.57, 279.22]),
        np.array([425.40, 73.56,  175.11]),
        np.array([330.45, 70.34,  151.82]),
        np.array([152.76, 82.14,  259.15]),
        np.array([204.9,  140.17, 270.19]),
        np.array([448.13, 90.18,  151.29]),
        np.array([178.38, 156.27, 404.99]) ]

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
    np.array([254.37, 254.41, 253.82]),
    np.array([9.39, 8.09, 8.52]),
    np.array([96.57, 96.95, 91.53]),
    np.array([160.44, 162.49, 159.06]),
    np.array([74.98, 63.23, 46.33]),
    np.array([121.83, 60.88, 18.31]),
    np.array([202.18, 154.88, 91.04]),
    np.array([193.06, 201.93, 206.55]),
    np.array([51.88, 57.14, 61.55]),
    np.array([30.88, 26.84, 25.32]),
    np.array([44.97, 85.96, 131.95]),
    np.array([236.02, 236.27, 230.70]),
    np.array([207.86, 131.20, 164.12]),
    np.array([99.83, 148.11, 188.17]),
    np.array([135.06, 131.92, 123.10]),
    np.array([135.96, 103.89, 66.88]) ]

nonskin_sigma = [
    np.array([2.77, 2.81, 5.46]),
    np.array([46.84, 33.59, 32.48]),
    np.array([280.69, 156.79, 436.58]),
    np.array([355.98, 115.89, 591.24]),
    np.array([414.84, 245.95, 361.27]),
    np.array([2502.24, 1383.53, 237.18]),
    np.array([957.42, 1766.94, 1582.52]),
    np.array([562.88, 190.23, 447.28]),
    np.array([344.11, 191.77, 433.40]),
    np.array([222.07, 118.65, 182.41]),
    np.array([651.32, 840.52, 963.67]),
    np.array([225.03, 117.29, 331.95]),
    np.array([494.04, 237.69, 533.52]),
    np.array([955.88, 654.95, 916.70]),
    np.array([350.35, 130.30, 388.43]),
    np.array([806.44, 642.20, 350.36]) ]


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
            x = image[idx_j, idx_i].astype(np.float_)[::-1]

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
                px = nonbin_img[idx_j, idx_i].astype(np.float_)[::-1]

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

    num_images = 30
    thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    points = roc_curve(num_images, thetas)

    for pair in points:
        print pair

