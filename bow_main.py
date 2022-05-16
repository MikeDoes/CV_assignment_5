import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import select
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = None  # numpy array, [nPointsX*nPointsY, 2]

    # todo
    vPoints = np.zeros((nPointsX * nPointsY, 2))
    w, h = img.shape

    # Are these integers? Should we make x Points integers
    mult_x = (w - border - 1) / nPointsX

    mult_y = (h - border - 1) / nPointsY

    xPoints = [int(i * mult_x + 8) for i in range(nPointsX)]
    yPoints = [int(i * mult_y + 8) for i in range(nPointsY)]
    # Wrap around i*mult_x + 8 the term int

    counter_g = 0
    for x in xPoints:
        for y in yPoints:
            vPoints[counter_g][0] = x
            vPoints[counter_g][1] = y
            counter_g += 1

    # CLEARED FOR DEBUGGING
    return vPoints


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    # to calculate the derivatives from an image
    
    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    # What is cv2.CV_16S? a numerical type in CV

    # tan^-1(dy  / dx)
    orientation = np.arctan2(grad_x, grad_y) * 180 / np.pi

    _, bin_edges_orientation = np.histogram(orientation, bins=nBins)
    # bin_edges_orientation = np.array([i * 45 for i in range(9)])

    descriptors = (
        []
    )  # list of descriptors for the current image, each entry is one 128-d vector for a grid point

    for point in vPoints:
        # nPointsX * nPointsY
        point_descriptor = []
        for pixel_x in range(w):
            # 4 width
            for pixel_y in range(h):
                # 4 height
                x_coord = int(point[0] - 2 + pixel_x)
                y_coord = int(point[1] - 2 + pixel_y)

                selection = orientation[x_coord, y_coord]
                # 8 for histogram size of 8 bins
                histogram, _ = np.histogram(selection, bins=bin_edges_orientation)
                point_descriptor += [histogram]

        descriptors += [point_descriptor]

    descriptors = np.asarray(
        descriptors
    )  # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)

    # Checking that we do have the right shape
    descriptors = np.reshape(descriptors, (vPoints.shape[0], 128))

    # CLEARED FOR DEBUGGING
    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, "*.png")))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, "*.png")))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []
    # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # todo
        # opens all images, creates the edges using HOG and then returns the centers

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        vFeatures += [descriptors]

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print("number of extracted features: ", len(vFeatures))

    # Cluster the features using K-Means
    print("clustering ...")
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]

    # CLEARED FOR DEBUGGING, REUSINE CLEAREd FUNCTIONS
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """

    # TODO

    histo = np.zeros(vCenters.shape[0])
    for descriptor in vFeatures:
        dist = np.linalg.norm(vCenters - descriptor, axis=1)
        chosen_cluster_center = np.argmin(dist)
        histo[chosen_cluster_center] += 1

    # Return a histogram based on the cluster centers
    # Error was in labeling np.linalg norm with axis = 0 instea of 1
    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, "*.png")))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]
        # TODO

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        # Adding to the histogram

        vBoW += [bow_histogram(descriptors, vCenters)]

    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW


def bow_recognition_nearest(histogram, vBoWPos, vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # ToDo

    # It shows that all elements in negative have 100, 0,0,0,0,0,0,0 Histograms
    """ DistNeg = np.argmin(np.linalg.norm(vBoWNeg - histogram))
    print('Item with the smallest distance', vBoWNeg[DistNeg])
    print('Histogram', histogram)
    print('Rest of items in set', DistNeg) """

    DistNeg = np.min(np.linalg.norm(vBoWNeg - histogram, axis=1))

    DistPos = np.min(np.linalg.norm(vBoWPos - histogram, axis=1))

    if DistPos < DistNeg:
        sLabel = 1
    else:
        sLabel = 0
    return sLabel


if __name__ == "__main__":
    nameDirPos_train = "data/data_bow/cars-training-pos"
    nameDirNeg_train = "data/data_bow/cars-training-neg"
    nameDirPos_test = "data/data_bow/cars-testing-pos"
    nameDirNeg_test = "data/data_bow/cars-testing-neg"

    # TODO
    k = 20
    numiter = 99

    # TODO END

    print("creating codebook ...")
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print("creating bow histograms (pos) ...")
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print("creating bow histograms (neg) ...")
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print("creating bow histograms for test set (pos) ...")
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print("testing pos samples ...")
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i : (i + 1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print("test pos sample accuracy:", acc_pos)

    # test neg samples
    print("creating bow histograms for test set (neg) ...")
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print("testing neg samples ...")
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i : (i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print("test neg sample accuracy:", acc_neg)
