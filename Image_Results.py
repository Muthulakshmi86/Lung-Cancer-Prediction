import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

No_of_Dataset = 3


def Image_Results():
    I = [[32, 54, 126, 55, 50], [2, 5, 8, 12, 32], [12, 64, 71, 75, 76]]
    for n in range(No_of_Dataset):
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('Groundtruth_' + str(n + 1) + '.npy', allow_pickle=True)
        UNET = np.load('Unet_' + str(n + 1) + '.npy', allow_pickle=True)
        RESUNET = np.load('Deeplab_' + str(n + 1) + '.npy', allow_pickle=True)
        Mobile = np.load('MobileNet_' + str(n + 1) + '.npy', allow_pickle=True)
        PROPOSED = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(len(I[n])):
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(Images[I[n][i]])
            plt.subplot(2, 3, 2)
            plt.title('GroundTruth')
            plt.imshow(GT[I[n][i]])
            plt.subplot(2, 3, 3)
            plt.title('UNET')
            plt.imshow(UNET[I[n][i]])
            plt.subplot(2, 3, 4)
            plt.title('FCN')
            plt.imshow(RESUNET[I[n][i]])
            plt.subplot(2, 3, 5)
            plt.title('DenseUnet')
            plt.imshow(Mobile[I[n][i]])
            plt.subplot(2, 3, 6)
            plt.title('PROPOSED')
            plt.imshow(PROPOSED[I[n][i]])
            plt.tight_layout()
            plt.show()


def Sample_Images():
    for n in range(No_of_Dataset):
        Orig = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        ind = [1, 10, 15, 20, 25, 30]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Dataset " + str(n + 1))
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
