import pandas as pd
import os
import cv2
import pydicom as dicom
from numpy import matlib
from skimage.feature import graycomatrix
from GAO import GAO
from GLCM import contrast_feature, homogeneity_feature, energy_feature, correlation_feature, dissimilarity_feature
from Global_Vars import Global_Vars
from LBP import lbp_calculated_pixel
from MOA import MOA
from Model_CNN import Model_CNN
from Model_DHNN import Model_DHNN
from Model_DeeplabV3 import Model_DeeplabV3
from Model_RCNN import Model_RCNN
from Model_Res_BiRNN import Model_Res_BiRNN
from Model_VGG19 import Model_VGG19
from Image_Results import *
from Plot_results import *
from Proposed import Proposed
from RBMO import RBMO
from SFOA import SFOA
from Shape_Feat import shape_features_extraction
from Size_Feat import extract_image_size_features
from objfun_feat import objfun_Feat, objfun

No_of_Dataset = 3


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (256, 256))
    return image


# Write Images for DeeplabV3
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Ground_Truth = np.load('Groundtruth_' + str(n + 1) + '.npy', allow_pickle=True)
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(len(Images)):
            print(i)
            gt = Ground_Truth[i]
            image = Images[i]
            cv.imwrite('./Seg_Image/Dataset_' + str(n + 1) + '/Images/' + 'image-%05d.png' % (i + 1), image)
            cv.imwrite('./Seg_Image/Dataset_' + str(n + 1) + '/Mask/' + 'image-%04d.png' % (i + 1), gt)

# Read Dataset 1
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Dataset_1/NSCLC Radiogenomics'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            subfolder = folder + '/' + in_dir[j]
            dir = os.listdir(subfolder)
            for k in range(len(dir)):
                folders = subfolder + '/' + dir[k]
                sub_dir = os.listdir(folders)
                for m in range(len(sub_dir)):
                    print(i, j, k, m)
                    FileName = folders + '/' + sub_dir[m]
                    ds = dicom.dcmread(FileName)
                    image = (ds.pixel_array / 13).astype('uint8')
                    if len(image.shape) > 2:
                        for w in range(len(image)):
                            imagess = image[w]
                    images_1 = cv.resize(image, (512, 512))
                    Image.append(images_1)
                    df = pd.read_csv('./Dataset_1/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv')
                    Id = df.values[:, 0]
                    Tar = df.values[:, 17]
                    Split_data = FileName.split('/')
                    if Split_data[3] in Id:
                        Ind = np.where(Id == Split_data[3])
                        Target.append(Tar[Ind[0]])
    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_1.npy', index)
    np.save('Image_1.npy', Shuffled_Datas)
    np.save('Target_1.npy', Shuffled_Target)

# Read Dataset 2
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Dataset_2/Lung-PET-CT-Dx'
    out_dir = os.listdir(path)
    del out_dir[0]
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            subfolder = folder + '/' + in_dir[j]
            dir = os.listdir(subfolder)
            for k in range(len(dir)):
                folders = subfolder + '/' + dir[k]
                sub_dir = os.listdir(folders)
                for m in range(len(sub_dir)):
                    print(i, j, k, m)
                    FileName = folders + '/' + sub_dir[m]
                    ds = dicom.dcmread(FileName)
                    image = (ds.pixel_array / 13).astype('uint8')
                    split_Data = out_dir[i].split('-')
                    if split_Data[1][0] == 'A':
                        Target.append(0)
                    elif split_Data[1][0] == 'B':
                        Target.append(1)
                    elif split_Data[1][0] == 'E':
                        Target.append(2)
                    else:
                        Target.append(3)
                    images = cv.resize(image, [256, 256])
                    Image.append(images)
    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_2.npy', index)
    np.save('Image_2.npy', Shuffled_Datas)
    np.save('Target_2.npy', Shuffled_Target)

# Read Dataset 3
an = 0
if an == 1:
    Image = []
    Target = []
    Path = './Dataset_3/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
    out_dir = os.listdir(Path)
    del out_dir[1]
    for i in range(len(out_dir)):
        folder = Path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            FileName = folder + '/' + in_dir[j]
            Img = ReadImage(FileName)
            Image.append(Img)
            Target.append(i)

    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_3.npy', index)
    np.save('Image_3.npy', Shuffled_Datas)
    np.save('Target_3.npy', Shuffled_Target)

# Generate Groundtruth
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Mask_Img = []
        for j in range(len(Images)):
            print(j, len(Images))
            img = Images[j]
            # Apply Gaussian Blur
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            # Otsu thresholding (binary inverse)
            thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # Connected Components Analysis
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4, ltype=cv2.CV_32S)
            # Initialize blank mask
            output_mask = np.zeros(img.shape, dtype="uint8")
            # Filter components by area and draw them as white
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if 0 < area < 1000:  # Change area limits if needed
                    component_mask = (labels == i).astype("uint8") * 255
                    output_mask = cv2.bitwise_or(output_mask, component_mask)
            Mask_Img.append(output_mask)
        np.save('Groundtruth_' + str(n + 1) + '.npy', np.asarray(Mask_Img))

# DeeplabV3 Segmentation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Unet_Path = './Seg_Image/Dataset_' + str(n + 1) + '/'
        Image_Path = 'Images'
        Mask_Path = 'Mask'
        Predict_Path = 'Predict - Images'
        Data = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Gt = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Eval, Images = Model_DeeplabV3(Unet_Path, Image_Path, Mask_Path, Predict_Path, Data, Gt, Target)
        np.save('Evaluate_Seg_all.npy', Eval)
        np.save('Proposed_' + str(n + 1) + '.npy', Images)

# Shape Feature Extraction
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        image = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        results = shape_features_extraction(image)
        np.save('Shape_Feat_' + str(n + 1) + '.npy', results)

# Size Feature Extraction
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        image = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = extract_image_size_features(image)
        np.save('Size_Feat_' + str(n + 1) + '.npy', Feat)

####### Texture Feature Extraction ##########
# Local Binary pattern
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        image = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        LBP = []
        for i in range(len(image)):
            print(i)
            image1 = image[i]
            height, width = image1.shape
            img_lbp = np.zeros((height, width),
                               np.uint8)

            for i in range(0, height):
                for j in range(0, width):
                    img_lbp[i, j] = lbp_calculated_pixel(image1, i, j)
            LBP.append(img_lbp)
        np.save("LBP_" + str(n + 1) + ".npy", np.asarray(LBP))

# GLCM
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Images = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        matrix_coocurrence = []
        GLCM = []
        GLCMFeat = []
        for i in range(len(Images)):
            print(i)
            image = Images[i]
            bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
            inds = np.digitize(image, bins)
            max_value = inds.max() + 1
            matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                              normed=False, symmetric=False)
            A = (contrast_feature(matrix_coocurrence))
            B = (dissimilarity_feature(matrix_coocurrence))
            C = (homogeneity_feature(matrix_coocurrence))
            D = (energy_feature(matrix_coocurrence))
            E = (correlation_feature(matrix_coocurrence))
            GLCM = np.append(E, np.append(D, np.append(C, np.append(B, A))))
            GLCMFeat.append(GLCM)
        np.save("GLCM_" + str(n + 1) + ".npy", np.asarray(GLCMFeat))

# Pattern Concatenation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        LBP = np.load('LBP_' + str(n + 1) + '.npy', allow_pickle=True)
        GLCM = np.load('GLCM_' + str(n + 1) + '.npy', allow_pickle=True)
        Lbp = np.reshape(LBP, (LBP.shape[0], LBP.shape[1] * LBP.shape[2]))
        Pattern = np.concatenate((Lbp, GLCM), axis=1)
        np.save('Texture_' + str(n + 1) + '.npy', Pattern)

# Feature Extraction using VGG19
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = Model_VGG19(Image, Target)
        np.save('VGG19_Feat_' + str(n + 1) + '.npy', Feat)

# Optimization for Weighted Fused Features
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feat1 = np.load('Shape_Feat_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Dataset
        Feat2 = np.load('Size_Feat_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat3 = np.load('Texture_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat4 = np.load('VGG19_Feat_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_Vars.Feat1 = Feat1
        Global_Vars.Feat2 = Feat2
        Global_Vars.Feat3 = Feat3
        Global_Vars.Feat4 = Feat4
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 4
        xmin = matlib.repmat(0.01 * np.ones((1, Chlen)), Npop, 1)
        xmax = matlib.repmat(0.99 * np.ones((1, Chlen)), Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = objfun_Feat
        Max_iter = 50

        print("GAO...")
        [bestfit1, fitness1, bestsol1, time1] = GAO(initsol, fname, xmin, xmax, Max_iter)  # GAO

        print("SFOA...")
        [bestfit2, fitness2, bestsol2, time2] = SFOA(initsol, fname, xmin, xmax, Max_iter)  # SFOA

        print("RBMO...")
        [bestfit3, fitness3, bestsol3, time3] = RBMO(initsol, fname, xmin, xmax, Max_iter)  # RBMO

        print("MOA...")
        [bestfit4, fitness4, bestsol4, time4] = MOA(initsol, fname, xmin, xmax, Max_iter)  # MOA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Enhanced MOA

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

#  Weighted Features
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feat1 = np.load('Shape_Feat_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat2 = np.load('Size_Feat_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat3 = np.load('Texture_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat4 = np.load('VGG19_Feat_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        bests = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        sol = bests[4, :]
        Set1 = Feat1 * sol[0]
        Set2 = Feat2 * sol[1]
        Set3 = Feat3 * sol[2]
        Set4 = Feat4 * sol[3]
        Feat = np.concatenate((Set1, Set2, Set3, Set4), axis=1)
        np.save('Selected_Feature_' + str(n + 1) + '.npy', Feat)

# Optimization for Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Selected_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron count, Learning Rate, Activation Function
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("GAO...")
        [bestfit1, fitness1, bestsol1, time] = GAO(initsol, fname, xmin, xmax, Max_iter)  # GAO

        print("SFOA...")
        [bestfit2, fitness2, bestsol2, time1] = SFOA(initsol, fname, xmin, xmax, Max_iter)  # SFOA

        print("RBMO...")
        [bestfit3, fitness3, bestsol3, time2] = RBMO(initsol, fname, xmin, xmax, Max_iter)  # RMBO

        print("MOA...")
        [bestfit4, fitness4, bestsol4, time3] = MOA(initsol, fname, xmin, xmax, Max_iter)  # MOA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('Sol_' + str(n + 1) + '.npy', BestSol)

# Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Data = np.load('Selected_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('Sol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Learn_Per = [0.35, 0.45, 0.55, 0.65, 0.75]
        for act in range(len(Learn_Per)):
            learnperc = round(Data.shape[0] * Learn_Per[act])  # Split Training and Testing Datas
            Train_Data = Data[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Data[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_Res_BiRNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                                   sol)  # Model Res BiRNN With optimization
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model CNN
            Eval[6, :], pred2 = Model_RCNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model RCNN
            Eval[7, :], pred3 = Model_DHNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model DHNN
            Eval[8, :], pred4 = Model_Res_BiRNN(Train_Data, Train_Target, Test_Data,
                                                Test_Target)  # Model Res BiRNN Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Evaluate_act.npy', Eval_all)  # Save Eval all

# Classification for Kfold
an = 0
if an == 1:
    Eval = []
    for n in range(No_of_Dataset):
        Feat = np.load('Selected_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('Sol_' + str(n + 1) + '.npy', allow_pickle=True)
        K = 5
        Per = 1 / 5
        Perc = round(Feat.shape[0] * Per)
        Fold = []
        for i in range(K):
            Eval = np.zeros((10, 25))
            for j in range(5):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
                Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
                test_index = np.arange(i * Perc, ((i + 1) * Perc))
                total_index = np.arange(Feat.shape[0])
                train_index = np.setdiff1d(total_index, test_index)
                Train_Data = Feat[train_index, :]
                Train_Target = Target[train_index, :]
                Eval[j, :], pred = Model_Res_BiRNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                                   sol)  # Model Res BiRNN With optimization
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model CNN
            Eval[6, :], pred2 = Model_RCNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model RCNN
            Eval[7, :], pred3 = Model_DHNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model DHNN
            Eval[8, :], pred4 = Model_Res_BiRNN(Train_Data, Train_Target, Test_Data,
                                                Test_Target)  # Model Res BiRNN Without optimization
            Eval[9, :] = Eval[4, :]
            Fold.append(Eval)
        Eval.append(Fold)
    np.save('Evaluate_all.npy', Eval)  # Save Eval all

plotConvResults()
Plots_Results()
Plot_ROC_Curve()
plot_seg_results()
Table()
Plot_Proposed_Results()
Image_Results()
Sample_Images()
