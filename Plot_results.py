import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pylab
from matplotlib.lines import Line2D
from sklearn import metrics

warnings.filterwarnings("ignore")

No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GAO-MCA-RBi-RNN', 'SFOA-MCA-RBi-RNN', 'RBMO-MCA-RBi-RNN', 'MOA-MCA-RBi-RNN',
                 'RAEMOA-MCA-RBi-RNN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    cls = ['CNN', 'Mask RCNN', 'DHNN', 'MCA-RBi-RNN', 'RAEMOA-MCA-RBi-RNN']
    for a in range(No_of_Dataset):
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'GAO-MCA-RBi-RNN', 'SFOA-MCA-RBi-RNN', 'RBMO-MCA-RBi-RNN', 'MOA-MCA-RBi-RNN',
                 'RAEMOA-MCA-RBi-RNN']
    Classifier = ['BatchSize', 'CNN', 'Mask RCNN', 'DHNN', 'MCA-RBi-RNN', 'RAEMOA-MCA-RBi-RNN']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    # Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    # Table_Terms = [0, 2, 4, 6, 9, 15]
    Graph_Terms = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).astype(int)
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['4', '8', '16', '32', '48']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batchsize)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('---------------------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plots_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    bar_width = 0.15
    kfold = [1, 2, 3, 4, 5]
    Algorithm = ['GAO-MCA-RBi-RNN', 'SFOA-MCA-RBi-RNN', 'RBMO-MCA-RBi-RNN', 'MOA-MCA-RBi-RNN', 'RAEMOA-MCA-RBi-RNN']
    Classifier = ['CNN', 'Mask RCNN', 'DHNN', 'MCA-RBi-RNN', 'RAEMOA-MCA-RBi-RNN']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Dataset- ' + str(i+1) + Terms[Graph_Terms[j]] + ' Algorithm Comparison of KFold')
            X = np.arange(len(kfold))
            plt.bar(X + 0.00, Graph[:, 0], color='darkblue', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[0])
            plt.bar(X + 0.15, Graph[:, 1], color='#9400d3', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[1])
            plt.bar(X + 0.30, Graph[:, 2], color='#a30046', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[2])
            plt.bar(X + 0.45, Graph[:, 3], color='#00bbf9', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[4])
            plt.xticks(X + bar_width * 2, ['1', '2', '3', '4', '5'], fontsize=12,
                       fontname="Arial",
                       fontweight='bold', color='k')
            plt.xlabel('Kfold', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['darkblue', '#9400d3', '#a30046', '#00bbf9', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                       frameon=False, ncol=len(Algorithm))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Dataset- ' + str(i+1) + Terms[Graph_Terms[j]] + ' Method Comparison of Kfold')
            X = np.arange(len(kfold))
            plt.bar(X + 0.00, Graph[:, 5], color='yellowgreen', edgecolor='w', width=0.15, label=Classifier[0])
            plt.bar(X + 0.15, Graph[:, 6], color='gold', edgecolor='w', width=0.15, label=Classifier[1])
            plt.bar(X + 0.30, Graph[:, 7], color='mediumpurple', edgecolor='w', width=0.15, label=Classifier[2])
            plt.bar(X + 0.45, Graph[:, 8], color='sandybrown', edgecolor='w', width=0.15, label=Classifier[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Classifier[4])
            plt.xticks(X + bar_width * 2, ['1', '2', '3', '4', '5'], fontname="Arial",
                       fontsize=12,
                       fontweight='bold', color='k')
            plt.xlabel('Kfold', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                       frameon=False, ncol=len(Classifier))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def plot_seg_results():
    Eval_all = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Methods = ['TERMS', 'Unet', 'FCN', 'DenseUnet', 'DeeplabV3', 'ASPPDV3-AM']
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Methods[0], Statistics[1::3])
            Table.add_column(Methods[1], stats[i, 0, 1::3])
            Table.add_column(Methods[2], stats[i, 1, 1::3])
            Table.add_column(Methods[3], stats[i, 2, 1::3])
            Table.add_column(Methods[4], stats[i, 3, 1::3])
            Table.add_column(Methods[5], stats[i, 4, 1::3])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Comparison for Segmentation', '--------------------------------------------------')
            print(Table)

            X = np.arange(len(Statistics) - 3)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Classification Comparison')
            ax.bar(X + 0.00, stats[i, 0, 0:3:2], color='brown', edgecolor='w', width=0.15, label=Methods[1])
            ax.bar(X + 0.15, stats[i, 1, 0:3:2], color='orchid', edgecolor='w', width=0.15, label=Methods[2])
            ax.bar(X + 0.30, stats[i, 2, 0:3:2], color='#FE420F', edgecolor='w', width=0.15, label=Methods[3])
            ax.bar(X + 0.45, stats[i, 3, 0:3:2], color='green', edgecolor='w', width=0.15, label=Methods[4])
            ax.bar(X + 0.60, stats[i, 4, 0:3:2], color='k', edgecolor='w', width=0.15, label=Methods[5])
            colors = ['brown', 'orchid', '#FE420F', 'green', 'k']
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in colors]
            plt.legend(dot_markers, ['Unet', 'FCN', 'DenseUnet', 'DeeplabV3', 'ASPPDV3-AM'], loc='upper center',
                       bbox_to_anchor=(0.5, 1.23), fontsize=10,
                       frameon=False, ncol=3)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.30, ('BEST', 'MEAN'), fontname="Arial", fontsize=12, fontweight='bold', color='#1d3557')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#1d3557')
            path = "./Results/Dataset_%s_%s_met.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Plot_Proposed_Results():
    eval = np.load('Evaluate_act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Learning_Percentage = ['35', '45', '55', '65', '75']
    Graph_Terms = [0, 3, 8, 12]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :5, :, Graph_Terms[j] + 4]
            Alg_Graph = np.array([Graph[:, 0], Graph[:, 1], Graph[:, 2], Graph[:, 3], Graph[:, 4]])
            values = [Alg_Graph[0, 4], Alg_Graph[1, 4], Alg_Graph[2, 4], Alg_Graph[3, 4], Alg_Graph[4, 4]]
            percent_labels = [f'{value:.2f}' for value in values]
            bar_colors = ['#003f5c', '#bc5090', '#ffa600', '#58508d', '#2f4b7c', '#00b6c4', '#94c849']
            arrow_colors = ['#003f5c', '#bc5090', '#ffa600', '#58508d', '#2f4b7c', '#00b6c4', '#94c849']
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(Learning_Percentage, values, color=bar_colors, width=0.6)
            yticks = np.arange(10, 100, 10)
            ax.set_yticks(yticks)
            ax.set_ylim(0, 100)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)
            for y in yticks:
                ax.text(-0.58, y, '▶', fontsize=10, va='center', color='gray')
            for bar, label, arrow_color in zip(bars, percent_labels, arrow_colors):
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.text(x, y + 3, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='white',
                        bbox=dict(facecolor=arrow_color, boxstyle='round,pad=0.2', edgecolor='none'))
                ax.text(x, y + 2, '▼', ha='center', va='center', fontsize=12, color=arrow_color)
            ax.set_xlabel('Learning Percentage →', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold',
                          color='#35530a')
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Proposed_bar.png" % (n + 1, Terms[Graph_Terms[j]])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                'Dataset-' + str(n + 1) + ' Learning Percentage vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    # plotConvResults()
    # Plots_Results()
    # Plot_ROC_Curve()
    # plot_seg_results()
    Table()
    # Plot_Proposed_Results()
