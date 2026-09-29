import numpy as np

from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_Res_BiRNN import Model_Res_BiRNN
from Relief_score import reliefF


def objfun_Feat(Soln):
    Feat1 = Global_Vars.Feat1
    Feat2 = Global_Vars.Feat2
    Feat3 = Global_Vars.Feat3
    Feat4 = Global_Vars.Feat4
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Set1 = Feat1 * sol[0]
            Set2 = Feat2 * sol[1]
            Set3 = Feat3 * sol[2]
            Set4 = Feat4 * sol[3]
            feat = np.concatenate((Set1, Set2, Set3, Set4), axis=1)
            rscore = reliefF(np.array(feat), np.array(Tar.reshape(-1)))
            Correlation_Coefficient = np.corrcoef(feat, Tar)
            chi_squared_stat = (((feat - Tar) ** 2) / feat).sum().sum()
            Fitn[i] = 1 / (rscore + chi_squared_stat + Correlation_Coefficient)
    else:
        sol = Soln
        Set1 = Feat1 * sol[0]
        Set2 = Feat2 * sol[1]
        Set3 = Feat3 * sol[2]
        Set4 = Feat4 * sol[3]
        feat = np.concatenate((Set1, Set2, Set3, Set4), axis=1)
        rscore = reliefF(np.array(feat), np.array(Tar.reshape(-1)))
        Correlation_Coefficient = np.corrcoef(feat, Tar)
        chi_squared_stat = (((feat - Tar) ** 2) / feat).sum().sum()
        Fitn = 1 / (rscore + chi_squared_stat + Correlation_Coefficient)
        return Fitn


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_Res_BiRNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(Test_Target, pred)
            Fitn[i] = 1 / Eval[16]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_Res_BiRNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(Test_Target, pred)
        Fitn = 1 / Eval[16]
        return Fitn
