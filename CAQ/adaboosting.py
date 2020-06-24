import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_curve, auc

MAX_LENGTH = 100
n_estimators = 200

def load_data(filename):
    data = []
    with open(filename,'r',encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            i += 1
            if i % 2 == 0:
                seq = line.strip()
                Length = len(seq)
                AciDic = {}
                for char in seq:
                    if AciDic.get(char) is None:
                        AciDic[char] = 1
                    else:
                        AciDic[char] += 1
                for k,v in AciDic.items():
                    v /= Length
                    AciDic[k] = v
                encode_seq = [0.0]*MAX_LENGTH
                for j in range(Length):
                    encode_seq[j] = AciDic[seq[j]]
                data.append(encode_seq)
    data = np.array(data)
    return data


def acu_curve(y, prob):
    # y真实prob预测
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontdict={'size':14})
    plt.ylabel('True Positive Rate',fontdict={'size':14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('AUC')
    plt.legend(loc="lower right",prop = {'size':14})

    plt.show()

if __name__ == '__main__':
    for i in range(2):
        train_pos_data = load_data('./data/Layer%d-positive.txt' %(i+1))
        train_pos_label = np.ones(train_pos_data.shape[0])
        train_neg_data = load_data('./data/Layer%d-negative.txt' %(i+1))
        train_neg_label = np.zeros(train_neg_data.shape[0])
        test_pos_data = load_data('./data/Layer%d-Ind-positive.txt' %(i+1))
        test_pos_label = np.ones(test_pos_data.shape[0])
        test_neg_data = load_data('./data/Layer%d-Ind-negative.txt' %(i+1))
        test_neg_label = np.zeros(test_neg_data.shape[0])
        train_data = np.vstack((train_pos_data, train_neg_data))
        train_label = np.hstack((train_pos_label, train_neg_label))
        test_data = np.vstack((test_pos_data, test_neg_data))
        test_label = np.hstack((test_pos_label, test_neg_label))
        dt_stump = DecisionTreeClassifier(max_depth=8, min_samples_leaf=2)
        dt_stump.fit(train_data, train_label)
        dt_stump_err = 1.0 - dt_stump.score(test_data, test_label)
        dt = DecisionTreeClassifier()
        dt.fit(train_data, train_label)
        dt_err = 1.0 - dt.score(test_data, test_label)
        ada = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators, learning_rate=0.3)
        ada.fit(train_data, train_label)
        fig = plt.figure()
        # 设置 plt 正确显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']
        ax = fig.add_subplot(111)
        ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-', label=u'决策树弱分类器 错误率')
        ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label=u'决策树模型 错误率')
        ada_err = np.zeros((n_estimators,))
        # 遍历每次迭代的结果 i 为迭代次数, pred_y 为预测结果
        for i, pred_y in enumerate(ada.staged_predict(test_data)):
            # 统计错误率
            ada_err[i] = zero_one_loss(pred_y, test_label)
        # 绘制每次迭代的 AdaBoost 错误率
        ax.plot(np.arange(n_estimators) + 1, ada_err, label='AdaBoost Test 错误率', color='orange')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('错误率')
        leg = ax.legend(loc='upper right', fancybox=True)
        plt.show()
        pred_label = ada.predict_proba(test_data)
        acu_curve(test_label, pred_label[:, 1])



