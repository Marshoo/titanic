import numpy as np
import pandas as pd
import seaborn as sns
train = pd.read_csv('E:\\任务\\titanic\\train.csv')
test = pd.read_csv('E:\\任务\\titanic\\test.csv')
# 一、认识数据
# #查看训练集和测试集有多大
print('训练数据集：',train.shape,'测试数据集：',test.shape)
# 查看一下训练集和测试集都有哪些列
train.columns
test.columns
# PassengerId => 乘客ID
# Pclass => 乘客等级(1/2/3等舱位)
# Name => 乘客姓名
# Sex => 性别
# Age => 年龄
# SibSp => 堂兄弟/妹个数
# Parch => 父母与小孩个数
# Ticket => 船票信息
# Fare => 票价
# Cabin => 客舱
# Embarked => 登船港口

train.Cabin.value_counts()
train_null = train.isnull()
# 处理缺失值
# Age 用所有乘客年龄的平均数去填充
# Cabin 客舱缺失值将近70%，而且有可能很多人是获救上岸后才补充的船舱信息，容易造成偏倚，应该将其删除
# Embarked 有两条缺失值，很少，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，
# 因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C用均值填充缺失值
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna('C')

# #######统计描述########
# 乘客各属性分布
% matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['font.family']='sans-serif'
# 各等级乘客获救情况
Survived_0 = train.Pclass[train.Survived == 0].value_counts() # 未获救
Survived_1 = train.Pclass[train.Survived == 1].value_counts() # 获救
df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
# 各性别获救情况
Survived = train.Sex[train.Survived == 1].value_counts()
unSurvived = train.Sex[train.Survived == 0].value_counts()
df = pd.DataFrame({u'获救':Survived,u'未获救':unSurvived})
df.plot(kind = 'bar', stacked = True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.show()
# 根据船舱等级和性别的获救情况
fig = plt.figure()
plt.title(u'根据舱等级和性别的获救情况')

ax1 = fig.add_subplot(141) # 将图像分为1行4列，从左到右从上到下的第1块
train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts().plot(kind = 'bar', label = 'female high class', color = '#FA2479')
ax1.set_xticklabels([u'获救',u'未获救'], rotation = 0) # 根据实际填写标签
ax1.legend([u'女性/高级舱'], loc = 'best')

ax2 = fig.add_subplot(142, sharey = ax1) # 将图像分为1行4列，从左到右从上到下的第2块
train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(kind = 'bar', label = 'female low class', color = 'pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = fig.add_subplot(143, sharey = ax1)
train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts().plot(kind = 'bar', label = 'male high class', color = 'lightblue')
ax3.set_xticklabels([u'未获救',u'获救'], rotation = 0)
plt.legend([u'男性/高级舱'], loc = 'best')

ax4 = fig.add_subplot(144, sharey = ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(kind = 'bar', label = 'male low class', color = 'steelblue')
ax4.set_xticklabels([u'未获救',u'获救'], rotation = 0)
plt.legend([u'男性/低级舱'], loc = 'bast')
plt.show()
# 各登陆港口乘客的获救情况
fig = plt.figure()
fig.set(alpha = 0.2)
Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title(u'各登陆港口乘客的获救情况')
plt.xlabel(u'登陆港口')
plt.ylabel(u'人数')
plt.show()
# 堂兄弟/妹，孩子/父母有几人，对是否获救的影响
g = train.groupby(['SibSp','Survived']) # 数据分组
df = pd.DataFrame(g.count()['PassengerId'])
print (df)
g = train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)
# 按Cabin有无看获救情况
fig = plt.figure()
fig.set(alpha = 0.2)
Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
df = pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind = 'bar', stacked = True)
plt.title(u'按Cabin有无看获救情况')
plt.xlabel(u'Cabin有无')
plt.ylabel(u'人数')
plt.show()
# 按船票价格分布看获救情况
plt.subplot2grid((2,3),(0,2))
plt.scatter(train.Survived, train.Fare) #为散点图传入数据
plt.ylabel(u"船票价") # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按船票价看获救分布 (1为获救)")
plt.show()

plt.subplot2grid((2,3),(1,0), colspan=2)
train.Fare[train.Survived == 0].plot(kind='kde') # 密度图
train.Fare[train.Survived == 1].plot(kind='kde')
plt.xlabel(u"船票价")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"不同生存结局的乘客船票价分布")
plt.legend((u'1未获救', u'2获救'),loc='best') # sets our legend for our graph.

# 统计分析
# 性别之间的生存率差异分析
train.Sex[train.Survived == 1].value_counts()
train.Sex[train.Survived == 0].value_counts()

from scipy.stats import chi2_contingency
from scipy.stats import chi2
table = [[109,233],[468,81]]
print(table)

stat,p,dof,expected = chi2_contingency(table) # stat卡方统计值，p：P_value，dof 自由度，expected理论频率分布
print('dof=%d'%dof)
print(expected)
prob = 0.95 # 选取95%置信度
critical = chi2.ppf(prob,dof)  # 计算临界阀值
print('probality=%.3f,critical=%.3f,stat=%.3f '%(prob,critical,stat))
# 第一种方法：比较卡方值
if abs(stat)>=critical:
    print('reject H0:Dependent')
else:
    print('fail to reject H0:Independent')
# 第二种方法 比较P值
alpha = 1-prob
print('significance=%.3f,p=%.3f'%(alpha,p))
if p<alpha:
    print('reject H0:Dependent')
else:
    print('fail to reject H0:Independent')
# 乘客等级之间的生存率差异
train.Pclass[train.Survived == 1].value_counts()
train.Pclass[train.Survived == 0].value_counts()
table2 = [[80,136],[97,87],[372,119]]
from scipy.stats import chi2_contingency
stat,p,dof,expected = chi2_contingency(table2)
prob = 0.95
alpha = 1-prob
print('significance=%.3f,p=%.3f'%(alpha,p))
if p<alpha:
    print('reject H0:Dependent')
else:
    print('fail to reject H0:Independent')

# 建立模型
# 因为逻辑回归建模时，需要输入的特征都是数值型特征，我们先对类目型的特征离散/因子化
# train['Cabin'].value_counts()
# 本来是想把Cabin列去掉的，因为它缺失值太多，而且值计数分散，但是想想先把它作为类目（有船舱信息和无船舱信息），加入特征，看看对Survival的分布影响如何
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
train = set_Cabin_type(train)

# 字符串类型-乘客姓名（Name）
# 注意到在乘客名字（Name）中，有一个非常显著的特点：
# 乘客头衔每个名字当中都包含了具体的称谓或者说是头衔，将这部分信息提取出来后可以作为非常有用一个新变量，可以帮助我们进行预测。
# 例如：
# Braund, Mr. Owen Harris
# Heikkinen, Miss. Laina
# Oliva y Ocana, Dona. Fermina
# Peter, Master. Michael J

# 定义函数：从姓名中获取头衔
def getTitle_1(name):
    str1 = name.split(', ')[1] # Mr. Owen Harris    Kelly, Mr. James
    str2 = str1.split('. ')[0] # Mr
    #strip()方法用于移除字符串头尾指定的字符（默认为空格）
    str3 = str2.strip()
    return str3
# 存放提取后的特征
titleDf_1 = pd.DataFrame()
titleDf_1['Title'] = train['Name'].map(getTitle_1)
titleDf_1
# 定义以下几种头衔类别：
# Officer政府官员
# Royalty王室（皇室）
# Mr已婚男士
# Mrs已婚妇女
# Miss年轻未婚女子
# Master有技能的人/教师

#姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
# map函数：对Series每个数据应用自定义的函数计算
titleDf_1['Title'] = titleDf_1['Title'].map(title_mapDict)
# 使用get_dummies进行one-hot编码
titleDf_1 = pd.get_dummies(titleDf_1['Title'],prefix= 'Name')
titleDf_1

# 建立家庭人数和家庭类别
# 家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
# （因为乘客自己也是家庭成员的一个，所以这里加1）
# 家庭类别：
# 小家庭Family_Single：家庭人数=1
# 中等家庭Family_Small: 2<=家庭人数<=4
# 大家庭Family_Large: 家庭人数>=5

# 存放家庭信息
familyDf = pd.DataFrame()
familyDf['FamilySize'] = train['Parch'] + train['SibSp'] + 1
# if 条件为真的的时候返回if前面内容，否则返回0
familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda e:1 if e == 1 else 0)
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda e:1 if 2 <= e <=4 else 0)
familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda e:1 if e >= 5 else 0)
familyDf

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集train，我们使用pandas的get_dummies来完成特征因子化，
# 并拼接在原来的train之上
dummies_Cabin = pd.get_dummies(train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix= 'Pclass')
df_train = pd.concat([train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,titleDf_1,familyDf], axis=1)
df_train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
df_train

#相关性矩阵
corrDf = df_train.corr()
corrDf
'''
查看各个特征与生成情况（Survived）的相关系数，
ascending=False表示按降序排列
'''
corrDf['Survived'].sort_values(ascending = False)

#热力图，查看Survived与其他特征间相关性大小
plt.figure(figsize=(20,20))
sns.heatmap(corrDf.corr(),cmap='BrBG',annot=True,
           linewidths=.5)
plt.xticks(rotation=45)


# 接下来对测试集做一样的操作
# 缺失值处理
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
test = set_Cabin_type(test)

# 处理姓氏
def getTitle_2(name):
    str4 = name.split(', ')[1] # Mr. Owen Harris    Kelly, Mr. James
    str5 = str4.split('. ')[0] # Mr
    #strip()方法用于移除字符串头尾指定的字符（默认为空格）
    str6 = str5.strip()
    return str6
# 存放提取后的特征
titleDf_2 = pd.DataFrame()
titleDf_2['Title'] = test['Name'].map(getTitle_2)
titleDf_2

#姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
# map函数：对Series每个数据应用自定义的函数计算
titleDf_2['Title'] = titleDf_2['Title'].map(title_mapDict)
# 使用get_dummies进行one-hot编码
titleDf_2 = pd.get_dummies(titleDf_2['Title'],prefix= 'Name')
titleDf_2
# 存放家庭信息
familyDf_2 = pd.DataFrame()
familyDf_2['FamilySize'] = test['Parch'] + test['SibSp'] + 1
# if 条件为真的的时候返回if前面内容，否则返回0
familyDf_2['Family_Single'] = familyDf_2['FamilySize'].map(lambda e:1 if e == 1 else 0)
familyDf_2['Family_Small'] = familyDf_2['FamilySize'].map(lambda e:1 if 2 <= e <=4 else 0)
familyDf_2['Family_Large'] = familyDf_2['FamilySize'].map(lambda e:1 if e >= 5 else 0)
familyDf_2
# pd.set_option('display.max_columns',None)   显示全部列
dummies_Cabin = pd.get_dummies(test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix= 'Pclass')
df_test = pd.concat([test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,titleDf_2,familyDf_2], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test

# 构建模型
# 根据各个特征与生成情况（Survived）的相关系数大小，我们选择了这几个特征作为模型的输入：头衔（前面所在的数据集titleDf）、
# 客舱等级（pclassDf）、家庭大小（familyDf）、船票价格（Fare）、船舱号（cabinDf）、登船港口（embarkedDf）、性别（Sex）、Cabin（船舱）
df_train_x = df_train.drop('Survived', axis = 1)
df_train_y = df_train['Survived']
# 导入机器学习模型包
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
# 1.设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)
# 2.汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())

# 3.不同机器学习交叉验证结果汇总
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,df_train_x,df_train_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))
# 4.求出模型得分的均值和标准差
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# 汇总数据
cvResDf = pd.DataFrame({'cv_mean': cv_means,
                        'cv_std': cv_std,
                        'algorithm': ['SVC', 'DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                      'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna']})

cvResDf

# 可视化查看不同算法的表现情况
sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})
cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='cv_mean',ascending=False),sharex=False,
            sharey=False,aspect=2)
cvResFacet.map(sns.barplot,'cv_mean','algorithm',**{'xerr':cv_std},
               palette='muted')
cvResFacet.set(xlim=(0.7,0.9))
cvResFacet.add_legend()

# 5.模型调优
#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(df_train_x,df_train_y)

#LogisticRegression模型
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(df_train_x,df_train_y)

# # 6.查看模型得分
#modelgsGBC模型
print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
#modelgsLR模型
print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)

# 7.查看模型ROC曲线
#求出测试数据模型的预测值
modelgsGBCtestpre_y=modelgsGBC.predict(df_train_x).astype(int)
#画图
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(df_train_y, modelgsGBCtestpre_y) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic GradientBoostingClassifier Model')
plt.legend(loc="lower right")
plt.show()


#求出测试数据模型的预测值
modelgsGBCtestpre_y=modelgsGBC.predict(df_train_x).astype(int)
#画图
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(df_train_y, modelgsGBCtestpre_y) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic GradientBoostingClassifier Model')
plt.legend(loc="lower right")
plt.show()#求出测试数据模型的预测值
modelgsGBCtestpre_y=modelgsGBC.predict(df_train_x).astype(int)


# Logistic Analysis求出测试数据模型的预测值
modelgsLRtestpre_y=modelgsLR.predict(df_train_x).astype(int)
# 画图
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(df_train_y, modelgsLRtestpre_y) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic Logistic Analysis Model')
plt.legend(loc="lower right")
plt.show()
# GradientBoostingClassifier和LogisticRegression模型ROC曲线均左上偏，AUC分别为0.832和0.821，
# 即GradientBoostingClassifier模型效果较好。

# ######构建模型######
train_np = df_train.values
# 获取y
Y = train_np[:, 0]
# 获取自变量x
X = train_np[:, 1:]
#TitanicGBC模型
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(df_train_x,df_train_y)
# 模型准确率
print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
# 对测试数据进行预测
predictions = modelgsGBC.predict(df_test)
# 输出结果
test_ID = df_test['PassengerId']
result = pd.DataFrame({'PassengerId': test_ID, 'Survived': predictions.astype(np.int32)})
result.to_csv('prediction.csv',index=False,sep=',')