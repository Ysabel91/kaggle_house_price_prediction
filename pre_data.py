# coding: utf-8
"""
data preparation for model-based task:
"""

##==================== Package ====================##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
from scipy import stats
import pickle  # to store temporary variable

##==================== File-Path (fp) ====================##
## raw data (for read)
fp_train = "./data/train.csv"
fp_test = "./data/test.csv"

## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
fp_train_f = "./data/train_f.csv"
fp_test_f = "./data/test_f.csv"


##==================== pre-Processing ====================##

## data reading
df_train_ini = pd.read_csv(fp_train, nrows=10)
df_train_org = pd.read_csv(fp_test)
df_test_org = pd.read_csv(fp_test)

# 检查缺失值情况
na_count = df_train_org.isnull().sum().sort_values(ascending=False)
na_rate = na_count/len(df_train_org)
na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'rate'])
print (na_data)

# 根据特征删除或是补充处理
df_train_org = df_train_org.drop(na_data[na_data["count"]>1].index, axis=1)
df_train_org = df_train_org.drop(df_train_org.loc[df_train_org['Electrical'].isnull()].index)
print (df_train_org.shape)

# 根据特征类型
df_X = df_train_org.drop('SalePrice', axis=1)
df_Y = df_train_org['SalePrice']
quantity = [attr for attr in df_X.columns if df_X.dtypes[attr] != 'object']
quality = [attr for attr in df_X.columns if df_X.dtypes[attr] == 'object']

# 类型变量缺失值补全
for c in quality:
    df_train_org[c] = df_train_org[c].astype('category')
    if df_train_org[c].isnull().any():
        df_train_org[c] = df_train_org[c].cat.add_categories(['MISSING'])
        df_train_org[c] = df_train_org[c].fillna('MISSING')

# 连续变量缺失值补全
quantity_miss_cal = df_train_org[quantity].isnull().sum().sort_values(ascending=False)  # 缺失量均在总数据量的10%以下
missing_cols = quantity_miss_cal[quantity_miss_cal>0].index
df_train_org[missing_cols] = df_train_org[missing_cols].fillna(0.)  # 从这些变量的意义来看，缺失值很可能是取 0
print (df_train_org[missing_cols].isnull().sum())  # 验证缺失值是否都已补全


# 一元方差分析（类型变量）
def anova(frame, qualitative):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)  # 某特征下不同取值对应的房价组合形成二维列表
        pval = stats.f_oneway(*samples)[1]  # 一元方差分析得到 F，P，要的是 P，P越小，对方差的影响越大。
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

# a = anova(df_train_org, quality)
# a['disparity'] = np.log(1./a['pval'].values)  # 悬殊度
# fig, ax = plt.subplots(figsize=(16,8))
# sns.barplot(data=a, x='feature', y='disparity')
# x=plt.xticks(rotation=90)
# plt.show()

# # 给房价分段，并由此查看各段房价内那些特征的取值会出现悬殊
# poor = df_train_org[df_train_org['SalePrice'] < 200000][quantity].mean()
# pricey = df_train_org[df_train_org['SalePrice'] >= 200000][quantity].mean()
# diff = pd.DataFrame()
# diff['attr'] = quantity
# diff['difference'] = ((pricey-poor)/poor).values
# plt.figure(figsize=(10,4))
# sns.barplot(data=diff, x='attr', y='difference')
# plt.xticks(rotation=90)
# plt.show()


# 查看数据分布
output,var,var1 = 'SalePrice', 'GrLivArea', 'TotalBsmtSF'
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
df_train_org.plot.scatter(x=var,y=output,ylim=(0,800000),ax=axes[0])
df_train_org.plot.scatter(x=var1,y=output,ylim=(0,800000),ax=axes[1])

print (df_train_org.sort_values(by = 'GrLivArea', ascending = False)[:2])  # 查找离群点

# 删除离群点
df_train_org = df_train_org.drop(df_train_org[df_train_org['Id'] == 1298].index)
df_train_org = df_train_org.drop(df_train_org[df_train_org['Id'] == 523].index)
# fig = plt.figure(figsize=(12,5))
# plt.subplot(121)
# sns.distplot(df_train_org[output])
# plt.subplot(122)
# res = stats.probplot(df_train_org[output], plot=plt)
# plt.show()

df_train_org['HasBasement'] = df_train_org['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_org['HasGarage'] = df_train_org['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_org['Has2ndFloor'] = df_train_org['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
# df_train_org['HasMasVnr'] = df_train_org['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_org['HasWoodDeck'] = df_train_org['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_org['HasPorch'] = df_train_org['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_org['HasPool'] = df_train_org['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_org['IsNew'] = df_train_org['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)
boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr',
           'HasWoodDeck', 'HasPorch', 'HasPool', 'IsNew']

def quadratic(feature):
    df_train_org[feature] = df_train_org[feature[:-1]]**2

qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
        '2ndFlrSF2', 'GrLivArea2']

for feature in qdr:
    quadratic(feature)

df_train_org = pd.get_dummies(df_train_org)
print (df_train_org.shape)
df_train_org.to_csv(fp_train_f)



print(' - finish - ')