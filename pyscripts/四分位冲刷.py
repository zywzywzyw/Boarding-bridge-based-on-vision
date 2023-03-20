# new_nums = list(set(deg)) #剔除重复元素
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共", len(deg), "个\n", deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
'''
# print("中位数:",np.median(deg))
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：", percentile)
# 以下为箱线图的五个特征值
Q1 = percentile[0]  # 上四分位数
Q3 = percentile[2]  # 下四分位数
IQR = Q3 - Q1  # 四分位距
ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if (llim < deg[i] and deg[i] < ulim):
        new_deg.append(deg[i])
print("清洗后数据共", len(new_deg), "个\n", new_deg)