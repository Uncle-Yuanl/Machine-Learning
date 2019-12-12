'''
https://www.cnblogs.com/54hys/p/10127055.html
序列平稳性是进行时间序列分析的前提条件,在大数定理和中心定理中要求样本同分布
（这里同分布等价于时间序列中的平稳性），而我们的建模过程中有很多都是建立在大数定理和中心极限定理的前提条件下的
如果它不满足，得到的许多结论都是不可靠的
我们需要对不平稳的序列进行处理将其转换成平稳的序列
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import warnings
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tools.sm_exceptions import HessianInversionWarning

def extend(a, b):
    return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a

# 将字符串索引转换成时间索引
def date_parser(data):
    return pd.datetime.strptime(data, '%Y-%m')

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=HessianInversionWarning)
    # warnings.filterwarnings(action='ignore', category=ValueWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    pd.set_option('display.width', 100)
    np.set_printoptions(linewidth=100, suppress=True)

    data = pd.read_csv('AirPassengers.csv', header=0,
                       parse_dates=['Month'], date_parser=date_parser, index_col=['Month'])
    data.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
    print(data.dtypes)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x = data['Passengers'].astype(np.float)
    x = np.log(x)
    print(x.head(5))

    show = 'diff'   # 'diff', 'ma', 'prime'
    d = 1
    # https://www.cnblogs.com/liulangmao/p/9301032.html
    diff = x - x.shift(periods=d)
    # https://baijiahao.baidu.com/s?id=1622798772654712959&wfr=spider&for=pc
    # 所以会少掉前面一段的数据
    ma = x.rolling(window=12).mean()
    xma = x - ma

    p = 2
    q = 2
    model = ARIMA(endog=x, order=(p, d, q))  # 自回归函数p,差分d,移动平均数q
    arima = model.fit(disp=-1)               # disp<0:不输出过程
    prediction = arima.fittedvalues
    print(type(prediction))                  # <class 'pandas.core.series.Series'>
    y = prediction.cumsum() + x[0]
    mse = ((x - y)**2).mean()
    rmse = np.sqrt(mse)

    plt.figure(facecolor='w')
    if show == 'diff':
        plt.plot(x, 'r-', lw=2, label=u'原始数据')
        plt.plot(diff, 'g-', lw=2, label=u'%d阶差分' % d)
        # plt.plot(prediction, 'r-', lw=2, label=u'预测数据')
        title = u'乘客人数变化曲线 - 取对数'
    elif show == 'ma':
        plt.plot(x, 'r-', lw=2, label=u'原始数据')
        plt.plot(ma, 'g-', lw=2, label=u'滑动平均数据')
        # plt.plot(xma, 'g-', lw=2, label=u'ln原始数据 - ln滑动平均数据')
        # plt.plot(prediction, 'r-', lw=2, label=u'预测数据')
        title = u'滑动平均值与MA预测值'
    else:
        plt.plot(x, 'r-', lw=2, label=u'原始数据')
        plt.plot(y, 'g-', lw=2, label=u'预测数据')
        title = u'对数乘客人数与预测值(AR=%d, d=%d, MA=%d)：RMSE=%.4f' % (p, d, q, rmse)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.title(title, fontsize=18)
    plt.tight_layout(2)
    # plt.savefig('%s.png' % title)
    plt.show()







