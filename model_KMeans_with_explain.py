import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def o_metric(array, center):
    """
    计算两数组之间的 欧氏 距离
    :param array: np.ndarray -->  X: 1*col
    :param center: np.ndarray --> Y: {dim}*col
    :return: Euclidean Metric of X and Y
    """
    d_s = array - center
    d_l = [((d.T * d).sum()) ** 0.5 for d in d_s]
    return np.array(d_l)


def m_metric(array, center):
    """
    计算两数组之间的 曼哈顿 距离
    :param array: np.ndarray -->  X
    :param center: np.ndarray --> Y
    :return: Manhattan Distance of X and Y
    """
    d_s = abs(array - center)
    d_l = [d for d in d_s]
    return np.array(d_l)


def color_set(n):
    color_list = ['#6495ED', '#00EE76', '#8470FF', '#FFF68F',
                  '#006400', '#FFFF00', '#CD5C5C', '#8B6969',
                  '#B22222', '#FF1493', '#8B8989', '#0000FF']
    color_l = np.random.choice(color_list, size=n, replace=False)
    return color_l


class KMeans(object):
    def __init__(self, k: int, dim: int, random_center: bool = True, metric: str = 'Euclidean'):
        """
        初始化 Object K-means
        :param k: 要聚多少类
        :param dim: 要训练的数据多少维（多少列）
        :param random_center: 初始化聚类中心时，是否使用随机中心 # True-->使用随机中心，False-->从训练样本中随机选取中心
        :param metric: 计算个样本点到聚类中心的距离方法 # Available candidate: Euclidean & Manhattan
        :var self.train_log: 用于储存(In Memory)训练记录
        :var self.cs: 若要画图，则储存各类别所对应的颜色
        :var self.fin_center: 聚类算法结束时，最终聚类中心坐标（dim维）
        """
        self._k = k
        self._dim = dim
        self._metric = metric
        self._random_center_flag = random_center
        if random_center:
            self._initial_center = pd.DataFrame({f'{i}': np.random.randn(dim) for i in range(k)}).T
        self.cs = color_set(self._k)
        self.train_log = pd.DataFrame()
        self.fin_center = pd.DataFrame()

    def init_center_from_data(self, data):
        """从训练样本中随机选取中心。(When self._random_center_flag = False)"""
        self._initial_center = data.sample(n=self._k).reset_index(drop=True)
        # 还可以让各质心离得最远开始 --> 各列取到 max & min , 并用其进行初始化。
        pass

    def train(self, data: pd.DataFrame, viewable: bool, epoch: int = 20):
        """
        训练 {epoch} 轮 {data}
        :param data: 需要训练的样本
        :param viewable: 是否使用plt进行可视化操作
        # :param save: 保存训练分类图片 (Using Only when viewable is True.)
        :param epoch: 要训练多少轮
        """
        if epoch <= 0:
            sys.exit('训练轮次至少为1')

        # 初始化 训练用的质心
        if self._random_center_flag:
            center = self._initial_center
        else:
            self.init_center_from_data(data=data)
            center = self._initial_center
        center.columns = data.columns.values
        # 初始化 和 data 一样多行的 训练记录
        self.train_log = pd.DataFrame(columns=['current_metric', 'last_metric', 'current_center', 'last_center'],
                                      data=[[np.inf, np.inf, -1, -1]] * data.shape[0], index=data.index.values)
        train_flag = True
        idx = 0
        data_s = data.to_numpy()
        while train_flag:
            idx += 1
            ccm, cct = [], []
            center_s = center.to_numpy()
            for cd in tqdm(data_s):
                if self._metric == 'Euclidean':
                    dis_array = o_metric(cd, center_s)
                elif self._metric == 'Manhattan':
                    dis_array = m_metric(cd, center_s)
                else:
                    sys.exit(f'暂无该距离算法\n{self._metric}')
                min_dis = min(dis_array)
                # 使用list添加，比直接用DataFrame.iloc赋值更快
                ccm.append(min_dis)
                cct.append(np.where(dis_array == min_dis)[0][0])
            self.train_log['current_metric'] = ccm
            self.train_log['current_center'] = cct

            # 传递中心及数据并绘制二维图
            if viewable:
                self._plot(data=data, center=center, idx=idx)

            # 操作完成后，更新center
            sse = self.train_log["current_metric"].sum().sum()
            if not self.train_log['current_center'].equals(self.train_log['last_center']):
                # 若本次SSE大于上次SSE，则停止训练，否则继续更新center
                if sse > self.train_log["last_metric"].sum().sum():
                    train_flag = False
                    print(f'第{idx}次迭代后，本次SSE: {sse:.3f} 大于上次迭代SSE，训练结束。\n', center)
                else:
                    center = self.update(data=data)
                    self.train_log['last_center'] = self.train_log['current_center']
                    self.train_log['last_metric'] = self.train_log['current_metric']
                    print(f'第{idx}次迭代后，质心：\n', center, f'\nSSE: {sse:.6f}')
            else:
                train_flag = False
                print(f'第{idx}次迭代，分类标签未变化，训练结束。\n', center, f'\nSSE: {sse:.3f}')

            if idx == epoch:
                train_flag = False
                print(f'第{idx}次迭代，达到指定次数，训练结束。\n', center, f'SSE: {sse:.3f}')

    def update(self, data):
        """更新聚类中心"""
        new_center = pd.DataFrame(columns=data.columns.values)
        for c in range(self._k):
            cdf = data.iloc[self.train_log[self.train_log['current_center'] == c].index].mean()
            # 若聚类中心某个dim没有任一值，则从 data 重新选取
            if cdf.isnull().any():
                cdf = data.sample(n=1).reset_index(drop=True)
            new_center = new_center.append(cdf, ignore_index=True)
        self.fin_center = new_center
        return new_center

    def _plot(self, data: pd.DataFrame, center: pd.DataFrame, idx: int):
        """
        可视化前两维数据
        TBD: 也可以是指定列
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        [x_label, y_label] = data.iloc[:, :2].columns.values
        self._color_dic = {}
        # plt.scatter(data[x_label].tolist(), data[y_label].tolist(),
        #             s=2, c=self._train_log['current_center'])
        for c in range(self._k):
            cdf = data.iloc[self.train_log[self.train_log['current_center'] == c].index.values]
            cdf_color = self.cs[c]
            self._color_dic[f'{c}'] = cdf_color
            plt.scatter(cdf[x_label].tolist(), cdf[y_label].tolist(),
                        s=5, c=cdf_color)
            plt.scatter(center.iloc[:, 0].tolist(), center.iloc[:, 1].tolist(),
                        s=35, c='#000000', marker='x')
        plt.title(f'第{idx}次迭代后，数据前两维可视化')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(self._color_dic.keys())
        plt.show()

    def save_res(self, fn: str, idx_df: pd.DataFrame, idx_name: str):
        res_df = pd.concat([idx_df, self.train_log[['current_metric', 'current_center']]], axis=1, join='outer')
        res_df.columns = [idx_name, 'metric', 'cluster']
        res_df.to_csv(fn, index=False)
        print(f'输出文件保存 @ {fn}')
