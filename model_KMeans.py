import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class KMeans(object):
    def __init__(self, k: int, dim: int, random_center: bool = True, metric: str = 'Euclidean'):
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
        self._initial_center = data.sample(n=self._k).reset_index(drop=True)

    def train(self, data: pd.DataFrame, viewable: bool, epoch: int = 20):
        if epoch <= 0:
            sys.exit('训练轮次至少为1')

        if self._random_center_flag:
            center = self._initial_center
        else:
            self.init_center_from_data(data=data)
            center = self._initial_center
        center.columns = data.columns.values
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
                ccm.append(min_dis)
                cct.append(np.where(dis_array == min_dis)[0][0])
            self.train_log['current_metric'] = ccm
            self.train_log['current_center'] = cct

            if viewable:
                self._plot(data=data, center=center, idx=idx)

            sse = self.train_log["current_metric"].sum().sum()
            if not self.train_log['current_center'].equals(self.train_log['last_center']):
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
        new_center = pd.DataFrame(columns=data.columns.values)
        for c in range(self._k):
            cdf = data.iloc[self.train_log[self.train_log['current_center'] == c].index].mean()
            if cdf.isnull().any():
                cdf = data.sample(n=1).reset_index(drop=True)
            new_center = new_center.append(cdf, ignore_index=True)
        self.fin_center = new_center
        return new_center

    def _plot(self, data: pd.DataFrame, center: pd.DataFrame, idx: int):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        [x_label, y_label] = data.iloc[:, :2].columns.values
        self._color_dic = {}
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
        plt.legend(self._color_dic.keys())
        plt.show()

    def save_res(self, fn: str, idx_df: pd.DataFrame, idx_name: str):
        res_df = pd.concat([idx_df, self.train_log[['current_metric', 'current_center']]], axis=1, join='outer')
        res_df.columns = [idx_name, 'metric', 'cluster']
        res_df.to_csv(fn, index=False)
        print(f'输出文件保存 @ {fn}')


def o_metric(array, center):
    d_s = array - center
    d_l = [((d.T * d).sum()) ** 0.5 for d in d_s]
    return np.array(d_l)


def m_metric(array, center):
    d_s = abs(array - center)
    d_l = [d for d in d_s]
    return np.array(d_l)


def color_set(n):
    color_list = ['#6495ED', '#00EE76', '#8470FF', '#FFF68F',
                  '#006400', '#FFFF00', '#CD5C5C', '#8B6969',
                  '#B22222', '#FF1493', '#8B8989', '#0000FF']
    color_l = np.random.choice(color_list, size=n, replace=False)
    return color_l


