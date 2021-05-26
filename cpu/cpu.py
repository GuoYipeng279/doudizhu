from cards import Cards, Legal
from learn import Model
import random
import numpy as np
from itertools import chain
from progressbar import ProgressBar


class Player(Cards):
    def __init__(self, take, model, situ, auto):
        super(Player, self).__init__(take, auto)
        self.brain = model
        self.parent = situ

    def condition(self, me, en, no, nu):
        '''将当前抽象牌型，己方持牌，敌方持牌翻译成神经网络可接受的格式
        如更换神经网络，需要更改此函数以适应网络输入
        '''
        otp = []
        otp.extend(chain.from_iterable(me))
        otp.extend(chain.from_iterable(en))
        otp.extend(no)
        otp.extend(nu)
        return np.array(otp, dtype=int).reshape(1, -1)

    def judge(self, legal_now=Legal()):
        '''选取最合适的牌型，输出牌型及打分，legal_now为具体牌型'''
        max_point = -99999.0
        otp = Legal()
        if len(legal_now.card_array) == 0:  # 当前无牌型，从所有牌型里选最高的
            for i in self.specific:
                me, en, no, nu = self.get_after(i, self.parent.transfer)
                point = self.brain.model.predict(
                    self.condition(me, en, no, nu))
                if point > max_point:
                    max_point = point
                    otp = i
            return otp, max_point
        now = legal_now.typ
        if (now[0], now[1], now[3]) in self.relation:  # 当前有牌型，查找合法牌型中最好
            for i in self.relation[(now[0], now[1], now[3])]:
                if i.card_array[0] > legal_now.card_array[0]:
                    me, en, no, nu = self.get_after(i, self.parent.transfer)
                    point = self.brain.model.predict(
                        self.condition(me, en, no, nu))
                    if point > max_point:
                        max_point = point
                        otp = i
            return otp, max_point
        return otp, max_point


class Situation:
    '''情景
    参数：maxi最大牌数，auto是否自动构建牌型统计
    duplic：查重方式
    '''
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.current = Legal()
        self.brand_new = self.new_game(kwargs)  # 在取样时标志是否全新

    def new_game(self, kwargs):
        '''随机发牌
        参数：maxi最大牌数，auto是否自动构建牌型统计
        duplic：查重方式
        '''
        maxi = kwargs['maxi'] if 'maxi' in kwargs else 27
        auto = kwargs['auto'] if 'auto' in kwargs else True
        duplic = kwargs['duplic'] if 'duplic' in kwargs else lambda x: False
        lst = list(range(0, 54))
        random.shuffle(lst)
        # menum = ennum = random.randint(1, maxi)
        # if random.randint(0, 1) == 0:
        #     menum = maxi
        # else:
        #     ennum = maxi
        menum = ennum = maxi
        melst = lst[:menum]
        enlst = lst[menum:menum+ennum]
        melst.sort()
        enlst.sort()
        self.melst = Player(melst, self.model, self, auto)
        self.enlst = Player(enlst, self.model, self, auto)
        if not duplic(self):
            self.melst.construct()
            self.enlst.construct()
            return True
        return False

    @property
    def clearzero(self):
        '''清除矩阵中无用的空行，注意要考虑牌型占位'''
        memat, enmat = self.melst, self.enlst
        arr = self.current.arr
        count = np.zeros(9)
        cut = [2, 3, 5]
        groups = []
        lastzero = 2
        lestzero = 2
        for i in range(3, 15):
            for j in range(0, 3):
                if memat.IOD(i, 'det', 3 - j):
                    count[3*j] += 1
                    if count[3*j] == cut[j]:
                        if lastzero != lestzero and i - lastzero < cut[j]:
                            groups.append(lastzero)
                            lestzero = lastzero
                            count = [min(k, i - lastzero) for k in count]
                if enmat.IOD(i, 'det', 3 - j):
                    count[3*j + 1] += 1
                    if count[3*j + 1] == cut[j]:
                        if lastzero != lestzero and i - lastzero < cut[j]:
                            groups.append(lastzero)
                            lestzero = lastzero
                            count = [min(k, i - lastzero) for k in count]
                if arr[i] >= 3 - j:
                    count[3*j + 2] += 1
                    if count[3*j + 2] == cut[j]:
                        if lastzero != lestzero and i - lastzero < cut[j]:
                            groups.append(lastzero)
                            lestzero = lastzero
                            count = [min(k, i - lastzero) for k in count]
            if not memat.IOD(i, 'det') and not enmat.IOD(
              i, 'det') and arr[i] < 3 - j:
                lestzero = lastzero
                lastzero = i
            else:
                groups.append(i)
                for j in range(0, 3):
                    if not memat.IOD(i, 'det', 3 - j):
                        count[3*j] = 0
                    if not enmat.IOD(i, 'det', 3 - j):
                        count[3*j + 1] = 0
                    if arr[i] < 3 - j:
                        count[3*j + 2] = 0
        groups.sort()
        otp = [0, 1, 2]
        otp.extend(groups)
        return otp

    def __hash__(self):
        return hash(tuple([(self.melst.take[i], self.enlst.take[i],
                            self.current.arr[i]) for i in self.clearzero]))

    def transfer(self, *args):
        '''过滤无用行后输出双方矩阵，以及变换后的牌型（3指标抽象，no，nu）'''
        keep = self.clearzero
        me = [list(self.melst.matr[i]) for i in keep]
        en = [list(self.enlst.matr[i]) for i in keep]
        no = [list(self.current.arr//self.current.multiple)[i] for i in keep]
        no.extend([0 for i in range(15-len(no))])
        nu = [0, 0, 0, 0]
        nu[self.current.multiple] = 1
        while len(me) < 15:
            me.append([0, 0, 0, 0])
        while len(en) < 15:
            en.append([0, 0, 0, 0])
        return me, en, no, nu

    def carding(self, start=Legal(), step_max=999):
        '''正常轮流出牌，start为具体牌型，输出胜负结果或预判分数'''
        step = 0
        self.current = start
        point = 0.5
        while not self.enlst.empty and step < step_max:
            self.current, point = self.melst.judge(self.current)
            self.melst.Out(self.current)
            self.melst, self.enlst = self.enlst, self.melst  # 交换出牌方
            step += 1
        if self.enlst.empty:
            return 1.0 if step % 2 == 1 else 0.0
        else:
            return point

    @staticmethod
    def get_into_dict(d, k, v):
        if k in d:
            return True
        d[k] = v
        return False

    @staticmethod
    def sampling(model, sample_size=10, card_num=lambda x: 10):
        H, X, y = dict(), [], []
        pgb = ProgressBar()
        vol = 0
        for i in pgb(range(sample_size)):
            sit = Situation(model, maxi=card_num(random.random()), auto=False,
                            duplic=lambda x: Situation.get_into_dict(H, x, -1))
            if sit.brand_new:
                me, en, no, nu = sit.transfer()
                X.append(sit.melst.condition(me, en, no, nu))
                y.append(sit.carding())
                vol += 1
        print('生成了'+str(vol)+'/'+str(sample_size)+'样本', flush=True)
        return X, y


if __name__ == '__main__':
    model = Model()
    X, y = Situation.sampling(model)
    model.model.fit(X, y)
