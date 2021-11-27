from copy import deepcopy
from cards import Cards, Legal
from learn import Binary, ModelC
import random
import heapq
import numpy as np
from itertools import chain
from progressbar import ProgressBar
from time import time, strftime
from tensorflow import keras
from tensorflow.keras import metrics
import tensorflow as tf
import math
import os
import sys


def timer(function):
    """
    装饰器函数timer
    :param function:想要计时的函数
    :return:
    """

    def wrapper(*args, **kwargs):
        time_start = time()
        res = function(*args, **kwargs)
        cost_time = time() - time_start
        Player.tot += cost_time
        return res

    return wrapper


class Player(Cards):
    def __init__(self, take, model, situ, auto):
        super(Player, self).__init__(take, auto)
        self.brain = 0
        self.parent = situ

    def condition(self, me, en, no, nu, mode=1):
        '''将当前抽象牌型，己方持牌，敌方持牌翻译成神经网络可接受的格式
        如更换神经网络，需要更改此函数以适应网络输入
        '''
        if mode == 0:
            otp = []
            otp.extend(chain.from_iterable(me))
            otp.extend(chain.from_iterable(en))
            otp.extend(no)
            otp.extend(nu)
            return tuple(otp)
        elif mode == 1:
            hand = [me, en]
            no.extend(nu)
            return np.array([hand, ]), np.array([no, ])

    def get_after(self, legal, f):
        '''在预计出牌后分析情况'''
        super().out(legal)
        self.parent.current = legal
        otp = f()
        super().ret()
        return otp

    def judge(self, legal_now=Legal(), num=1, brain_ind=0):
        '''选取最合适的牌型，输出牌型及打分，legal_now为具体牌型'''
        if brain_ind == 0:
            brain = Trainer.brain
        else:
            brain = Trainer.old_brain
        lst = []  # 分数，具体牌型列表
        if self.parent.enlst.empty:
            return [(0.100, '结束')]
        if len(legal_now.card_array) == 0:  # 当前无牌型，从所有牌型里选最高的
            for i in self.specific:
                hand, curr = self.condition(
                    *self.get_after(i, self.parent.transfer))
                point = brain.model.predict(
                    {'in_hand': hand, 'current': curr})
                # np.array(self.condition(me, en, no, nu)).reshape(1, -1))
                lst.append((float(point), i))
            lst.sort(reverse=True)
            return lst if num == -1 else lst[:min(num, len(lst))]
        noa = Legal()  # 考虑不出情况
        hand, curr = self.condition(*self.get_after(noa, self.parent.transfer))
        point1 = brain.model.predict(
            {'in_hand': hand, 'current': curr})
        # lst.append((float(point), noa))
        now = legal_now.typ
        if (now[0], now[1], now[3]) in self.relation:  # 当前有牌型，查找合法牌型中最好
            for i in self.relation[(now[0], now[1], now[3])]:
                if i.card_array[0] < legal_now.card_array[0]:
                    hand, curr = self.condition(
                        *self.get_after(i, self.parent.transfer))
                    point = brain.model.predict(
                        {'in_hand': hand, 'current': curr})
                    lst.append((float(point), i))
            lst.sort(reverse=True)
            if not lst:
                lst.append((float(point1), noa))
            return lst if num == -1 else lst[:min(num, len(lst))]
        # raise ValueError('Card Error')
        if not lst:
            lst.append((float(point1), noa))
        return lst if num == -1 else lst[:min(num, len(lst))]

    def __str__(self):
        return super().__str__()


def predict(func, at):
    Situation.queue.append(func)


class Situation:
    '''情景
    参数：maxi最大牌数，auto是否自动构建牌型统计
    duplic：查重方式
    '''
    queue = []

    def into(self, lst):
        self.num_situ = len(lst)
        lst.append(self)
        self.next = []

    def __init__(self, model=None, **kwargs):
        self.model = 0
        self.current = Legal()
        self.brand_new = self.new_game(kwargs)  # 在取样时标志是否全新
        self.desc = str(self)
        self.point = 0.5

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return str(hash(self))

    def __str__(self):
        return '我方：' + str(self.melst) + ' 敌方：' + str(self.enlst)

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
        menum = random.randint(1, maxi-1)
        ennum = maxi-menum
        melst = lst[:menum]
        enlst = lst[menum:menum+ennum]
        # melst = [35,10,10]
        # enlst = [7]
        self.meini, self.enini = melst.copy(), enlst.copy()
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
        nu[self.current.multiple-1] = 1
        while len(me) < 15:
            me.append([0, 0, 0, 0])
        while len(en) < 15:
            en.append([0, 0, 0, 0])
        return me, en, no, nu

    @property
    def score(self):
        if self.enlst.empty:
            return 0.100
        hand, curr = self.melst.condition(*self.transfer())
        return Trainer.brain.model.predict(
                        {'in_hand': hand, 'current': curr})[0][0]

    def __lt__(self, other):
        return self.point > other.point

    def exch(self):
        self.melst, self.enlst = self.enlst, self.melst

    def carding1(self, start=Legal(), maxi=20):
        '''正常轮流出牌，start为具体牌型，输出胜负结果或预判分数，堆流程，对手固定'''
        self.current = start
        self.point = self.score
        pq = [self]  # 建堆
        searching = [self]
        points = [-0.900 for i in range(maxi)]
        previous = [None for i in range(maxi)]
        nexts = [[] for i in range(maxi)]
        cur_rec = [[None, None] for i in range(maxi)]
        self.num_situ = 0
        previous[0] = -1
        count = 0
        high = 0.100
        recording = ''
        while pq and count < maxi:
            sitning = heapq.heappop(pq)
            highests = sitning.melst.judge(sitning.current, num=-1)
            code = sitning.num_situ
            for i in highests:
                subsit = deepcopy(sitning)
                subsit.into(searching)
                sub_code = subsit.num_situ
                nexts[code].append(sub_code)
                while sub_code >= len(points):
                    points.append(-0.900)
                    previous.append(None)
                    nexts.append([])
                    cur_rec.append([None, None])
                previous[sub_code] = code
                subsit.point = i[0]
                subsit.current = i[1]
                cur_rec[sub_code][0] = i[1]
                if i[1] == '结束':
                    cur_rec[sub_code][0] = '输'
                    points[sub_code] = 0.100
                    continue
                subsit.melst.Out(i[1])
                points[sub_code] = subsit.point
                heapq.heappush(pq, subsit)

                subsit.exch()
                sub_high = subsit.melst.judge(
                    subsit.current, num=1, brain_ind=1)[0]
                if sub_high[1] == '结束':
                    cur_rec[sub_code][1] = '赢'
                    points[sub_code] = 0.900
                    subsit.exch()
                    pq = None
                    break
                subsit.melst.Out(sub_high[1])
                subsit.current = sub_high[1]
                cur_rec[sub_code][1] = sub_high[1]
                subsit.exch()
            count += len(highests)
            if count >= maxi:
                break
        champ = -1
        for i, j in enumerate(points):
            if j >= 0.100 and not nexts[i] and high <= j:
                champ = i
                high = j
        seq = []
        while previous[champ] >= 0:
            seq.append(champ)
            champ = previous[champ]
        seq.reverse()
        for i in seq:
            recording += str(cur_rec[i][0]) + ' '
            recording += str(cur_rec[i][1]) + ' '
        return high, recording, cur_rec[seq[0]][0]
    
    @staticmethod
    def carding2(selfs, start=[Legal()], maxi=20):
        '''正常轮流出牌，start为具体牌型，输出胜负结果或预判分数，堆流程，对手固定'''
        pq = []
        searching = []
        points = []
        previous = []
        nexts = []
        cur_rec = []
        count = []
        high = []
        recording = []
        for i in range(len(selfs)):
            selfs[i].current = start
            selfs[i].point = selfs[i].score
            pq.append([selfs[i]])  # 建堆
            searching.append([selfs[i]])
            points.append([-0.900 for i in range(maxi)])
            previous.append([None for i in range(maxi)])
            nexts.append([[] for i in range(maxi)])
            cur_rec.append([[None, None] for i in range(maxi)])
            selfs[i].num_situ.append(0)
            previous[i][0] = -1
            count.append(0)
            high.append(0.100)
            recording.append('')
        while pq and count < maxi:
            sitning = heapq.heappop(pq)
            highests = sitning.melst.judge(sitning.current, num=-1)
            code = sitning.num_situ
            for i in highests:
                subsit = deepcopy(sitning)
                subsit.into(searching)
                sub_code = subsit.num_situ
                nexts[code].append(sub_code)
                while sub_code >= len(points):
                    points.append(-0.900)
                    previous.append(None)
                    nexts.append([])
                    cur_rec.append([None, None])
                previous[sub_code] = code
                subsit.point = i[0]
                subsit.current = i[1]
                cur_rec[sub_code][0] = i[1]
                if i[1] == '结束':
                    cur_rec[sub_code][0] = '输'
                    points[sub_code] = 0.100
                    continue
                subsit.melst.Out(i[1])
                points[sub_code] = subsit.point
                heapq.heappush(pq, subsit)

                subsit.exch()
                sub_high = subsit.melst.judge(
                    subsit.current, num=1, brain_ind=1)[0]
                if sub_high[1] == '结束':
                    cur_rec[sub_code][1] = '赢'
                    points[sub_code] = 0.900
                    subsit.exch()
                    pq = None
                    break
                subsit.melst.Out(sub_high[1])
                subsit.current = sub_high[1]
                cur_rec[sub_code][1] = sub_high[1]
                subsit.exch()
            count += len(highests)
            if count >= maxi:
                break
        champ = -1
        for i, j in enumerate(points):
            if j >= 0.100 and not nexts[i] and high <= j:
                champ = i
                high = j
        seq = []
        while previous[champ] >= 0:
            seq.append(champ)
            champ = previous[champ]
        seq.reverse()
        for i in seq:
            recording += str(cur_rec[i][0]) + ' '
            recording += str(cur_rec[i][1]) + ' '
        return high, recording, cur_rec[seq[0]][0]

    @staticmethod
    def collect_sample(node, samples=[], trans=1):
        max_score = -999
        for i in node:
            if type(i) == float:
                return i if trans == 1 else 1-i
            else:
                score = Situation.collect_sample(node[i], samples, -trans)
                if trans > 0:
                    samples.append((i, score))
                max_score = max(max_score, score)
        return max_score

    @staticmethod
    def get_into_dict(d, k, v):
        if hash(k) in d:
            return True
        d[hash(k)] = v
        return False


class Trainer:
    brain = None
    old_brain = None

    @staticmethod
    def round_acc(y_true, y_pred):
        return metrics.binary_crossentropy(tf.round(y_true), tf.round(y_pred))

    def __init__(self, impo=None):
        if impo is None:
            self.model = ModelC(loss="mse", metric=Trainer.round_acc)
        else:
            self.model = ModelC(metric=Trainer.round_acc, modeling=impo)
        Trainer.old_brain = self.model
        Trainer.brain = self.model
        self.birth = strftime("%Y-%m-%d_%H-%M-%S")

    def sampling(self, model, sample_size=500, vali_size=0.1,
                 card_num=lambda x: 5*x, depth=20):
        H, X, y = dict(), [], []
        pgb = ProgressBar()
        vol, tot = 0, 0
        for i in pgb(range(sample_size)):
            sit = Situation(model, maxi=card_num(), auto=False,
                            duplic=lambda x: Situation.get_into_dict(H, x, -1))
            pos = hash(sit)
            if sit.brand_new:
                # sample_tree = dict()
                # recording = sit.carding(tree=[6,1,2,1,1,1], high=sample_tree)  # noqa E231
                samples = [None, None]
                sit.exch()  # 交换敌我，因为实际上是测算敌方选择不出时我方败率
                main = sit.melst.condition(*sit.transfer())
                sit.exch()
                score, recording, optim = sit.carding1(maxi=depth)
                # score = Situation.collect_sample(sample_tree, samples)
                samples[0] = (main, 1.000 - score)  # 取反值，因为此为敌方视角
                sit.melst.Out(optim)
                sit.current = optim
                sit.exch()
                dual = sit.melst.condition(*sit.transfer())
                dual_sc = score
                samples[1] = (dual, dual_sc)
                H[pos] = []
                for i in samples:
                    X.append(i[0])
                    y.append(i[1])
                    H[pos].append((i[1], sit.desc, i[0], recording))
                    tot += 1
                vol += 1
            else:
                for i in H[pos]:
                    X.append(i[2])
                    y.append(i[0])
                    tot += 1
        vsize = math.floor(tot*vali_size)
        os.system("tada.wav")
        to_write = []
        for i in H:
            sample_y = int(H[i][0][0]*100)
            predic_y = int(model.model.predict(H[i][0][2])[0][0]*100)
            init_pat = H[i][0][1]
            proc_pat = H[i][0][3]
            fixing = 'OK'
            if (sample_y - 50)*(predic_y - 50) <= 0:
                fixing = 'FX'
            if sample_y < 10:
                sample_y = "0" + str(sample_y)
            if predic_y < 10:
                predic_y = "0" + str(predic_y)
            inn = [fixing, str(sample_y), str(predic_y), init_pat, '流程：', proc_pat]  # noqa E128
            to_write.append(' '.join(inn))
            print(inn)
        to_write.sort()
        to_write = '\n'.join(to_write)
        if not os.path.exists(f'{sys.path[0]}\\{self.birth}'):
            os.makedirs(f'{sys.path[0]}\\{self.birth}')
        log_file = open(f'{sys.path[0]}\\{self.birth}\\{strftime("%Y-%m-%d_%H-%M-%S")}_log.txt', 'w')  # noqa E128
        log_file.write(to_write)
        log_file.close()
        print('生成了'+str(vol)+'/'+str(sample_size)+'/'+str(tot)+'样本')
        return np.array([x[0] for x in X[:vsize]]).reshape(vsize, 2, 15, 4),\
            np.array([x[1] for x in X[:vsize]]).reshape(vsize, 19),\
            np.array(y[:vsize]).reshape(vsize, 1),\
            np.array([x[0] for x in X[vsize:]]).reshape(tot-vsize, 2, 15, 4),\
            np.array([x[1] for x in X[vsize:]]).reshape(tot-vsize, 19),\
            np.array(y[vsize:]).reshape(tot-vsize, 1)

    def train(self):
        start_card = 2
        ssize = 100
        depth = 20
        while start_card < 20:
            vX1, vX2, vy, X1, X2, y = self.sampling(self.model,
                sample_size=ssize, card_num=lambda: start_card, depth=depth)  # random.randint(2, start_card))  # noqa E128
            # vy = tf.constant(vy)
            metric = Binary(({'in_hand': vX1, 'current': vX2}, vy))
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tb_dir', histogram_freq=1)
            self.model.model.fit({'in_hand': X1, 'current': X2}, y, epochs=40,
                                 validation_split=0.2,
                                 callbacks=[metric, tensorboard],
                                 verbose=0)
            if metric.val_precisions[-1] > .95 and\
                    metric.val_recalls[-1] > .95:
                start_card += 1
                print('提升张数为：', start_card)
                sav_name = f'{sys.path[0]}\\{self.birth}\\{strftime("%Y-%m-%d_%H-%M-%S")}_model.h5'  # noqa E501
                self.model.model.save(sav_name)
                Trainer.old_brain = Trainer.brain
                Trainer.brain = ModelC(metric=Trainer.round_acc,
                                       modeling=sav_name)
            else:
                ssize *= 2
                depth *= 2
                print('提升样本量：', ssize)
                print('提升深度：', depth)


if __name__ == '__main__':
    tr = Trainer()
    tr.train()
