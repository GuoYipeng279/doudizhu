from copy import deepcopy
from cards import Cards, Legal, Card
from learn import Binary, ModelL
import random
import heapq
import numpy as np
from progressbar import ProgressBar
from time import strftime, sleep, time
from tensorflow import keras
from tensorflow.keras import metrics
import tensorflow as tf
import math
import os
import sys
from threading import Thread, Lock

# 模型预测的是：输入先手持牌，后手持牌，先手方面对牌，给出先手方胜率，10输90赢


def thread_counter(function):
    def wrapper(*args, **kwargs):
        with Trainer.thread_lock:
            Trainer.tot_thread += 1
        res = function(*args, **kwargs)
        with Trainer.thread_lock:
            Trainer.tot_thread -= 1
        return res
    return wrapper


def thread_show():
    print('Threads:')
    print('0', end='')
    maxi = 0
    Trainer.need_counter = True
    while Trainer.need_counter:
        with Trainer.thread_lock:
            tore = Trainer.tot_thread
        maxi = max(tore, maxi)
        print(''.join(['\r', str(Trainer.til_now), '/',
                       str(2*Trainer.depth), ',', str(tore),
                       '/', str(maxi), ',', str(Trainer.bay),
                       'I' if Trainer.implem else '', '     ',
                       ]), end='', )
        sleep(0.5)


def timer(function):
    def wrapper(*args, **kwargs):
        st = time()
        res = function(*args, **kwargs)
        with Trainer.thread_lock:
            name = function.__name__
            if name in Trainer.functional:
                Trainer.functional[name] += time()-st
            else:
                Trainer.functional[name] = time()-st
        return res
    return wrapper


class Player(Cards):
    def __init__(self, take, model, situ, auto):
        super(Player, self).__init__(take, auto)
        self.brain = 0
        self.parent = situ

    @timer
    def condition(self, me, en, no, nu, mode=2, not_reverse=False):
        '''将当前抽象牌型，己方持牌，敌方持牌翻译成神经网络可接受的格式
        如更换神经网络，需要更改此函数以适应网络输入
        '''
        if mode == 0:
            pass
            # otp = []
            # otp.extend(chain.from_iterable(me))
            # otp.extend(chain.from_iterable(en))
            # otp.extend(no)
            # otp.extend(nu)
            # return tuple(otp)
        elif mode == 1:
            hand = [en, me]
            if not_reverse:
                hand = [me, en]
            no.extend(nu)
            return np.array([hand, ]), np.array([no, ])
        elif mode == 2:
            no.extend(nu)
            en.reverse()
            me.reverse()
            en0 = np.array(en)
            me0 = np.array(me)
            hand = np.hstack([en0, me0])
            if not_reverse:
                hand = np.hstack([me0, en0])
            return hand, np.array(no)

    @timer
    def get_after(self, legal, f):
        '''在预计出牌后分析情况'''
        super().out(legal)
        self.parent.current = legal
        otp = f()
        super().ret()
        return otp

    @thread_counter
    @timer
    def judge(self, brain, legal_now=Legal(), num=1, im_at=-1,
              leader=[], target=-1, output=None, position=-1, aux=None):
        '''
        选取最合适的牌型，输出牌型及打分，legal_now为当前己方面临具体牌型
        输出的是牌型以及敌方视角下的敌方胜率（敌，我，敌面临）
        选取前num名，此为第im_at轮，领路人为leader（领路对接吹号人）
        领路人计划收取target个完成信号，若0则非领路人
        '''
        with Trainer.locker:
            Trainer.bay += 1
        lst = []  # 分数，具体牌型列表
        serves = []  # 总执行过后的抓取位置
        possibles = []
        if aux is None:
            aux = dict()
        over = False

        def no_rep(h, c, i=None):
            samp = hash(str(h) + str(c))
            with Trainer.locker:
                if samp in aux:
                    tore = True
                else:
                    aux[samp] = (h, c, i)
                    tore = False
            return tore
        if self.parent.enlst.empty:
            over = True
        elif len(legal_now.card_array) == 0:  # 当前无牌型，从所有牌型里选最高的
            for i in self.specific:
                # 得到出牌之后的敌我与当前牌型（敌方，敌方的敌方（我方），敌方面对牌型）
                # condition函数已经调换敌我
                hand, curr = self.condition(
                    *self.get_after(i, self.parent.transfer))
                if no_rep(hand, curr, i):
                    hand[0][3] = -1
                possibles.append(i)
                with Trainer.locker:
                    serves.append(len(Trainer.to_predict[im_at]))
                    Trainer.to_predict[im_at].append(
                        {'in_hand': hand, 'current': curr})
        else:
            now = legal_now.typ
            if (now[0], now[1], now[3]) in self.relation:  # 当前有牌型，查找合法牌型中最好
                for i in self.relation[(now[0], now[1], now[3])]:
                    if i.card_array[0] < legal_now.card_array[0]:
                        hand, curr = self.condition(
                            *self.get_after(i, self.parent.transfer))
                        if no_rep(hand, curr, i):
                            hand[0][3] = -1
                        possibles.append(i)
                        with Trainer.locker:
                            serves.append(len(Trainer.to_predict[im_at]))
                            Trainer.to_predict[im_at].append(
                                {'in_hand': hand, 'current': curr})
            if not possibles:
                noa = Legal()  # 考虑不出情况
                possibles.append(noa)
                hand, curr = self.condition(
                    *self.get_after(noa, self.parent.transfer))
                no_rep(hand, curr)
                with Trainer.locker:
                    serves.append(len(Trainer.to_predict[im_at]))
                    Trainer.to_predict[im_at].append(
                        {'in_hand': hand, 'current': curr})
        with Trainer.locker:  # 注意不要死锁while
            leader.append(92)
            Trainer.bay -= 1
        # 如果是领路人并且完成未集齐，在此卡主
        while target >= 0 and len(leader) < target:
            sleep(0.1)
        with Trainer.locker:
            # （非）领路人至此，领路人向吹号人提交
            if target >= 0:
                Trainer.counter[im_at] += 1
        while Trainer.counter[im_at] < Trainer.eff_size + sum(Trainer.fin_size[:im_at]):  # noqa E501
            sleep(1)
        with Trainer.locker:
            Trainer.bay += 1
            Trainer.implement(im_at, brain)  # 吹号人
        if over:
            tore = [(0.900, '结束')]  # 己方发现敌方已无牌，最优只有认输，敌方胜率90%
            if output is not None:
                with Trainer.locker:
                    output[position] = tore
            with Trainer.locker:
                Trainer.bay -= 1
            return tore
        for a, i in enumerate(possibles):
            # lst.append((0.1, i))
            lst.append((float(Trainer.to_predict[im_at][serves[a]]), i))
        lst.sort(reverse=False)
        tore = lst if num == -1 else lst[:min(num, len(lst))]
        if output is not None:
            with Trainer.locker:
                output[position] = tore
        with Trainer.locker:
            Trainer.bay -= 1
        return tore

    def __str__(self):
        return super().__str__()


class Situation:
    '''情景
    参数：maxi最大牌数，auto是否自动构建牌型统计
    duplic：查重方式
    '''
    @timer
    def into(self, lst):
        self.num_situ = len(lst)
        lst.append(self)
        self.next = []

    def __init__(self, **kwargs):
        self.model = 0
        self.current = Legal()
        self.hash_value = None
        self.brand_new = self.new_game(kwargs)  # 在取样时标志是否全新
        self.desc = str(self)
        self.point = 0.5

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return str(hash(self))

    def __str__(self):
        return '我方：' + str(self.melst) + ' 敌方：' + str(self.enlst)

    @timer
    def new_game(self, kwargs):
        '''随机发牌
        参数：maxi最大牌数，auto是否自动构建牌型统计
        duplic：查重方式
        '''
        maxi = kwargs['maxi'] if 'maxi' in kwargs else 27
        auto = kwargs['auto'] if 'auto' in kwargs else True
        duplic = kwargs['duplic'] if 'duplic' in kwargs else lambda x: False
        records = kwargs['records'] if 'records' in kwargs else None
        lst = list(range(0, 54))
        # lst.reverse()
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
        if records is not None:
            records.append((melst, enlst))


        # melst = ['7', '3']
        # enlst = ['小王', '2', 'K', '5', '5']
        # self.current = Legal()
        # melst = [Card.to_int(i) for i in melst]
        # enlst = [Card.to_int(i) for i in enlst]

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
        '''清除矩阵中无用的空行，注意要考虑牌型占位与敌方牌的关系'''
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
              i, 'det') and arr[i] < 1:
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
        if self.melst.take[1] == 0 and self.enlst.take[1] == 0 and arr[1] == 0:
            otp = [1, 0, 2]  # 没有小王，大王下降
        if self.melst.take[2] == 0 and self.enlst.take[2] == 0 and arr[2] == 0:
            # 这里·应该考虑王炸情况
            otp = [2, otp[0], otp[1]]  # 没有2，王下降
        otp.extend(groups)
        return otp

    @property
    def clearzero1(self):
        '''清除矩阵中无用的空行，注意要考虑牌型占位与敌方牌的关系'''
        memat, enmat = self.melst, self.enlst
        arr = list(self.current.arr)
        if max(arr) > 0:
            t = self.current.typ
            abstr = (t[0], t[1], t[3])
            one_up = None
            if abstr in memat.relation:
                able = memat.relation[abstr]
                for i in able:
                    if i < self.current and (one_up is None or one_up < i):
                        one_up = i
            if one_up is not None:
                displace = t[2] - one_up.typ[2] - 1
            elif t[2] > 3:
                displace = t[2] - 3
            else:
                displace = 0
            arr = arr[displace:]
            arr.extend([0 for i in range(displace)])
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
            if not memat.IOD(i, 'det') and not enmat.IOD(i, 'det') and arr[i] < 1:
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
        if memat.take[1] == 0 and enmat.take[1] == 0 and arr[1] == 0:
            otp = [1, 0, 2]  # 没有小王，大王下降
        if memat.take[2] == 0 and enmat.take[2] == 0 and arr[2] == 0:
            # 这里·应该考虑王炸情况
            otp = [2, otp[0], otp[1]]  # 没有2，王下降
        otp.extend(groups)
        return otp, np.array(arr)

    def __hash__(self):
        if self.hash_value is None:
            keep, arr = self.clearzero1
            useful = [(self.melst.take[i], self.enlst.take[i],
                       arr[i]) for i in keep]
            self.hash_value = hash(tuple(useful))
        return self.hash_value

    @timer
    def transfer(self, *args):
        '''过滤无用行后输出双方矩阵，以及变换后的牌型（3指标抽象，no，nu）'''
        keep, arr = self.clearzero1  # , self.current.arr
        me = [list(self.melst.matr[i]) for i in keep]
        en = [list(self.enlst.matr[i]) for i in keep]
        no = [list(arr//self.current.multiple)[i] for i in keep]
        no.extend([0 for i in range(15-len(no))])
        nu = [0, 0, 0, 0]
        nu[self.current.multiple-1] = 1
        while len(me) < 15:
            me.append([0, 0, 0, 0])
        while len(en) < 15:
            en.append([0, 0, 0, 0])
        return me, en, no, nu

    @thread_counter
    @timer
    def score(self, brain, im_at):
        pass
        # with Trainer.locker:
        #     Trainer.bay += 1
        # tore = None
        # if self.enlst.empty:
        #     tore = 0.100
        # else:
        #     serve = None
        #     hand, curr = self.melst.condition(*self.transfer())
        #     with Trainer.locker:
        #         serve = len(Trainer.to_predict[im_at])
        #         Trainer.to_predict[im_at].append(
        #             {'in_hand': hand, 'current': curr})
        # with Trainer.locker:
        #     Trainer.counter[im_at] += 1
        #     Trainer.bay -= 1
        # while Trainer.counter[im_at] < Trainer.eff_size:
        #     sleep(1)
        # with Trainer.locker:
        #     Trainer.bay += 1
        #     Trainer.implement(im_at, brain)
        # if tore is None:
        #     with Trainer.locker:
        #         tore = Trainer.to_predict[im_at][serve]
        # with Trainer.locker:
        #     Trainer.bay -= 1
        # return tore

    def __lt__(self, other):
        return self.point > other.point

    def exch(self):
        self.melst, self.enlst = self.enlst, self.melst

    @thread_counter
    @timer
    def carding1(self, start=Legal(), maxi=20,
                 output=None, position=-1, output2=None):
        '''正常轮流出牌，start为具体牌型，输出胜负结果或预判分数，堆流程，对手固定，
        出牌流程为永远选取对敌方最不利的出牌方式，
        然后交由敌方，选择对己方最不利的出牌方式，来决定此形势己方分数，
        分数为己方'''
        self.current = start
        im_at = 0
        self.point = -100  # self.score(Trainer.brain, im_at)
        im_at += 1
        pq = [self]  # 建堆，大根堆，根为己方胜率最高的打法
        searching = [self]
        no_rep_m, no_rep_n = dict(), dict()
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
            # 所有己方出牌的打分（所有牌型之后，敌方胜率（judge特性））
            highests = sitning.melst.judge(
                Trainer.brain, sitning.current,
                num=-1, im_at=im_at, leader=[], target=1, aux=no_rep_m)
            im_at += 1
            code = sitning.num_situ
            sub_pool = []
            pro_pool = []
            submit_box = []  # 各二级线程向一级线程提交完成的箱子
            for a, i in enumerate(highests):
                last = a == len(highests)-1
                target = len(highests) if last else -1
                st = time()
                subsit = deepcopy(sitning)
                if output2:
                    output2[position] = time()-st
                pro_pool.append(subsit)
                subsit.into(searching)
                sub_code = subsit.num_situ
                nexts[code].append(sub_code)
                while sub_code >= len(points):
                    points.append(-0.900)
                    previous.append(None)
                    nexts.append([])
                    cur_rec.append([None, None])
                previous[sub_code] = code
                # subsit.point = i[0]
                subsit.current = i[1]
                cur_rec[sub_code][0] = i[1]
                
                if i[1] == '结束':
                    cur_rec[sub_code][0] = '输'
                    points[sub_code] = 0.100  # 己方认输，己方胜率10%
                    submit_box.append(-1)
                    # 如果是领路人并且完成未集齐，在此卡主
                    while target >= 0 and len(submit_box) < target:
                        sleep(0.1)
                    # （非）领路人至此，领路人向吹号人提交
                    if target >= 0:
                        with Trainer.locker:
                            Trainer.counter[im_at] += 1
                    continue
                subsit.melst.Out(i[1])
                points[sub_code] = subsit.point

                subsit.exch()
                # 敌我转变，敌方视角
                sub_pool.append(Thread(target=subsit.melst.judge, args=[
                    Trainer.old_brain, subsit.current, -1,
                    im_at, submit_box, target,
                    sub_pool, len(sub_pool), no_rep_n]))
            # 这里注意：后进线程池的有可能先完成（参见judge领路人设计）
            for i in sub_pool:
                i.start()
            for i in sub_pool:
                if type(i) == Thread:
                    i.join()
            for a, i in enumerate(sub_pool):
                subsit = pro_pool[a]
                sub_code = subsit.num_situ
                sub_high = i[0]
                for ii in i:  # 从低到高选择大于0的最低敌敌方（己方）胜率
                    if ii[0] >= 0:
                        sub_high = ii
                        break
                if sub_high[1] == '结束':
                    cur_rec[sub_code][1] = '赢'
                    points[sub_code] = 0.900  # 敌方认输，己方胜率90%
                    subsit.exch()
                    pq = None
                    break
                subsit.point = sub_high[0]
                subsit.melst.Out(sub_high[1])
                subsit.current = sub_high[1]
                cur_rec[sub_code][1] = sub_high[1]
                subsit.exch()
                heapq.heappush(pq, subsit)
            im_at += 1
            count += len(highests)
            if count >= maxi:
                break
        champ = -1
        for i, j in enumerate(points):
            if j >= 0.100 and not nexts[i] and j >= high:
                champ = i
                high = j
        seq = []
        while previous[champ] >= 0:
            seq.append(champ)
            champ = previous[champ]
        seq.reverse()
        tracker = []
        for i in seq:
            recording += str(cur_rec[i][0]) + ' '
            recording += str(cur_rec[i][1]) + ' '
            tracker.append(cur_rec[i])
        if output is not None:
            output[position] = (high, recording, cur_rec[seq[0]][0], tracker)
        with Trainer.locker:
            Trainer.fin_size[im_at-1] -= 1
        # print([points[i] for i in seq])
        return high, recording, cur_rec[seq[0]][0], tracker

    @staticmethod
    @timer
    def get_into_dict(d, k, v):
        if hash(k) in d:
            return True
        d[hash(k)] = v
        return False


class Trainer:
    not_rando = False
    brain = None
    old_brain = None
    to_predict = []
    depth = 20
    ssize = 100
    fin_size = []
    counter = []
    locker = Lock()
    thread_lock = Lock()
    implemented = []
    eff_size = 0
    tot_thread = 0
    til_now = 0
    need_counter = True
    bay = 0
    implem = False
    functional = dict()
    imp = None

    @staticmethod
    def round_acc(y_true, y_pred):
        return metrics.binary_crossentropy(tf.round(y_true), tf.round(y_pred))

    def __init__(self, impo=None):
        if impo is None:
            model = ModelL(loss="mse", metric=Trainer.round_acc)
        else:
            Trainer.imp = impo
            model = ModelL(metric=Trainer.round_acc, modeling=impo)
        Trainer.old_brain = model
        Trainer.brain = model
        self.birth = strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    @timer
    def implement(im_at, brain):
        if Trainer.implemented[im_at]:
            return
        else:
            Trainer.implemented[im_at] = True
        with Trainer.thread_lock:
            Trainer.implem = True
        Trainer.til_now += 1
        pred = Trainer.to_predict[im_at]
        specials = dict()
        for a, i in enumerate(pred):
            samp = i['in_hand']
            if samp[0][3] < 0:
                specials[a] = -0.135  # 无效标记
            # elif np.max(samp[1]) == 0:
            #     specials[a] = 0.100  # 固定输局
        hand = np.array([i['in_hand'] for i in pred])
        curr = np.array([i['current'] for i in pred])
        pred = brain.model.predict(
            {'in_hand': hand, 'current': curr})
        for i in specials:
            pred[i] = specials[i]  # 修改特殊值
        for i in range(len(pred)):
            if pred[i] > 0:
                pred[i] = max(0.150, pred[i])
                pred[i] = min(0.850, pred[i])
        Trainer.to_predict[im_at] = pred
        with Trainer.thread_lock:
            Trainer.implem = False

    @timer
    def sampling(self, sample_size=500, vali_size=0.1,
                 card_num=lambda x: 5*x, H=None, X=None, y=None):
        st = time()
        if H is None:
            H, X, y = dict(), [], []
        pgb = ProgressBar()
        vol, tot = 0, 0
        new_set = []
        rep_set = []
        no_random = []
        for i in pgb(range(sample_size)):
            sit = Situation(maxi=card_num(), auto=False,
                            duplic=lambda x: Situation.get_into_dict(H, x, -1),
                            records=no_random)
            if sit.brand_new:
                new_set.append(sit)
            else:
                rep_set.append(sit)
        if not os.path.exists(f'{sys.path[0]}\\{self.birth}'):
            os.makedirs(f'{sys.path[0]}\\{self.birth}')
        log_file = open(f'{sys.path[0]}\\{self.birth}\\{strftime("%Y-%m-%d_%H-%M-%S")}_gene.txt', 'w')  # noqa E128
        log_file.write(str(no_random))
        log_file.close()
        sizing = 4 * Trainer.depth
        Trainer.fin_size = [0] * sizing
        Trainer.counter = [0] * sizing
        Trainer.to_predict = [[]] * sizing
        Trainer.implemented = [False] * sizing
        Trainer.eff_size = len(new_set)
        Trainer.til_now = 0
        Trainer.bay = 0
        Trainer.implem = False
        threads = []
        outputs = []
        mains = []
        timee = []
        for sit in new_set:
            pos = hash(sit)
            # condition交换敌我，因为实际上是测算‘敌方选择不出时’我方败率（敌方胜率）
            # 这里不交换敌我，因为生成的样本就是给我方的
            mains.append(sit.melst.condition(*sit.transfer(), not_reverse=True))
            threads.append(Thread(target=sit.carding1,
                                  args=[Legal(), Trainer.depth,
                                        outputs, len(outputs), timee]))
            outputs.append(None)
            timee.append(None)
        thread_count = Thread(target=thread_show)
        thread_count.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with Trainer.locker:
            Trainer.need_counter = False
        for a, sit in enumerate(new_set):
            pos = hash(sit)
            score, recording, optim0, track = outputs[a]
            samples = []
            samples.append((mains[a], score))  # 取反值，因为carding1为敌方视角
            # 也可能不取反值，因为样本本身就是己方面临情况
            dual_sc = score
            for optim in track:
                if type(optim[0]) == Legal:
                    sit.melst.Out(optim[0])
                    sit.current = optim[0]
                    dual = sit.melst.condition(*sit.transfer())
                    dual_sc = 1.000 - dual_sc
                    samples.append((dual, dual_sc))
                sit.exch()
                if score > 1 or score < 0 or score > 0.11 and score < 0.89:
                    sit.exch()
                    break
                if type(optim[1]) == Legal:
                    sit.melst.Out(optim[1])
                    sit.current = optim[1]
                    dual = sit.melst.condition(*sit.transfer())
                    dual_sc = 1.000 - dual_sc
                    samples.append((dual, dual_sc))
                sit.exch()
            H[pos] = []
            for i in samples:
                X.append(i[0])
                y.append(i[1])
                H[pos].append((i[1], sit.desc, i[0], recording))
                tot += 1
            vol += 1
        for sit in rep_set:
            pos = hash(sit)
            for i in H[pos]:
                X.append(i[2])
                y.append(i[0])
                tot += 1
        os.system("tada.wav")
        to_write = []
        print(time()-st)
        print(sum(timee))
        # hand = np.array([H[i][0][2][0][0] for i in H])
        # curr = np.array([H[i][0][2][1][0] for i in H])
        # predic_y = Trainer.brain.model.predict(
        #     {'in_hand': hand, 'current': curr})
        hand, curr = [], []
        for i in H:
            samp = H[i][0]
            hand.append(samp[2][0])
            curr.append(samp[2][1])
        hand = np.array(hand)
        curr = np.array(curr)
        tore = Trainer.brain.model.predict(
            {'in_hand': hand, 'current': curr})
        for i, j in zip(H, tore):
            samp = H[i][0]
            sample_y = int(samp[0]*100)
            predic_y = j
            predic_y = int(predic_y[0]*100)
            init_pat = samp[1]
            proc_pat = samp[3]
            fixing = 'OK'
            if (sample_y - 50)*(predic_y - 50) <= 0:
                fixing = 'FX'
            if sample_y < 10:
                sample_y = "0" + str(sample_y)
            if predic_y < 10:
                predic_y = "0" + str(predic_y)
            inn = [fixing, str(sample_y), str(predic_y), init_pat, '流程：', proc_pat]  # noqa E128
            to_write.append(' '.join(inn))
            print(*inn)
        to_write.sort()
        to_write.append(str(Trainer.imp))
        to_write = '\n'.join(to_write)
        if not os.path.exists(f'{sys.path[0]}\\{self.birth}'):
            os.makedirs(f'{sys.path[0]}\\{self.birth}')
        log_file = open(f'{sys.path[0]}\\{self.birth}\\{strftime("%Y-%m-%d_%H-%M-%S")}_log.txt', 'w')  # noqa E128
        log_file.write(to_write)
        log_file.close()
        print('生成了'+str(vol)+'/'+str(sample_size)+'/'+str(tot)+'样本')
        return H, X, y

    @timer
    def train(self, start_card=2, samp_size=100, depth=20):
        print('张数：', start_card)
        Trainer.ssize = samp_size
        Trainer.depth = depth
        H, X, y = dict(), [], []
        while start_card < 20:
            H, X, y = dict(), [], []
            H, X, y = self.sampling(
                sample_size=Trainer.ssize, card_num=lambda: start_card,
                H=H, X=X, y=y)  # random.randint(2, start_card))  # noqa E128
            # vy = tf.constant(vy)
            tot = len(X)
            # ran = [i for i in range(tot)]
            # random.shuffle(ran)
            # X = X[ran]
            # y = y[ran]
            vsize = math.floor(tot*0.1)
            vX1, vX2, vy, X1, X2, y = \
                np.array([x[0] for x in X[:vsize]]).reshape(vsize, 15, 8),\
                np.array([x[1] for x in X[:vsize]]).reshape(vsize, 19),\
                np.array(y[:vsize]).reshape(vsize, 1),\
                np.array([x[0] for x in X[vsize:]]).reshape(tot-vsize, 15, 8),\
                np.array([x[1] for x in X[vsize:]]).reshape(tot-vsize, 19),\
                np.array(y[vsize:]).reshape(tot-vsize, 1)
            metric = Binary(({'in_hand': vX1, 'current': vX2}, vy))
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tb_dir', histogram_freq=1)
                
            Trainer.brain.model.fit({'in_hand': X1, 'current': X2},
                                    y, epochs=30,
                                    callbacks=[metric, tensorboard],
                                    verbose=0)
            if metric.val_precisions[-1] > .95 and\
                    metric.val_recalls[-1] > .95:
                start_card += 1
                print('提升张数为：', start_card)
                sav_name = f'{sys.path[0]}\\{self.birth}\\{strftime("%Y-%m-%d_%H-%M-%S")}_model.h5'  # noqa E501
                Trainer.brain.model.save(sav_name)
                Trainer.imp = sav_name
                Trainer.old_brain = Trainer.brain
                Trainer.brain = ModelL(metric=Trainer.round_acc,
                                       modeling=sav_name)
                Trainer.depth = 20
            else:
                Trainer.ssize *= 2
                Trainer.depth = int(1.2 * Trainer.depth)
                print('提升样本量：', Trainer.ssize)
                print('提升深度：', Trainer.depth)


if __name__ == '__main__':
    if False:
        tr = Trainer(input('导入模型：'))
        tr.train(int(input('起始张数：')))
    if True:
        a = '2021-09-19_13-41-25_model.h5'
        b = '2021-09-19_13-30-40'
        c = 'D:/study/uni notes/08/trai/self/doudizhu/cpu'
        if False:
            tr = Trainer(f'{c}/{b}/{a}')
            tr.not_rando = True
            tr.train(8, 10, 20)
        else:
            tr = Trainer()
            tr.train(2, 100, 20)
