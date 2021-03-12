import random
import numpy as np


class Card:
    '''牌'''
    typs = ['黑桃', '红心', '方块', '梅花']
    nums = ['大王','小王','2','A','K','Q','J','10','9','8','7','6','5','4','3'] # noqa E231
    
    def __init__(self, num, typ):
        '''输入大小与花色'''
        self.num = num
        self.typ = typ
    
    def __init__(self, lab):
        '''输入卡牌编号'''
        if lab < 2:
            self.num = lab
            self.typ = 0
        else:
            lab -= 2
            self.num = lab // 4
            self.typ = lab % 4
    
    def __str__(self):
        if self.num < 2:
            return Card.nums[self.num]
        return Card.typs[self.typ] + Card.nums[self.num]


class Legal:
    '''合法牌型'''
    def __init__(self, card_array, multiple, auxi, auxmul):
        '''输入主牌数组，主牌重复数，带牌数组，带牌重复数'''
        self.card_array, self.multiple, self.auxi, self.auxmul = card_array, multiple, auxi, auxmul
    
    def __str__(self):
        if len(self.card_array) == 0:
            return '不出'
        if len(self.card_array) == 1:
            if self.multiple == 1:
                return Card.nums[self.card_array[0]]
            if self.multiple == 2:
                return '对' + Card.nums[self.card_array[0]]
            if self.multiple == 3:
                if self.auxmul == 0:
                    return '三个' + Card.nums[self.card_array[0]]
                if self.auxmul == 1:
                    return '三' + Card.nums[self.card_array[0]] + '带' + Card.nums[self.auxi[0]]
                if self.auxmul == 2:
                    return '三' + Card.nums[self.card_array[0]] + '带一对' + Card.nums[self.auxi[0]]
            if self.multiple == 4:
                if self.auxmul == 0:
                    return Card.nums[self.card_array[0]] + '炸弹'
                if self.auxmul == 1:
                    return '四' + Card.nums[self.card_array[0]] + '带' + str([Card.nums[i] for i in self.auxi])
                if self.auxmul == 2:
                    return '四' + Card.nums[self.card_array[0]] + '带两对' + str([Card.nums[i] for i in self.auxi])
        if (self.card_array, self.multiple, self.auxi, self.auxmul) == ([1, 0], 1, [], 0):
            return '王炸'
        if self.multiple == 1:
            return str([Card.nums[i] for i in self.card_array]) + '顺子'
        if self.multiple == 2:
            return str([Card.nums[i] for i in self.card_array]) + '连对'
        if self.multiple == 3:
            return str([Card.nums[i] for i in self.card_array]) + '飞机'
        return 'cards'

    @staticmethod
    def larger(now, me):
        '''比大小，输入对方牌，己方牌'''
        if now[1] == 2 and now[2] == 1:
            return False  # 王炸要不起
        if me[1] == 2 and me[2] == 1:
            return True  # 王炸炸所有
        if me[0] == 3 and me[3] == 0:  # 炸弹
            if now[0] != 3 or now[3] != 0:
                return True  # 对方不是炸弹，可炸
        # 一般情况比大小
        if now[0] != me[0] or now[1] != me[1] or now[2] <= me[2] or now[3] != me[3]:
            return False  # 不匹配或不够大
        return True  # 可压


class Situation:
    '''情景'''
    def __init__(self):
        '''随机发牌'''
        maxi = 27
        lst = list(range(0, 54))
        random.shuffle(lst)
        menum = ennum = random.randint(1, maxi)
        if random.randint(0, 1) == 0:
            menum = maxi
        else:
            ennum = maxi
        melst = [lst[x] for x in range(0, menum)]
        enlst = [lst[x] for x in range(menum, menum+ennum)]
        melst.sort()
        enlst.sort()
        self.melst, self.enlst = [id[x] for x in melst], [id[x] for x in enlst]


class Cards:
    '''持牌情况'''
    def __init__(self,  take):
        '''输入持牌情况（数列）'''
        self.take = take

    @property
    def tomat(self):
        '''牌数列转矩阵'''
        mematrix = np.zeros((15, 4))
        rec = -1
        num = 0
        for i in self.take:
            if i != rec:
                rec = i
                num = 0
            if self:
                c = num
            else:
                c = 3 - num
            mematrix[rec, c] = 1
            num += 1
        return mematrix
    
    def getPossible(self, now):
        '''当前持牌面对当前牌型的所有可选出牌'''
        mat = self.tomat
        lst = []
        count = [0, 0, 0, 0]
        conse = [5, 3, 2, 20]  # 连牌数量，顺子5连，连对3连，飞机2连
        if mat[0][0] == 1 and mat[1][0] == 1:
            lst.append(Legal(0, 2, 1, 0))
        for i in range(0, 15):
            if i == 3:
                count = [0, 0, 0, 0]  # 顺子，连对，飞机，最大从A开始
            for j in range(0, 4):
                if mat[i][j] == 1:
                    toadd = (j, 1, i, 0)
                    lst.append(toadd)  # 单，对，三，炸
                    if j > 1:
                        lst.append(Legal(j, 1, i, 1))
                        lst.append(Legal(j, 1, i, 2))
                    count[j] += 1
                    if count[j] >= conse[j]:
                        toadd = (j, count[j], i, 0)
                        lst.append(toadd)  # 顺，连对，飞机不带
                        if j > 1:
                            lst.append(Legal(j, count[j], i, 1))
                            lst.append(Legal(j, count[j], i, 2))
                else:
                    count[j] = 0
        if now[1] == 0:
            return lst
        otp = [(0, 0, 0, 0)]  # 若当前有牌型，则可选择不出，否则不可不出牌
        for i in lst:
            if Legal.larger(now, i):
                otp.append(i)
        return otp

