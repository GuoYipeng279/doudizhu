'''

'''

import numpy as np


class Card:
    '''牌'''
    typs = ['黑桃', '红心', '方块', '梅花']
    nums = ['大王','小王','2','A','K','Q','J','10','9','8','7','6','5','4','3'] # noqa E231

    def __init__(self, *args):
        if len(args) == 2:
            '''输入大小与花色'''
            self.num = args[0]
            self.typ = args[1]
        elif len(args) == 1:
            '''输入卡牌编号'''
            lab = args[0]
            if lab < 2:
                self.num = lab
                self.typ = 0
            else:
                lab -= 2
                self.num = lab // 4 + 2
                self.typ = lab % 4

    @staticmethod
    def to_int(char):
        for i, j in enumerate(Card.nums):
            if j == char:
                if i < 2:
                    return i
                else:
                    return 2 + 4*(i-2)

    def __str__(self):
        if self.num < 2:
            return Card.nums[self.num]
        return Card.typs[self.typ] + Card.nums[self.num]


class Legal:
    '''具体合法牌型
        __init__:输入主牌数组，主牌重复数，带牌数组，带牌重复数
        例如，三个K带对2，是([K],3,[2],2)
        创建具体牌型，以及需求指针列表
        或
        输入字符串，生成合法牌型，以_分割主牌与带牌
        未设置合法检查，除测试中，禁用
    '''
    def __init__(self, *args):
        if len(args) == 1:
            lst = args[0].split('_')
            self.card_array = [Card.nums.index(x) for x in set(lst[0])]
            self.multiple = len(lst[0])//len(self.card_array)
            if len(lst) == 2:
                self.auxi = [Card.nums.index(x) for x in set(lst[1])]
                self.auxmul = len(lst[1])//len(self.auxi)
            else:
                self.auxi = []
                self.auxmul = 0
        elif len(args) == 4:
            self.card_array = args[0]
            self.multiple = args[1]
            self.auxi = args[2]
            self.auxmul = args[3]
        else:  # 空牌
            self.card_array = []
            self.multiple = 1
            self.auxi = []
            self.auxmul = 0
        # 2021-9-9 fix bug: AAA带A需求问题
        self.upper = []
        repetition = dict()
        for i in self.card_array:
            repetition[i] = len(self.upper)
            self.upper.append([i, self.multiple])
        for i in self.auxi:
            if i in repetition:
                self.upper[repetition[i]][1] += self.auxmul
            else:
                self.upper.append([i, self.auxmul])
        self.stillin = True  # Flag: 该牌型是否已被打出，因而已不存在

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
                    return '三' + Card.nums[self.card_array[0]] \
                        + '带' + Card.nums[self.auxi[0]]
                if self.auxmul == 2:
                    return '三' + Card.nums[self.card_array[0]] \
                        + '带一对' + Card.nums[self.auxi[0]]
            if self.multiple == 4:
                if self.auxmul == 0:
                    return Card.nums[self.card_array[0]] + '炸弹'
                if self.auxmul == 1:
                    return '四' + Card.nums[self.card_array[0]] \
                        + '带' + str([Card.nums[i] for i in self.auxi])
                if self.auxmul == 2:
                    return '四' + Card.nums[self.card_array[0]] \
                        + '带两对' + str([Card.nums[i] for i in self.auxi])
        if (self.card_array, self.multiple, self.auxi, self.auxmul) == ([1, 0], 1, [], 0): # noqa E501
            return '王炸'
        if self.multiple == 1:
            return str([Card.nums[i] for i in self.card_array]) + '顺子'
        if self.multiple == 2:
            return str([Card.nums[i] for i in self.card_array]) + '连对'
        if self.multiple == 3:
            return str([Card.nums[i] for i in self.card_array]) + '飞机'
        return 'wrong cards'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self.card_array) + str(self.multiple) +
                    str(self.auxi) + str(self.auxmul))

    @staticmethod
    def abstr(tup):
        if tup[1] == 0:
            return '不出'
        if tup[1] == 1:
            if tup[0] == 0:
                return Card.nums[tup[2]]
            if tup[0] == 1:
                return '对' + Card.nums[tup[2]]
            if tup[0] == 2:
                if tup[3] == 0:
                    return '三个' + Card.nums[tup[2]]
                if tup[3] == 1:
                    return '三' + Card.nums[tup[2]] \
                        + '带一'
                if tup[3] == 2:
                    return '三' + Card.nums[tup[2]] \
                        + '带一对'
            if tup[0] == 3:
                if tup[3] == 0:
                    return Card.nums[tup[2]] + '炸弹'
                if tup[3] == 1:
                    return '四' + Card.nums[tup[2]] \
                        + '带二'
                if tup[3] == 2:
                    return '四' + Card.nums[tup[2]] \
                        + '带两对'
        if tup == (0, 2, 1, 0):
            return '王炸'
        rg = str(Card.nums[tup[2]])+'-'+str(Card.nums[tup[2]-tup[1]+1])
        if tup[0] == 0:
            return rg + '顺子'
        if tup[0] == 1:
            return rg + '连对'
        if tup[0] == 2:
            return rg + '飞机'
        return 'wrong cards '+str(tup)

    @property
    def arr(self):
        otp = np.zeros(15, dtype=int)
        for i in self.card_array:
            otp[i] = self.multiple
        return otp

    @property
    def typ(self):
        '''由实际牌总结牌型（具体转抽象）'''
        if len(self.card_array) == 0:
            return (0, 0, 0, 0)
        return (self.multiple-1, len(self.card_array),
                min(self.card_array), self.auxmul)

    @staticmethod
    def larger(now, me):
        '''比大小，输入对方抽象牌型，己方抽象牌型'''
        if now[1] == 2 and now[2] == 1:
            return False  # 王炸要不起
        if me[1] == 2 and me[2] == 1:
            return True  # 王炸炸所有
        if me[0] == 3 and me[3] == 0:  # 炸弹
            if now[0] != 3 or now[3] != 0:
                return True  # 对方不是炸弹，可炸
        # 一般情况比大小
        if now[0] != me[0] or now[1] != me[1] \
                or now[2] <= me[2] or now[3] != me[3]:
            return False  # 不匹配或不够大
        return True  # 可压

    def __lt__(self, other):
        return Legal.larger(other.typ, self.typ)


class Cards:
    '''持牌情况'''
    def __init__(self, take, auto=True):
        '''输入持牌情况（数列）
        生成所有具体牌型，并写入固定需求表
        judge
        '''
        self.empty = True
        self.take = np.zeros(15, dtype=int)
        for i in take:
            self.IOD(Card(i).num, 'in')
        self.orii = self.take.copy()
        if auto:
            self.construct()
        self.supervise = []  # 行为监视器

    def construct(self):
        self.table = [[[] for i in range(4)] for j in range(15)]
        abstract = self.getPossible()
        specific = []
        typing = set([(i[0], i[1], i[3]) for i in abstract])
        self.relation = dict([(i, set()) for i in typing])
        for i in abstract:
            specific.extend(self.translate(i))
        self.specific = set(specific)
        self.out_stack = []
        self.Out_Stack = []
        self.empty = False

    def __str__(self):
        ret = ''
        for i, elm in enumerate(self.take):
            for j in range(elm):
                ret += Card.nums[i]
                ret += ' '
        return ret

    def __repr__(self):
        return str(self)

    @property
    def matr(self):
        otp = list(np.zeros((15, 4), dtype=int))
        for i in range(15):
            for j in range(self.take[i]):
                otp[i][j] = 1
        return otp

    def _IOD(self, fak, num, typ, n, bk):
        if typ == 'out':
            fak[num] -= n
            if fak[num] < 0:
                print('no')
                print(bk)
                print(self.supervise)
                print(self.take)
                print(self.specific)
                print(self.parent.desc)
                print([][0])
            return
        elif typ == 'in':
            fak[num] += n
            return
        elif typ == 'det':
            return fak[num] >= n
        else:
            return

    def IOD(self, num, typ, n=1, card=None):
        '''typ=抓取in，打出out，检测det'''
        return self._IOD(self.take, num, typ, n, card)

    def getPossible(self, **kwargs):
        '''初始化用，当前持牌面对当前抽象牌型的所有可选抽象出牌
        格式为叠张（单张为0），连张，最小牌（最大数），带牌叠张
        输出为抽象牌型列表
        '''
        now = (0, 0, 0, 0)
        if 'now' in kwargs:
            now = kwargs['now']
        show = False
        if 'show' in kwargs:
            show = kwargs['show']
        lst = []
        count = [0, 0, 0, 0]
        conse = [5, 3, 2, 20]  # 连牌数量，顺子5连，连对3连，飞机2连
        if self.IOD(0, 'det') and self.IOD(1, 'det'):
            lst.append((0, 2, 1, 0))  # 判王炸
        for i in range(0, 15):
            if i == 3:
                count = [0, 0, 0, 0]  # 顺子，连对，飞机，最大从A开始
            for j in range(0, 4):
                if self.IOD(i, 'det', j+1):
                    toadd = (j, 1, i, 0)
                    lst.append(toadd)  # 单，对，三，炸
                    if j > 1:
                        other_max = 0
                        for e, k in enumerate(self.take):
                            if e != i:
                                other_max = max(k, other_max)
                        if other_max > 0:
                            lst.append((j, 1, i, 1))
                        if other_max > 1:
                            lst.append((j, 1, i, 2))
                    count[j] += 1
                    if count[j] >= conse[j]:
                        for k in range(count[j], conse[j]-1, -1):
                            toadd = (j, k, i, 0)
                            lst.append(toadd)  # 顺，连对，飞机不带
                            # 飞机部分，待完成
                            # if j > 1:
                            #     lst.append((j, k, i, 1))
                            #     lst.append((j, k, i, 2))
                else:
                    count[j] = 0
        # print('debug')
        # print(lst)
        if now[1] == 0:
            if show:
                print([Legal.abstr(x) for x in lst])
            return lst
        otp = [(0, 0, 0, 0)]  # 若当前有牌型，则可选择不出，否则不可不出牌
        for i in lst:
            if Legal.larger(now, i):
                otp.append(i)
        if show:
            print([Legal.abstr(x) for x in lst])
        return otp

    def combination(self, num, dub, otp, i=0, lst=[]):
        '''连带出牌的排列组合情况（用于抽象转具体）
        num带牌组数（三带一为1，三带对为1，四带二为2，345飞机为3）
        dub带牌叠张（带1张为1，带对为2）
        otp为空白需求单
        可选参数otp,i,lst请留空
        '''
        if num <= 0:
            otp.append(lst.copy())
        else:
            for j in range(i, 15):
                if self.IOD(j, 'det', dub):
                    lst.append(j)
                    self.IOD(j, 'out', dub)
                    self.combination(num-1, dub, otp, j, lst)
                    self.IOD(j, 'in', dub)
                    lst.pop()

    def translate1(self, me):
        '''根据抽象牌型生成出牌后的矩阵与具体牌型, 疑似有问题'''
        fake = self.take.copy()  # 疑似有问题
        if me == (0, 0, 0, 0):
            return [(fake, ([], 1, [], 0))]  # 不出牌
        news, lst, main = [], [], []
        for i in range(me[2], me[2]-me[1], -1):
            self._IOD(fake, i, 'out', me[0]+1)
            main.append(i)
        if me[3] == 0:
            lst = [[]]
        else:
            self.combination(max(0, (me[0]-1)*me[1]), me[3], lst)
        for i in lst:
            toadd = fake.copy()
            for j in i:
                self._IOD(toadd, lst[j], 'out', me[3])
            news.append((toadd, (main, me[0]+1, i, me[3])))
        return news  # 结果矩阵，（主牌型，叠张，连带牌型，连带张）

    def translate2(self, legal):
        '''根据具体牌型生成出牌后的矩阵与具体牌型'''
        fake = self.take.copy()
        for i in legal.card_array:
            self._IOD(fake, i, 'out', legal.multiple)
        for i in legal.auxi:
            self._IOD(fake, i, 'out', legal.auxmul)
        return (fake, legal)  # 结果矩阵，（主牌型，叠张，连带牌型，连带张）

    def translate(self, me):
        '''只用于初始化，根据抽象牌型生成全部具体牌型'''
        news, lst, main = [], [], []
        for i in range(me[2], me[2]-me[1], -1):
            self.IOD(i, 'out', me[0]+1)
            main.append(i)
        if me[3] == 0:
            lst = [[]]
        else:
            self.combination(max(0, (me[0]-1)*me[1]), me[3], lst)
        for i in lst:
            legal = Legal(main, me[0]+1, i, me[3])
            self.relation[(me[0], me[1], me[3])].add(legal)
            news.append(legal)
            for j in legal.upper:
                self.table[j[0]][j[1]-1].append(legal)  # 将具体牌型加入需求表
        for i in range(me[2], me[2]-me[1], -1):
            self.IOD(i, 'in', me[0]+1)
        return set(news)  # （主牌型，叠张，连带牌型，连带张）

    def Out(self, legal):
        '''具体出牌函数，将处理需求池，因而难以可逆，20210618：仍尝试可逆'''
        self.supervise.append(("Out", legal))
        toRem = set()
        recording = dict()
        for i in legal.card_array:  # 主牌型打出
            loss = range(self.take[i]-legal.multiple, self.take[i])
            for j in loss:
                while self.table[i][j]:
                    outing = self.table[i][j].pop()
                    if outing in recording:
                        recording[outing].add((i, j))
                    else:
                        recording[outing] = set([(i, j)])
                    if outing.stillin:
                        outing.stillin = False  # 标记打出
                        toRem.add(outing)
            self.IOD(i, 'out', legal.multiple)
        for i in legal.auxi:  # 带牌打出
            loss = range(self.take[i]-legal.auxmul, self.take[i])
            for j in loss:
                while self.table[i][j]:
                    outing = self.table[i][j].pop()
                    if outing in recording:
                        recording[outing].add((i, j))
                    else:
                        recording[outing] = set([(i, j)])
                    if outing.stillin:
                        outing.stillin = False  # 标记打出
                        toRem.add(outing)
            self.IOD(i, 'out', legal.auxmul)
        # 去除由于出牌造成的可出牌型减少
        ori_num = len(self.specific)
        for i in toRem:
            t = i.typ
            t = (t[0], t[1], t[3])
            self.relation[t].remove(i)
            if len(self.relation[t]) == 0:
                self.relation.pop(t)
        if len(self.relation) == 0:
            self.empty = True
        self.specific = self.specific.difference(toRem)
        self.Out_Stack.append((recording, toRem, legal))
        return len(self.specific), ori_num

    def Ret(self):
        recording, toAdd, legal = self.Out_Stack.pop()
        self.supervise.append(("Ret", legal))
        for i in legal.card_array:
            self.IOD(i, 'in', legal.multiple)
        for i in legal.auxi:
            self.IOD(i, 'in', legal.auxmul)
        for i in recording:
            for j in recording[i]:
                self.table[j[0]][j[1]].append(i)
        self.empty = False
        self.specific = self.specific.union(toAdd)
        for i in toAdd:
            t = i.typ
            t = (t[0], t[1], t[3])
            i.stillin = True
            if t not in self.relation:
                self.relation[t] = set()
            self.relation[t].add(i)

    def out(self, legal):
        self.supervise.append(("out", legal))
        self.out_stack.append(legal)
        for i in legal.card_array:
            self.IOD(i, 'out', legal.multiple)
        for i in legal.auxi:
            self.IOD(i, 'out', legal.auxmul)

    def ret(self):
        legal = self.out_stack.pop()
        self.supervise.append(("ret", legal))
        for i in legal.card_array:
            self.IOD(i, 'in', legal.multiple)
        for i in legal.auxi:
            self.IOD(i, 'in', legal.auxmul)


if __name__ == '__main__':
    print(Card.to_int('K'))