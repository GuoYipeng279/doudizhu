import random

class Card:
    '''牌'''
    typs=['黑桃','红心','方块','梅花']
    nums=['大王','小王','2','A','K','Q','J','10','9','8','7','6','5','4','3']
    
    def __init__(self, num, typ):
        self.num=num
        self.typ=typ
    
    def __init__(self, lab):
        if lab<2:
            self.num=lab
            self.typ=0
        else:
            lab-=2
            self.num=lab//4
            self.typ=lab%4
    
    def __str__(self):
        if num<2:return nums[self.num]
        return typs[self.typ]+nums[self.num]

class Legal:
    '''合法牌型'''
    def __init__(self, card_array, multiple, auxi, auxmul):
        self.card_array, self.multiple, self.auxi, self.auxmul
        =card_array, multiple, auxi, auxmul
    
    def __str__(self):
        if len(card_array)==0:return '不出'
        if len(card_array)==1:
            if multiple==1:return nums[card_array[0]]
            if multiple==2:return '对'+nums[card_array[0]]
            if multiple==3:
                if auxmul==0:return '三个'+nums[card_array[0]]
                if auxmul==1:return '三'+nums[card_array[0]]+'带'+nums[auxi[0]]
                if auxmul==2:return '三'+nums[card_array[0]]+'带一对'+nums[auxi[0]]
            if multiple==4:
                if auxmul==0:return nums[card_array[0]]+'炸弹'
                if auxmul==1:return '四'+nums[card_array[0]]+'带'+str([nums[i] for i in auxi])
                if auxmul==2:return '四'+nums[card_array[0]]+'带两对'+str([nums[i] for i in auxi])
        if (card_array, multiple, auxi, auxmul)==([1,0],1,[],0):return '王炸'
        if multiple==1:return str([nums[i] for i in card_array])+'顺子'
        if multiple==2:return str([nums[i] for i in card_array])+'连对'
        if multiple==3:return str([nums[i] for i in card_array])+'飞机'
        return cards

class Situation:
    '''情景'''
    def __init__(self):
        lst=list(range(0,54))
        random.shuffle(lst)
        if rando:menum=ennum=random.randint(1,maxi)
        else:menum=ennum=int(input())
        if random.randint(0,1)==0:menum=maxi
        else:ennum=maxi
        melst=[lst[x] for x in range(0,menum)]
        enlst=[lst[x] for x in range(menum,menum+ennum)]
        melst.sort()
        enlst.sort()
        self.melst,self.enlst
        =[id[x] for x in melst],[id[x] for x in enlst]

class Cards:
    '''持牌情况'''
    def __init__(self, take):
        self.take=take 

    def tomat(self):
        mematrix=np.zeros((15,4))
        rec=-1
        num=0
        for i in ids:
            if i!=rec:
                rec=i
                num=0
            if self:c=num
            else:c=3-num
            mematrix[rec,c]=1
            num+=1
        return mematrix
    
    def getPossible(mat,now):
        lst=[]
        count=[0,0,0,0]
        conse=[5,3,2,20]
        if mat[0][0]==1 and mat[1][0]==1:lst.append((0,2,1,0))
        for i in range(0,15):
            if i==3:count=[0,0,0,0]#顺子，连对，飞机，最大从A开始
            for j in range(0,4):
                if mat[i][j]==1:
                    toadd=(j,1,i,0)
                    lst.append(toadd)#单，对，三，炸
                    if j>1:
                        lst.append((j,1,i,1))
                        lst.append((j,1,i,2))
                    count[j]+=1
                    if count[j]>=conse[j]:
                        toadd=(j,count[j],i,0)
                        lst.append(toadd)#顺，连对，飞机不带
                        if j>1:
                            lst.append((j,count[j],i,1))
                            lst.append((j,count[j],i,2))
                else:count[j]=0
        if now[1]==0:return lst
        otp=[(0,0,0,0)]#若当前有牌型，则可选择不出，否则不可不出牌
        for i in lst:
            if larger(now,i):otp.append(i)
        return otp
