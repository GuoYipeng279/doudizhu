# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import random,sys,math,time
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
import scipy.io as scio
#tf.compat.v1.disable_eager_execution()
np.set_printoptions(threshold=1000000000)
name=['大王','小王']
nums=['大王','小王','2','A','K','Q','J','10','9','8','7','6','5','4','3']
id=[0,1]
for i in range(2,15):
    name+=['黑桃'+nums[i],'红心'+nums[i],'方块'+nums[i],'梅花'+nums[i]]
    id+=[i,i,i,i]
def sigmoid(input):
    return np.tanh(input/math.sqrt(len(input)))
def dsig(input):
    a=np.tanh(input/math.sqrt(len(input)))
    return 1-np.multiply(a,a)
def printout(cards):
    if len(cards[0])==0:return '不出'
    if len(cards[0])==1:
        if cards[1]==1:return nums[cards[0][0]]
        if cards[1]==2:return '对'+nums[cards[0][0]]
        if cards[1]==3:
            if cards[3]==0:return '三个'+nums[cards[0][0]]
            if cards[3]==1:return '三'+nums[cards[0][0]]+'带'+nums[cards[2][0]]
            if cards[3]==2:return '三'+nums[cards[0][0]]+'带一对'+nums[cards[2][0]]
        if cards[1]==4:
            if cards[3]==0:return nums[cards[0][0]]+'炸弹'
            if cards[3]==1:return '四'+nums[cards[0][0]]+'带'+str([nums[i] for i in cards[2]])
            if cards[3]==2:return '四'+nums[cards[0][0]]+'带两对'+str([nums[i] for i in cards[2]])
    if cards==([1,0],1,[],0):return '王炸'
    if cards[1]==1:return str([nums[i] for i in cards[0]])+'顺子'
    if cards[1]==2:return str([nums[i] for i in cards[0]])+'连对'
    if cards[1]==3:return str([nums[i] for i in cards[0]])+'飞机'
    return cards
def activa(act,tot):
    a=np.zeros(tot)
    a[act]=1
    return a


# %%
def initialize(rando,maxi):#随机发牌初始化
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
    return [id[x] for x in melst],[id[x] for x in enlst]
#持牌矩阵
def tomat(ids,self):
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


# %%
def larger(now,me):
    if now[1]==2 and now[2]==1:return False#王炸要不起
    if me[1]==2 and me[2]==1:return True#王炸炸所有
    if me[0]==3 and me[3]==0:#炸弹
        if now[0]!=3 or now[3]!=0:return True#对方不是炸弹，可炸
    #一般情况比大小
    if now[0]!=me[0] or now[1]!=me[1] or now[2]<=me[2] or now[3]!=me[3]:return False#不匹配或不够大
    return True#可压
#当前牌型
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
def leftmove(lst,mov):#左移列表
    for i in range(0,5):
        if i>=4 or lst[i]==0:break
    for j in range(i-mov,i):lst[j]=0
def rightmove(lst,mov):#右移列表
    for i in range(0,5):
        if i>=4 or lst[i]==0:break
    for j in range(i,i+mov):lst[j]=1
def combination(mat,num,dub,i,otp,lst):#连带出牌的排列组合情况
    if num<=0:otp.append([i for i in lst])
    else:
        for j in range(i,15):
            if mat[j][dub-1]==1:
                leftmove(mat[j],dub)
                lst.append(j)
                combination(mat,num-1,dub,j,otp,lst)
                lst.pop()
                rightmove(mat[j],dub)


# %%
def clearzero(memat,enmat):
    count=np.zeros(6)
    cut=[2,3,5]
    groups=[]
    lastzero=-1
    lestzero=-1
    for i in range(0,12):
        for j in range(0,3):
            if memat[i][2-j]>0:
                count[2*j]+=1
                if count[2*j]==cut[j]:
                    if lastzero!=lestzero and i-lastzero<cut[j]:
                        groups.append(lastzero)
                        lestzero=lastzero
                        count=[min(k,i-lastzero) for k in count]
            if enmat[i][2-j]<0:
                count[2*j+1]+=1
                if count[2*j+1]==cut[j]:
                    if lastzero!=lestzero and i-lastzero<cut[j]:
                        groups.append(lastzero)
                        lestzero=lastzero
                        count=[min(k,i-lastzero) for k in count]
        if memat[i][0]==0 and enmat[i][0]==0:
            lestzero=lastzero
            lastzero=i
        else:
            groups.append(i)
            for j in range(0,3):
                if memat[i][2-j]<=0:
                    count[2*j]=0
                if enmat[i][2-j]>=0:
                    count[2*j+1]=0
    groups.sort()
    return groups
def transfer(memat,enmat,min):
    keep=[0,1,2]
    keep+=[i+3 for i in clearzero(memat[3:15],enmat[3:15])]
    for i in range(0,len(keep)+1):
        if i>=len(keep) or keep[i]>=min:
            min=i
            break
    me=[memat[i] for i in keep]
    en=[enmat[i] for i in keep]
    while len(me)<15:
        me.append([0,0,0,0])
    while len(en)<15:
        en.append([0,0,0,0])
    return me,en,i
def translate(me,mat,read):#根据牌型生成出牌后的矩阵，用法：当前牌型，己方矩阵
    otp=list([list(i) for i in mat])
    if me==(0,0,0,0):return [(otp,([],1,[],0))]#不出牌
    news=[]
    lst=[]
    main=[]
    for i in range(me[2],me[2]-me[1],-1):
        leftmove(otp[i],me[0]+1)
        main.append(i)
    if me[3]==0:lst=[[]]
    else:combination(otp,max(0,(me[0]-1)*me[1]),me[3],0,lst,[])
    for i in lst:
        toadd=[k for k in otp]
        for j in i:
            toadd[j]=[k for k in toadd[j]]
            leftmove(toadd[j],me[3])
        if read:
            rea=[]
            for k in toadd:
                kk=0
                for m in range(0,4):kk+=k[m]
                rea.append(int(kk))
            toadd=rea
        news.append((toadd,(main,me[0]+1,i,me[3])))
    return news#结果矩阵，（主牌型，叠张，连带牌型，连带张）
def imtrans(out):#输入translate函数的输出，取得牌型
    if out[0]==[]:return (0,0,0,0)#任意牌型
    return (out[1]-1,out[0][0]-out[0][-1]+1,out[0][-1],out[3])


# %%

from itertools import chain
import heapq
def tovec(inp,enm,read=False):#由全局信息获得输入向量，用法：translate函数输出列表的元素，敌方矩阵
    style=imtrans(inp[1])
    m,n,s2=transfer(inp[0],enm,style[2])
    md=list(np.zeros(36))#0~14张数，15~29最小牌，30~32额外张，33~35连带张
    md[style[0]+30]=md[style[1]]=md[s2+15]=md[style[3]+33]=1
    nud=[i for i in chain.from_iterable(m)]+[i for i in chain.from_iterable(n)]+md
    if read==False:return nud
    sum=0
    for i in range(0,156):sum+=nud[i]*2**i
    return sum
def tove(mem,enm,sty):#由全局信息获得输入向量，用法：translate函数输出列表的元素，敌方矩阵
    m,n,s2=transfer(mem,enm,sty[2])
    md=list(np.zeros(36))#0~14张数，15~29最小牌，30~32额外张，33~35连带张
    md[sty[0]+30]=md[sty[1]]=md[s2+15]=md[sty[3]+33]=1
    nud=[i for i in chain.from_iterable(m)]+[i for i in chain.from_iterable(n)]+md
    return nud
        


# %%
ita,beta,batch_div=0.005,0.2,5.0#步长，动量项权重，batch精度
print('步长：'+str(ita))
print('动量项权重：'+str(beta))
print('batch精度：'+str(batch_div))
color=['red','blue','yellow']
class Net:#子网络
    def __init__(self,Set,id):
        self.id=id
        self.samples=[]
        self.M,self.B,self.Vm,self.Vb=[np.zeros(1)],[np.zeros(1)],[np.zeros(1)],[np.zeros(1)]
        self.lo=len(Set)-1
        for i in range(0,self.lo):
            self.M.append(np.mat(np.random.uniform(-1,1,size=(Set[i+1],Set[i]))))
            self.Vm.append(np.zeros((Set[i+1],Set[i])))
            self.B.append(np.mat([random.uniform(-1,1) for j in range(0,Set[i+1])]).T)
            self.Vb.append(np.zeros((Set[i+1],1)))
        self.V,self.U,self.D=[None for i in Set],[None for i in Set],[None for i in Set]
    def forward(self,input):
        #根据己方剩牌，敌方剩牌，出完后敌方面临的牌型，得出出牌打分（敌方先手败率、己方后手胜率）
        self.V[0]=np.mat(input).T
        for i in range(1,self.lo+1):
            self.U[i]=np.matmul(self.M[i],self.V[i-1])-self.B[i]
            self.V[i]=sigmoid(self.U[i])
        return self.V[-1]
    def backward(self,y):
        lo=self.lo
        self.D[-1]=np.multiply((self.V[lo]-y),dsig(self.U[lo]))
        for k in range(1,lo):
            i=lo-k
            self.D[i]=np.multiply(np.matmul(self.M[i+1].T,self.D[i+1]),dsig(self.U[i]))
        for i in range(1,lo+1):
            self.Vm[i]=beta*self.Vm[i]+(1-beta)*np.matmul(self.D[i],self.V[i-1].T)
            #self.M[i]=self.M[i]-ita*np.matmul(self.D[i],self.V[i-1].T)
            self.M[i]=self.M[i]-ita*self.Vm[i]
            self.Vb[i]=beta*self.Vb[i]+(1-beta)*self.D[i]*np.mat(np.ones(self.D[i].shape[1])).T
            #self.B[i]=self.B[i]-ita*self.D[i]
            self.B[i]=self.B[i]-ita*self.Vb[i]
    def findbest(self,memat,enmat,current):#寻找敌方先手败率最高出牌，用法：己方矩阵，敌方矩阵，当前面临牌型
        otp,maxi=None,None
        for j in [translate(i,memat,False) for i in getPossible(memat,current)]:
            for p in j:
                score=self.forward(tovec(p,enmat,False))#敌方先手败率
                #print(np.mat(tovec(p,enmat,False)).shape)
                #print(score)
                if maxi==None or score>maxi:
                    maxi=score
                    otp=(p[0],p[1],maxi)
        return otp#输出为得分最高的（出牌后己方矩阵，所出牌张，打分(y)）
    def train(self,tolerence,percentage,batch_size=1):
        if batch_size==0:batch_size=int(math.pow(len(self.samples),1/batch_div))
        print('子网'+str(self.id)+'，batch_size：'+str(batch_size))
        if len(self.samples)==0:return
        maxi=0
        while True:
            if batch_size!=1:
                batch=[]
                for i in range(0,len(self.samples)//batch_size):
                    bat=random.sample(self.samples,batch_size)
                    batch.append((np.mat([i[0] for i in bat]),0,np.mat([i[2] for i in bat])))
            else:batch=self.samples
            count=0
            for s in batch:
                divv=chain.from_iterable(np.array(abs(self.forward(s[0])-s[2])))
                for t in divv:
                    if t>tolerence:self.backward(s[2])
                    else:count+=1
            rat=100*count/len(batch)/batch_size
            if rat>maxi:sys.stdout.write(' '+str(int(rat)))
            else:sys.stdout.write('.')
            sys.stdout.flush()
            maxi=max(maxi,rat)
            if count/len(batch)/batch_size>percentage:
                self.samples=[]
                sys.stdout.write('\n')
                sys.stdout.flush()
                return


# %%
from multiprocessing import Process
class Par:
    def __init__(self):
        self.fails=[-1]
        self.ff=0
        self.high=-1
        self.sett,self.maxcard,self.chi,self.s_sz=None,None,None,None
        a=int(input('录入：'))
        #a=1
        if a==0:
            print('子网结构')
            self.sett=setset(1)
            self.maxcard=0
            self.s_sz=100
            self.chi=[(Net(self.sett,0),Net(self.sett,0))]#子网络，信任网络
        else:self.load(a)
    def getmax(self,samp):#通过信任网络判断最擅长的子网
        otp,ma=-1,None
        for c in self.chi:
            trust=c[1].forward(samp)
            if trust>otp:
                otp=trust
                ma=c
        return ma
    def formax(self,memat,enmat,current):#由实际判断最擅长的子网（考试）（低效）
        otp,ma=0,None
        for c in self.chi:
            f=c[0].forward(tove(memat,enmat,current))
            if f>otp:
                otp=f
                ma=c
        return ma
    def forsamax(self,samp,single=False):#由最擅长的子网计算预估胜负
        if single:return self.chi[0][0].forward(samp)
        return self.getmax(samp)[0].forward(samp)
    def forwamax(self,memat,enmat,current):#由最擅长的子网计算预估胜负
        vec=tove(memat,enmat,current)
        return self.getmax(vec)[0].forward(vec)
    def findbestmax(self,memat,enmat,current):#有最擅长的子网计算最佳出牌
        vec=tove(memat,enmat,current)
        return self.getmax(vec)[0].findbest(memat,enmat,current)
    def Y_Generator(self,memat,enmat,cur,hand,show,otp):#生成己方后手胜败，hand奇me己en敌，偶me敌en己
        current=imtrans(cur)
        otp.append(tovec((memat,cur),enmat))
        muq=''
        if hand%2==0:
            muq='敌：'
            if not np.any(memat):
                return 1#sigmoid(sum(chain.from_iterable(enmat))/2)#敌方先手时，发现己方无牌，认负，己方胜
        else:
            muq='己：'
            if not np.any(memat):
                return -1#-sigmoid(sum(chain.from_iterable(enmat))/2)#己方先手时，发现敌方无牌，认负，敌方胜
        best=self.findbestmax(enmat,memat,current)#先手方寻找后手方先手败率、先手方后手胜率最高的出牌
        if show:print(muq+printout(best[1]))
        return self.Y_Generator(best[0],memat,best[1],hand+1,show,otp)#交换出牌方
    def train(self,samples,single=False):
        if single:
            self.chi[0][0].samples=samples
            self.chi[0][0].train(0.3,0.98,0)
        return
        for s in samples:
            minn=999
            c=None
            for c0 in self.chi:#进行子网考试，先由1号网做，若合适则训练信任网，不合适则测试2号网
                d=abs(c0[0].forward(s[0])-s[2])
                if d<0.5 and d<minn:#距离较近者最近者，否则交给新网
                    c=c0
                    minn=d
            if c==None:c=self.chi[-1]#都不成功，则交给新网
            c[0].samples.append(s)
            c[1].samples.append((s[0],0,1))#对应网络对此添加信任
            for c0 in self.chi:
                if c0!=c:c0[1].samples.append((s[0],0,-1))#其他网络对此添加不信任
        for c in self.chi:
            c[1].train(0.7,0.9,1)#信任度训练
            c[0].train(0.3,0.98,1)#新网训练
    def sample(self,samp_sz=1,show=False,maxi=27):#生成训练样本，由己方牌，敌方牌，当前牌型，计算敌方先手的败率
        win,lose,tot,wro=0,0,0,0
        otp=[]
        while win<samp_sz or lose<samp_sz or len(otp)==0:
            tot+=1
            if tot>4*samp_sz:
                print('总样本数：'+str(len(otp)))
                return otp
            m,n=initialize(True,maxi)
            mmat,nmat=tomat(m,True),tomat(n,True)
            #我方任意先手，取得敌方当前规则下最佳出牌
            aux=[]
            real=self.Y_Generator(mmat,nmat,([],1,[],0),0,show,aux)#敌方先手、己方后手实际胜败
            if win<samp_sz and real>0 or lose<samp_sz and real<0 or len(otp)==0:
                non=real
                for au in aux:
                    otp.append([au,0,non])
                    non=-non
                if real>0:win+=1
                else:lose+=1
        print('总样本数：'+str(len(otp)))
        return otp#输出（输入向量，预测敌方先手败率、己方后手胜率，实际敌方输赢）
    def record(self,count):
        np.save('sav{}.npy'.format(count),(self.maxcard,self.s_sz,self.sett,self.chi))
        # Embedding_UI = 'E://Neo/Matr{}.mat'.format(str(count))
        # scio.savemat(Embedding_UI, {'train':self.chi})
        # self.data=(self.maxcard,self.sett)
        # Embedding_UI = 'E://Neo/Data{}.mat'.format(str(count))
        # scio.savemat(Embedding_UI, {'train':self.data})
    def load(self,count):
        self.maxcard,self.s_sz,self.sett,self.chi=np.load('sav{}.npy'.format(count),allow_pickle=True)
        print('最大张数：'+str(self.maxcard))
        print('样本容量：'+str(self.s_sz))
        print('网络结构：'+str(self.sett))
        # dataFile1 = 'E://Neo/Matr{}.mat'.format(str(count))
        # data = scio.loadmat(dataFile1)
        # self.chi=data['train']
        # dataFile1 = 'E://Neo/Data{}.mat'.format(str(count))
        # data = scio.loadmat(dataFile1)
        # self.maxcard,self.sett=data['train']
    def work(self,single=False):
        self.maxcard+=1
        while True:#采样-训练 循环
            print('最大：'+str(self.maxcard)+' 网络数：'+str(len(self.chi)))
            samples=self.sample(samp_sz=self.s_sz,maxi=self.maxcard)
            count=0
            for s in samples:
                if abs(np.mean(self.forsamax(s[0],single)-s[2]))<0.6:count+=1
            ra=count/len(samples)*100
            print('初始准确率：'+str(round(ra,2)))
            self.train(samples,single)
            samples=self.sample(100,maxi=self.maxcard)
            count=0
            for s in samples:
                if abs(np.mean(self.forsamax(s[0],single)-s[2]))<0.6:count+=1
            ra=count/len(samples)*100
            #若成功做出样本则增加最大牌数，若提升不理想，则新增网络
            if ra>92:
                self.record(self.maxcard)
                self.maxcard+=1
                self.high=-1
                self.fails=[-1]
                self.ff=0
                #sample_size=100
                self.s_sz=int(self.s_sz*1.2)
                print('正确率：'+str(round(ra,2))+' NEXT')
            elif ra>self.high:
                self.high=ra
                self.ff=0
                self.fails.append(ra)
                self.s_sz=int(self.s_sz*1.5)
                print('正确率：'+str(round(ra,2))+' HIGH '+str([round(i,2) for i in self.fails]))
            else:
                while self.fails[-1]>ra:
                    self.fails.pop()
                    self.ff+=1
                self.fails.append(ra)
                self.s_sz=int(self.s_sz*1.8)
                print('正确率：'+str(round(ra,2))+' FALL '+str([round(i,2) for i in self.fails])+' high:'+str(round(self.high,2))+',bad:'+str(self.ff))
                if self.ff>9999:
                    self.ff=0
                    self.fails=[-1]
                    self.s_sz=100
                    self.chi.append((Net(self.sett,len(self.chi)),Net(self.sett,len(self.chi))))


# %%
def setset(out):
    otp=[156]
    count=0
    print('设置神经网络，第一层156，推荐[100,100]')
    while True:
        count+=1
        a=int(input('隐含层第'+str(count)+'层神经元数量：'))
        if a==0:
            otp.append(out)
            return otp
        otp.append(a)


# %%
import traceback
par=Par()
try:
    par.work(True)
except:
    traceback.print_exc()
    r=input('错误')


# %%
#ita=float(input('训练步长，推荐0.001：'))
mode=input('1-训练模式，2-采样模式，3-对战模式')
if mode==1:
    ita=0.001
    working()
elif mode==2:
    loa=int(input('输入参数：'))
    global maxcard,M,V,U,D,B,lo,Vf,Uf,Df
    if loa>0:
        load(loa)
        sys.stdout.write('156')
        for b in B:sys.stdout.write('->'+str(b.shape))
        sys.stdout.flush()
        lo=len(M)-1
        r=range(0,len(M))
        V,U,D=[None for i in r],[None for i in r],[None for i in r]
        Vf,Uf,Df=[None for i in r],[None for i in r],[None for i in r]
    while True:
        sample(1,True,maxi,[])
        if int(input('继续？1/0：'))!=1:break
else:
    loa0=int(input('我方AI：'))
    loa1=int(input('敌方AI：'))
    


# %%



