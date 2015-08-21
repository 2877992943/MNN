import random
import os
import sys
import math
import numpy as np



trainPath = "D://python2.7.6//MachineLearning//MLNN//trainingDigits"
testPath = "D://python2.7.6//MachineLearning//MLNN//testDigits"
outfile1 = "D://python2.7.6//MachineLearning//MLNN//1.txt"


     

global classDic;classDic={}
global dataList
global labelList
global epoch;epoch=15
global alpha;alpha=0.2
 
numw=10 #10 class
 
numc=64
lbd=0.05#lambda loss=loss+lambda*ww
global dataMat,yMat,outputMat 
######################

def loadData():
    global dataMat,yMat,outputMat 
    global dataList
    global labelList
    dataList=[]
    labelList=[]
    ###################build obs list 
    for filename in os.listdir(trainPath):
        pos=filename.find('_')
        clas=int(filename[:pos])
        if clas not in classDic:classDic[clas]=1.0
        else:classDic[clas]+=1.0
        labelList.append(clas)
        ##########
        obs=[]
        content=open(trainPath+'/'+filename,'r')
        line=content.readline().strip('\n')
        while len(line)!=0:
            hang=[]
            for num in line:
                hang.append(float(num))
            obs.append(hang)
            line=content.readline().strip('\n')
        #print '1',len(obs) # 1x1024 dim for each obs
        #print '1',len(obs),len(obs[0])  #obs 32x32
        #########change 32x32 into 16x16
        x256=[]
        for i in range(16):
            a=i*2;b=i*2+1
            for j in range(16):
                m=j*2;n=j*2+1
                ######
                point=0.0
                for hang1 in [a,b]:
                    for lie1 in [m,n]:
                        point+=obs[hang1][lie1]
                ######
                x256.append(point)
        ###############
        #print '2',len(x256)
        dataList.append(x256)
    ##########
    print '%d obs loaded'%len(dataList),len(labelList),'labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
    outPutfile=open(outfile1,'w')
    for obs in dataList:
        for n in obs:
            outPutfile.write(str(n))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    ##### creat valid set for calc LL based on same data
    dataMat=np.mat(dataList)
    
    
    n,d=np.shape(dataMat)
    outputMat=np.zeros((n,10));outputMat=np.mat(outputMat)
    yMat=np.zeros((n,10));yMat=np.mat(yMat)
    for i in range(n):
        truey=labelList[i]
        yMat[i,truey]=1.0
    ######
    

 

def initialH():
    global dataMat,yMat,outputMat 
    global hMat 
    num,dim=np.shape(dataMat)
    hMat=np.zeros((num,numc));hMat=np.mat(hMat)
     

def initialPara():
    global dataMat,yMat,outputMat 
    #global hMat,hhMat,hhhMat
    global Cmat,Wmat #initial from random eps
    num,dim=np.shape(dataMat)
     
    Cmat=np.zeros((numc,dim));Cmat=np.mat(Cmat)
    Wmat=np.zeros((numw,numc));Wmat=np.mat(Wmat)
    for i in range(numc):
        for j in range(dim):
            Cmat[i,j]=random.uniform(0,0.1)
    for i in range(numw):
        for j in range(numc):
            Wmat[i,j]=random.uniform(0,0.1)
    #######
     
    global mat2B,matB
    
    mat2B=np.mat(np.zeros((1,numw)))
    for j in range(numw):
        mat2B[0,j]=random.uniform(0,0.1)
    mat2B=np.tile(mat2B,[num,1])
     
    ##
    matB=np.mat(np.zeros((1,numc)))
    for j in range(numc):
        matB[0,j]=random.uniform(0,0.1)
    matB=np.tile(matB,[num,1])
    

            
def initialErr():
    global errW,errC
    global dataMat
    n,d=np.shape(dataMat)
    errW=np.mat(np.zeros((n,numw)))
    errC=np.mat(np.zeros((n,numc)))
     
    
def initialGrad():
    global dataMat;
    n,dim=np.shape(dataMat)
    global gradW,gradB,gradC,grad2B 
    gradW=np.mat(np.zeros((numw,numc))) #total numw pieces of vectot, each one with dim of numc
    gradB=np.mat(np.zeros((1,numc)))

    gradC=np.mat(np.zeros((numc,dim)))
    grad2B=np.mat(np.zeros((1,numw)))

     

def forward(i):#use one obs to calc output, h , i==the index of obs
    global Cmat,Wmat
    global mat2B,matB####
    global dataMat,yMat,outputMat 
    global hMat 
    global errW,errC 
    global gradW,gradB,gradC,grad2B 
    
    num,dim=np.shape(dataMat)
    #####
    hMat[i,:]=dataMat[i,:]*Cmat.T  #1x256  x  256x64 == 1x64
    #print '1' , np.shape(hMat),np.shape(matB) #matB into 1934x256???
    hMat[i,:]=hMat[i,:]+matB[i,:] ##########
    hMat[i,:]=activation(hMat[i,:],0)
    ##
    outputMat[i,:]=hMat[i,:]*Wmat.T  #1x64  x  64x10 ==1x10
    outputMat[i,:]=outputMat[i,:]+mat2B[i,:] #############
    outputMat[i,:]=softmax(outputMat[i,:])
    #outputMat=normalize(outputMat,'prob')
    #print outputMat
    
def calcLoss(): #use one obs to calc grad forward, but calc loss based on valid dataset 
    global Cmat,Wmat
    global mat2B,matB 
    global dataMat,yMat,outputMat  
    global hMat 
    global errW,errC 
    global gradW,gradB,gradC,grad2B 
    #num=np.shape(validMat)[0]  use all data to calc loss cuz donot want to calc valid data h hh hhh 
    num=np.shape(dataMat)[0]
    
    loss=0.0
    for n in range(num)[500:701]:############  (y-fx)^2 not cross entropy as loss
        diff=outputMat[n,:]-yMat[n,:]
        s=diff*diff.T;s=s[0,0]
        loss+=s
    ####regularization
    r3=Cmat.A**2
    r3=r3.sum(1).sum(0)*lbd;#r3=r3[0,0]
    r4=Wmat.A**2
    r4=r4.sum(1).sum(0)*lbd;#r4=r4[0,0]
    ####
    loss+=r3+r4
    #print 'total loss',loss
    return loss
        
def calcGrad(i):#obs=1,2 ,3...the index of obs
    global Cmat,Wmat
    global mat2B,matB 
    global dataMat,yMat,outputMat 
    global hMat 
    global errW,errC 
    global gradW,gradB,gradC,grad2B 
    global labelList
    num,dim=np.shape(dataMat)
     
    ###gradient back to 0 to accumulate grad of all obs
    initialGrad()
    ####
    ####lambda x w
    r3=Cmat.sum(1).sum(0);r3=r3[0,0]*lbd  #a.sum(1).sum(0)-> [[0]]
    r4=Wmat.sum(1).sum(0);r4=r4[0,0]*lbd

    ####calc err w floor
    fy=outputMat[i,:]-yMat[i,:] #1x10
    sgm=outputMat[i,:].A*(1.0-outputMat[i,:].A)#1 x10  array
    errW[i,:]=np.mat(fy.A*sgm) #1x10

     
    ##grad
    g1=errW[i,:].T*hMat[i,:]+r4;g1=normalize(g1,'vector') #10x1  x  1x64==10x64
    g2=normalize(errW[i,:],'vector')#1x10
    gradW=g1   
    grad2B=g2  #all obs grad accumulate
     
    ###calc err c floor
    errC[i,:]=errW[i,:]*Wmat# 1x10  x  10x64 == 1x64
    sgm=hMat[i,:].A*(1.0-hMat[i,:].A) #1x64 array
    errC[i,:]=np.mat(errC[i,:].A*sgm)#1x64 matric
    
    ###grad
    g1=errC[i,:].T*dataMat[i,:]+r3;g1=normalize(g1,'vector')  #64x1  x  1x256
    g2=normalize(errC[i,:],'vector')#1x64
    gradC=gradC+g1 #64x1  x  1x256  == 64x256
    gradB=gradB+g2 
       
    ###################divide num per batch and normalize gradient
    #gradW=gradW/float(nb)
    #gradB=gradB/float(nb);gradC=gradC/float(nb);grad2B=grad2B/float(nb)
     
    #gradW=normalize(gradW,'vector')
    #gradB=normalize(gradB,'vector')
    #gradC=normalize(gradC,'vector')
    #grad2B=normalize(grad2B,'vector')
     
    

def updatePara(i):
    global Cmat,Wmat
    global mat2B,matB 
    global dataMat,yMat,outputMat 
    global hMat 
    global errW,errC 
    global gradW,gradB,gradC,grad2B 
    global labelList
    num,dim=np.shape(dataMat)

    ##Nmat,Mmat,Cmat,Wmat
     
    Cmat=Cmat-alpha*gradC   
    Wmat=Wmat-alpha*gradW

    ##mat4B,mat3B,mat2B,matB
     
    mat2B[i,:]=mat2B[i,:]-alpha*grad2B
    matB[i,:]=matB[i,:]-alpha*gradB
    
        
#######################################support function
def softmax(outputMat): #1x10
    num,dim=np.shape(outputMat)
    for n in range(num):
        vec=np.exp(outputMat[n,:])  #1x10  #wh+b
        svec=vec.sum(1);svec=svec[0,0]
        outputMat[n,:]=vec/svec
    return outputMat
        
def activation(hmat,opt):
    if opt==0:
        hmat=1.0/(1+np.exp((-1)*hmat))

    if opt==1:
        n,m=np.shape(hmat)
        for nn in range(n):
            for mm in range(m):
                if hmat[nn,mm]<0.0:
                    hmat[nn,mm]=0.0
    if opt!=0 and opt!=1:print 'wrong option,only 0 1'
    return hmat
        
    
def normalize(outmat,opt):
     
    n,d=np.shape(outmat)
    if opt=='prob':
        for i in range(n):
            ss=outmat[i,:].sum(1)
            ss=ss[0,0]
            outmat[i,:]=outmat[i,:]/(ss+0.0001)
    if opt=='vector':
        for i in range(n):
            ss=outmat[i,:]*outmat[i,:].T
            ss=math.sqrt(ss[0,0])
            outmat[i,:]=outmat[i,:]/(ss+0.0001)
    if opt not in ['vector','prob']:
        print 'wrong '
    return outmat 
    
        
def shuffle():
    global dataMat
    num,dim=np.shape(dataMat) #1394 piece of obs
    order=range(num)[:]  #0-100  for loss calc,101...for train obs by obs ///not work. must use whole set to train
    #import copy
    #orderList=copy.copy(order)
    #random.shuffle(orderList)
    random.shuffle(order)
    return order
 

###################main
loadData()
  
#####
initialH()
initialPara()
initialErr()
initialGrad()

###  calc forward output & h latent variable node

#forward()
#calcLoss()

###train
ni,dim=np.shape(dataMat)

for ep in range(epoch):
    orderList=shuffle()#order =[1,9,4,231,...] obs index
    print 'epoch %d'%ep 
    alpha/=1.5
    for obs in orderList:  #range(ni)=[01,2,3,4...]
        forward(obs)  #aim to get h hh hhh output
        calcGrad(obs)  
        updatePara(obs)
        #loss=calcLoss() #not base on one obs, based on stonestill valid set
    loss=calcLoss()
    print 'loss',loss


#####output para w m n c ,b
outPath="D://python2.7.6//MachineLearning//MLNN//parax256"
outfile2 = "N.txt"
outfile3 = "M.txt"
outfile4 = "C.txt"
outfile5 = "W.txt"

outfile6 = "B.txt"
outfile7 = "BB.txt"
outfile8 = "BBB.txt"
outfile9 = "BBBB.txt"
global Cmat,Wmat
global mat2B,matB 
 ##
outPutfile=open(outPath+'/'+outfile4,'w')
n,m=np.shape(Cmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Cmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
##
outPutfile=open(outPath+'/'+outfile5,'w')
n,m=np.shape(Wmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Wmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
###
outPutfile=open(outPath+'/'+outfile6,'w')
n,m=np.shape(matB)

for j in range(m):
    outPutfile.write(str(matB[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
## 
outPutfile=open(outPath+'/'+outfile7,'w')
n,m=np.shape(mat2B)

for j in range(m):
    outPutfile.write(str(mat2B[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
 
        
        
        
        
    
    
    







    
    
