import random
import os
import sys
import math
import numpy as np



trainPath = "D://python2.7.6//MachineLearning//MLNN//trainingDigits"
testPath = "D://python2.7.6//MachineLearning//MLNN//testDigits"
outfile1 = "D://python2.7.6//MachineLearning//MLNN//1.txt"

outPath="D://python2.7.6//MachineLearning//MLNN//para1"
 
outfile4 = "C.txt"
outfile5 = "W.txt"

outfile6 = "B.txt"
outfile7 = "BB.txt"
 

     

global classDic;classDic={}
global dataList
global labelList
global epoch;epoch=2
global alpha;alpha=0.1
global nBatch;nBatch=20
numw=10 #10 class
numc=256
 
lbd=0.5#lambda loss=loss+lambda*ww
global dataMat,yMat,outputMat
######################

def loadData():
    global dataMat,yMat,outputMat
    global dataList
    global labelList
    dataList=[]
    labelList=[]
    ###################build obs list 
    for filename in os.listdir(testPath):
        pos=filename.find('_')
        clas=int(filename[:pos])
        if clas not in classDic:classDic[clas]=1.0
        else:classDic[clas]+=1.0
        labelList.append(clas)
        ##########
        obs=[]
        content=open(testPath+'/'+filename,'r')
        line=content.readline().strip('\n')
        while len(line)!=0:
            for num in line:
                obs.append(float(num))
            line=content.readline().strip('\n')
        #print '1',len(obs) # 1x1024 dim for each obs
         
        dataList.append(obs);#print 'datalist',len(dataList)
    ##########
    print '%d test obs loaded'%len(dataList),len(labelList),'labels',len(dataList[0]),'dim'
    #print labelList,classDic
     
     
    dataMat=np.mat(dataList)
    n,d=np.shape(dataMat)
    outputMat=np.zeros((n,10));outputMat=np.mat(outputMat)
    yMat=np.zeros((n,10));yMat=np.mat(yMat)
    for i in range(n):
        truey=labelList[i]
        yMat[i,truey]=1.0
    ######
    #######
def loadPara():
    global mat2B,matB 
    global Cmat,Wmat
    global dataMat,yMat,outputMat
    num,dim=np.shape(dataMat)
     
    
    ###############
    content=open(outPath+'/'+outfile4,'r')  #cmat
    line=content.readline().strip('\n').strip(' ')
    Clist=[] 
    while len(line)!=0:
        obs=[]
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                obs.append(float(n))
                 
        line=content.readline().strip('\n').strip(' ')
        Clist.append(obs)
    ####
    Cmat=np.mat(Clist)
    numc,dim=np.shape(np.mat(Clist));
    #####################
    content=open(outPath+'/'+outfile5,'r')  #wmat
    line=content.readline().strip('\n').strip(' ')
    Wlist=[] 
    while len(line)!=0:
        obs=[]
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                obs.append(float(n))
                 
        line=content.readline().strip('\n').strip(' ')
        Wlist.append(obs)
    ####
    Wmat=np.mat(Wlist)
    numw,numc=np.shape(np.mat(Wlist));
    ##########
    print numw,numc 
    ################
    content=open(outPath+'/'+outfile6,'r')  #matb
    line=content.readline().strip('\n').strip(' ')
    Blist=[] 
    while len(line)!=0:
         
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                Blist.append(float(n))
                #Nmat[n,i]=float(line[i])
        line=content.readline().strip('\n').strip(' ')  #len(' ')== 1
         
    ####
    matB=np.mat(Blist)
    print np.shape(np.mat(Blist));
    #######################
    content=open(outPath+'/'+outfile7,'r')  #mat2b
    line=content.readline().strip('\n').strip(' ')
    B2list=[] 
    while len(line)!=0:
         
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                B2list.append(float(n))
                #Nmat[n,i]=float(line[i])
        line=content.readline().strip('\n').strip(' ')  #len(' ')== 1
         
    ####
    mat2B=np.mat(B2list)
    print np.shape(np.mat(B2list));
 
        
        

def initialH():
    global dataMat,yMat,outputMat
    global hMat 
    num,dim=np.shape(dataMat)
    hMat=np.zeros((num,numc));hMat=np.mat(hMat)
      

def forward():
    global Cmat,Wmat
    global mat2B,matB####
    global dataMat,yMat,outputMat 
    global hMat 
     
    num,dim=np.shape(dataMat)
    
    hMat=dataMat*Cmat.T  #2x1024  x  1024x256 == 2x256
    hMat=hMat+np.tile(matB,[num,1])
    hMat=activation(hMat,0)
    ##
    outputMat=hMat*Wmat.T  #2x30  x  30x10 ==2x10
    outputMat=outputMat+np.tile(mat2B,[num,1])
    outputMat=softmax(outputMat)
    #outputMat=normalize(outputMat,'prob')
    print outputMat
    
def predict():
    global dataMat,yMat,outputMat
    global labelList
    num,dim=np.shape(dataMat)
    err=0.0
    for n in range(num):
        truey=labelList[n]
        maxP=-10;maxL=0
        for w in range(10):
            if maxP==-10 or maxP<outputMat[n,w]:
                maxP=outputMat[n,w]  #update leizhu
                maxL=w
        ######final winner
        if truey!=maxL:err+=1.0
        print truey,maxL
    #####
    print 'err',err/float(num)
    
    
    
    
         
#######################################support function
def softmax(outputMat): #2x10
    num,dim=np.shape(outputMat)
    for n in range(num):
        vec=np.exp(outputMat[n,:])
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
    
 

###################main
loadData()
loadPara()
 
#### initial
initialH()
forward()
predict()
 

 



    
    
