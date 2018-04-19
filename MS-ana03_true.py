#-*- coding:utf-8 -*-
import os
import shutil
import glob
import math
import copy
import scipy.optimize as opt
import numpy as np
import matplotlib.pylab as plt

from pprint import pprint

#print (u"MS解析プログラム")
path1="spectrum/*.txt"
#path2=""
files=[]
files=glob.glob(path1)
#files+=glob.glob(path2)
#print (files)

c1=1
c2=1
c3=1
c4=1
c5=1
c6=1
c7=1
c8=1

k1=len(files)

st="st"
st1="st1"
st2="st2"
st3="st3"

for i1 in range(k1):
    os.mkdir("./gauss")
    #今回使う配列の宣言
    flist0=[]
    flist1=[]
    flist2=[[]]
    flist3=[]
    flist4=[]
    flist5=[]
    flist6=[]
    flist7=[]
    flist8=[]
    flist9=[]
    flist10=[]
    flist1000=[]

    f=open(files[i1],"r")
    flist0=files[i1].split(".")
    #print("flist0",flist0)
    line=f.readline()

    cst1=0

    while line:
        #数値がtabで区切られているため、tabで分割
        flist1=line.split("	")
        #２つ目の要素から改行コードの消去
        flist1[1]=flist1[1].replace("\n","")
        #３つ目の要素の追加（X軸の値）
        flist1.append(0)
        #4つ目要素（順番）の追加（y軸の値）
        flist1.append(cst1)
        cst1=cst1+1

      #list2にx,y座標をfloat型で収納
        flist1[0]=float(flist1[0])
        flist1[1]=float(flist1[1])
        flist2.append(flist1)
        line=f.readline()


    f.close()

    #list2定義の時点で空リストがあるため消去
    flist2.pop(0)

        # print("flist2",flist2)
    #y軸の値をもとにソート(降順なので大きい値から)
    flist3=sorted(flist2,key=(lambda x: x[1]),reverse=True)
    #print (flist3)

    c1=flist3[0][0]
    c2=flist3[0][1]
    c3=c2*0.075

    k2=len(flist3)
    print("最大値・最大値x軸値・ガウス関数フィッティング範囲y・プロット数＝",c1,c2,c3,k2)

    #flist4:最大値*0.075の値までのリスト
    for i2 in range(k2):
        if flist3[i2][1]>c3:
            #print(flist3[i2][1])
            flist4.append(flist3[i2])



    #flist4のそれぞれのプロットが極大値であるかの判定


    k3=len(flist4)
    for i3 in range(k3):
        centerY=flist4[i3][1]
        number=flist4[i3][3]
     #検索プロットの両隣のプロット値(y)の代入
        try:
            centerY_minus1=flist2[number-1][1]
            centerY_plus1=flist2[number+1][1]
        except IndexError:
            centerY_minus1=centerY
            centerY_plus1=centerY
        leftdif=centerY - centerY_minus1
        rightdif=centerY - centerY_plus1


    #flist5は極大値プロットリスト

        if leftdif>0 and rightdif>0:
            flist5.append(flist4[i3])

    #print ("flist5",flist5)
    #gaussian fitting の範囲を決める
    k4=len(flist5)
    k5=int(k2/2)

    flist10=copy.deepcopy(flist2)
    for i4 in range(k4):
        peakY=flist5[i4][3]
        flist6.append(flist5[i4])
        #左肩のプロット勾配
        for i5 in range(k5):
            endpeakleft=flist2[peakY-i5][1]-flist2[peakY-i5-1][1]
            if endpeakleft>0:
                flist6.append(flist2[peakY-i5-1])
            else:
                break
        #右肩のプロット勾配
        for i7 in range(k5):
            endpeakright=flist2[peakY+i7][1]-flist2[peakY+i7+1][1]
            if endpeakright>0:
                flist6.append(flist2[peakY+i7+1])
            else:
                break

        #１つの極大に関してプロットリストをｘ軸順にソート
        flist7=sorted(flist6,key=(lambda x: x[3]))
        flist6.clear()
        #pprint(flist7)
        #flist7は一つの極大をもつデータリスト(x軸順)
        str1='gauss/gauss-plot'
        str2=str(i4)+'.txt'
        str3=str1+str2
        file=open(str3,'w+')

        #flist7をもとにflist1のリストからフィッティング範囲を取り除く

        lh1=min(flist7,key=(lambda x: x[3]))
        lh2=max(flist7,key=(lambda x: x[3]))
        h1=int(lh1[3])
        h2=int(lh2[3])+1
    # print("h1,h2=",h1,h2)

        for i in range(h1,h2):
            flist10[i]=[]

        k6=len(flist7)
        for i8 in range(k6):
            str4=str(flist7[i8])
            str5=str4.replace('[','').replace(']','')
            slist=str5.rsplit(',',2)
            str6=slist[0]
            file.write(str6+"\n")
        file.close()

#gaussフィッティングをするためにx,y軸をそれぞれのリストに分ける
        data1=np.genfromtxt(str3,delimiter=',')
        x=data1[:,0]
        y=data1[:,1]
# print(x)
# print(y)
        #gauss関数定義
        def gaussian(x,A,mu,sigma):
            gauss=A/math.sqrt(2.0*math.pi)/sigma * np.exp(-((x-mu)/sigma)**2/2)
            return(gauss)

        def residual_with_error(param_fit,y,x,yerr):
            A,mean,sigma,base=param_fit
            err=(y-(gaussian(x,A,mean,sigma)+base))/yerr
            return(err)

        mu=flist5[i4][0]
        sig=np.sqrt(np.sum(y*(x-mu)**2)/np.sum(y))

        # print('フィッティング極大のx値,標準偏差,フィッティング極大y値',mu,sig,max(y))

        # x=np.linspace(min(x),max(x),100)
        popt,pcov=opt.curve_fit(gaussian,x,y,p0=[max(y),mu,sig])
        xnew = np.linspace(x.min(),x.max(),300)
        # plt.plot(xnew,gaussian(xnew,popt[0],popt[1],popt[2]),'b',label='gauss fit')
        xnew=xnew.tolist()
        gau=gaussian(xnew,popt[0],popt[1],popt[2]).tolist()
        ii2=len(xnew)
        for i in range(ii2):
            flist101=[]
            flist101.append(xnew[i])
            flist101.append(gau[i])
            flist1000.append(flist101)

    #gaussフィットしていない残りの点は隣接する3点で平均化
    plt.xlabel('X')
    plt.ylabel('Y')

    x=[]
    y=[]
        #元データのプロット
        #print(flist7)
        #flist9.remove(flist7)
        #data=np.genfromtxt('sp/mass01.txt',delimiter='	')
        #x=data[:,0]
        #y=data[:,1]

    x=[d[0] for d in flist2]
    y=[d[1] for d in flist2]

    plt.plot(x,y,'y.',label='Raw')

    #flist10の空リストを消去
    flist10=[x for x in flist10 if x]
        #pprint(flist10)
    h3=int(len(flist10)/3)
        #print("h3=",h3)


    #flist10のy軸値を3点ごとに平均化
    for i5 in range(h3):
        i6=3*i5
        k1=flist10[i6][1]
        k2=flist10[i6+1][1]
        k3=flist10[i6+2][1]
        ave=k1+k2+k3
        ave=ave/3
        flist10[i6][1]=ave
        flist10[i6+1][1]=ave
        flist10[i6+2][1]=ave
        #pprint(flist10)

    flist1000.extend(flist10)
    flist1000=sorted(flist1000,key=(lambda x: x[0]))

    x=[d[0] for d in flist1000]
    y=[d[1] for d in flist1000]
    plt.plot(x,y,'b',label='gauss fit')
    #グラフ表示を最後に
    plt.legend()
    plt.show()
    shutil.rmtree("./gauss")

flist0.clear()
flist1.clear()
flist2.clear()
flist3.clear()
flist4.clear()
flist5.clear()
flist6.clear()
flist7.clear()
flist8.clear()
flist9.clear()
flist10.clear()
flist1000.clear()
