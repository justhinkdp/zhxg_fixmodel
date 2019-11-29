# encoding:utf-8
# 提前特征词
def extractFeatures():
    from jieba import analyse
    tpk = 250
    rt = 2.0
    path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_3\\'
    file1 = open(path+'dcxz.txt', 'rb') # 单c新装
    data = file1.read()
    w1={}
    for w in sorted(analyse.extract_tags(data,withWeight=True,topK=tpk),key=lambda d:d[1], reverse=True):
        w1[w[0]] = w[1]

    # print 'end 1\n'
    file2 = open(path+'kdts.txt', 'rb') #宽带提速
    data = file2.read()
    w2={}
    for w in sorted(analyse.extract_tags(data,withWeight=True,topK=tpk),key=lambda d:d[1],reverse=True):
        w2[w[0]]=w[1]

    # print 'end 2\n'
    file3 = open(path+'sjhx.txt', 'rb') # 手机换新
    data = file3.read()
    w3={}
    for w in sorted(analyse.extract_tags(data,withWeight=True,topK=tpk),key=lambda d:d[1],reverse=True):
        w3[w[0]]=w[1]

    # print 'end 3\n'
    file4 = open(path+'others.txt', 'rb')
    data = file4.read()
    w4={}
    for w in sorted(analyse.extract_tags(data,withWeight=True,topK=tpk*5),key=lambda d:d[1],reverse=True):
        w4[w[0]]=w[1]

    # print 'end 4\n'
    wtp=[]
    for w in w1:
        if not w in w2 and not w in w3 and not w in w4:
            pass
        else:
            f=True
            if w in w2:
                if w2[w]*rt>w1[w]:
                    f=False
            if w in w3:
                if w3[w]*rt>w1[w]:
                    f=False
            if w in w4:
                if w4[w]*rt>w1[w]:
                    f=False
            if f:
                wtp.append(w)
    w11=wtp

    wtp=[]
    for w in w2:
        if not w in w1 and not w in w3 and not w in w4:
            pass
        else:
            f=True
            if w in w1:
                if w1[w]*rt>w2[w]:
                    f=False
            if w in w3:
                if w3[w]*rt>w2[w]:
                    f=False
            if w in w4:
                if w4[w]*rt>w2[w]:
                    f=False
            if f:
                wtp.append(w)
    w22=wtp

    wtp=[]
    for w in w3:
        if not w in w2 and not w in w1 and not w in w4:
            pass
        else:
            f=True
            if w in w2:
                if w2[w]*rt>w3[w]:
                    f=False
            if w in w1:
                if w1[w]*rt>w3[w]:
                    f=False
            if w in w4:
                if w4[w]*rt>w3[w]:
                    f=False
            if f:
                wtp.append(w)
    w33=wtp

    wtp=[]
    for w in w4:
        if not w in w2 and not w in w1 and not w in w3:
            pass
        else:
            f=True
            if w in w2:
                if w2[w]*rt>w4[w]:
                    f=False
            if w in w1:
                if w1[w]*rt>w4[w]:
                    f=False
            if w in w3:
                if w3[w]*rt>w4[w]:
                    f=False
            if f:
                wtp.append(w)
    w44=wtp
    # for w in w11:
    #     print 'w11',w
    # for w in w22:
    #     print 'w22',w
    # for w in w33:
    #     print 'w33',w
    # for w in w44:
    #     print 'w44',w
    tow = open(path+'keywords_single_'+str(tpk)+'.txt', 'wb')
    for w in w11:
        tow.write(w.encode("utf-8")+b" ")
    tow.write("1\n")
    for w in w22:
        tow.write(w.encode("utf-8")+b" ")
    print
    tow.write("2\n")

    for w in w33:
        tow.write(w.encode("utf-8")+b" ")
    tow.write("3\n")

# extractFeatures()
