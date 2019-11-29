
# -*- coding:utf-8 -*-
import json
import random
import jieba
import predict_ZQJ_3_1
import DNNmodel as BertModel
import tensorflow as tf
import os


### 智慧星光会调用prd()函数，所以我们的集成在prd()里面进行 ###
def prd(data):
    # 将传入的data转换为dict,rawdata是dict，key为文本号，value为要预测的语句
#    json_string = json.dumps(data)
#    rawdata = json.loads(json_string)
    
    
    
###################################################################################################### 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True     
    rawdata={}
    truelabel={}
    file_open = open('./ceshi.txt','r',encoding='UTF-8')
    for line in file_open:
        line = line.replace('\n','')
        split_item = line.split('|')
        rawdata[split_item[0]] = split_item[1] 
        truelabel[split_item[0]] = split_item[2]
    file_out=open('./reout.txt','w',encoding='UTF-8')
 ######################################################################################################         
        
        
    ########################################################################
    # 调用你代码的预测接口，传入rawdata，进行预测，预测结果写入 yuntao.txt
    # 每一行的格式为  key|class1|rate1|class2|rate2|class3|rate3
    # 给出一个dict，key为种类，value为该类预测准确率，命名为cat_rate_yuntao
    log_file = open( "./yuntao.txt", "w", encoding="UTF-8")
    M = BertModel.Bert_Classify(max_sequence_length=512,config=config,Modelname="Model1")
    M.load()  
    for k, v in rawdata.items():
        output=M.outputdata(k,v)
        log_file.write(output)
        log_file.flush()
    cat_rate_yuntao = {'rhxz': 0.73, 'zjxy': 0.813, 'tcqy': 0.883, 'yjqx': 0.637, 'khwl': 0.894, 'jz': 0.955,
                     'dcxz': 0.860, 'kdts': 0.923, 'sjhx': 0.897, 'kdgz': 0.909, 'kdwsm': 0.897, 'ydwlxhc': 0.832,
                     'fscpjb': 0.945, 'dkxz': 0.824, 'dtvxz': 0.867, 'rhlw': 0.737, 'lw': 0.738, 'rhcf': 0.705,
                     'znzw': 0.939, 'kdwzy': 0.868,'others':0.8}
    ###########################################################################

    # 处理你的结果，化成dict ，key是文本号，value是三个最高概率种类和对应概率值
    result_yuntao = {}
    file_open = open('./yuntao.txt','r',encoding='UTF-8')
    for line in file_open:
        line = line.replace('\n','')
        split_item = line.split('|')
        result_yuntao[split_item[0]] = split_item[1:]  
    # 将rawdata传入预测函数predict_ZQJ_3_1.key_cv，进行预测，预测的内容写入了result.txt中
    # 从result.txt中读取内容，存储到result中，result为dict
    predict_ZQJ_3_1.key_cv(rawdata)
    result = {}
    return_result = {}
    for line in open('result.txt','r',encoding='UTF-8'):
        line = line[1:]
        line = line.replace("[", "").replace("]", "")
        w = line.replace("\n", "").split("$$")
        # w[0]是文本号;w[1]w[2]w[3]是最高概率的三个种类
        # w[1]组成为   名称|概率|命中关键词···
        result[w[0]] = list(w[1:])
    # 对每一类的预测准确率
    cat_rate_deng = {'rhxz':0.784,'zjxy':0.864,'tcqy':0.818,'yjqx':0.872,'khwl':0.964,'jz':0.898,
                     'dcxz':0.865,'kdts':0.884,'sjhx':0.79,'kdgz':0.836,'kdwsm':0.876,'ydwlxhc':0.87,
                     'fscpjb':0.928,'dkxz':0.89,'dtvxz':0.914,'rhlw':0.85,'lw':0.698,'rhcf':0.778,
                     'znzw':0.93,'kdwzy':0.92,'others':0.93}
    # 名称转换
    convert = {}
    convert['XZ-RHXZ'] = 'rhxz'
    convert['TCQY-BXLLTCQY'] = 'tcqy'
    convert['XYXF-ZJXY'] = 'zjxy'
    convert['WTWJ-YJQX'] = 'yjqx'
    convert['CFLH-ZGWHKWL'] = 'khwl'
    convert['XFXY-JF'] = 'jz'
    convert['XZ-DCXZ'] = 'dcxz'
    convert['TCQY-KDJS'] = 'kdts'
    convert['HJHKHGX-SJHX'] = 'sjhx'
    convert['WTWJ-KDGZ'] = 'kdgz'
    convert['WTWJ-KDWSM'] = 'kdwsm'
    convert['WTWJ-YDXHC'] = 'ydwlxhc'
    convert['JZ-FSCPJB'] = 'fscpjb'
    convert['XZ-DKXZ'] = 'dkxz'
    convert['XZ-DTVXZ'] = 'dtvxz'
    convert['CFLH-RHLW'] = 'rhlw'
    convert['CFLH-LWALL'] = 'lw'
    convert['CFLH-RHCF'] = 'rhcf'
    convert['JZ-ZNZW'] = 'znzw'
    convert['WTWJ-KDWZY'] = 'kdwzy'
    convert['others'] = 'others'

    # 反向转换
    reverse_convert = {v: k for k, v in convert.items()}
    # 统计你预测的最大和概率种类不在我预测的三个最大种类的量
    wrong_num = 0

    # 每条语句
    for item1 in result_yuntao:
        # ensemble是将我们对该条文本（文本号为item1）分别预测的三个概率最大场景合起来，key是场景名称，value是概率。用于查找该文本最大可能场景
        ensemble = {}
        ensemble[result_yuntao[item1][0]] = float(result_yuntao[item1][1])*cat_rate_yuntao[result_yuntao[item1][0]]
        ensemble[result_yuntao[item1][2]] = float(result_yuntao[item1][3])*cat_rate_yuntao[result_yuntao[item1][2]]
        ensemble[result_yuntao[item1][4]] = float(result_yuntao[item1][5])*cat_rate_yuntao[result_yuntao[item1][4]]
#        print(ensemble)
        # keyword存储该条文本对应的种类要输出的命中关键词
        keyword = {}
        num = 0
        # 处理我预测的结果result.txt
        for item2 in result[item1]:
            split_list = item2.split('|')
            cat = convert[split_list[0]]
            rate = float(split_list[1])
            keyword[cat] = split_list[2:]# 存储的是list
            # 将我预测的结果合并到ensemble中，曾经出现则概率相加，未出现则新增一项
            if(cat in ensemble):
                ensemble[cat] += rate/5*cat_rate_deng[cat]
            else:
                ensemble[cat] = rate/5*cat_rate_deng[cat]
            num += 1
            # 只找前三个最大
            if(num == 3):
                break
#        print(ensemble)
        # 如果我们两个人3个预测结果都不相同，该文本归到others
        # 或者如果你预测到的最大可能性的种类没有在我预测的3个类中出现，归为others
        if(len(ensemble) == 6 ):
            return_result[item1] = 'others|'+ str(random.random())+'|'
            continue

        # return_value_list存储要返回的三个最大概率场景list，list每个元素包括   场景名称|概率|命中关键词
        return_value_list = []
        # 查找三个概率最大的场景
        for i in range(3):
            max_cata = max(ensemble, key=ensemble.get)
            keyword_str = ''
            # 如果找到的该类在我预测的几个种类中
            if(max_cata in keyword):
                for child_list in keyword[max_cata]:
                    keyword_str += str(child_list) + '|'
                return_value = str(reverse_convert[max_cata])+'|'+str(round(ensemble[max_cata]/2,2))+'|'+keyword_str
                return_value = return_value[:-1]
                return_value_list.append(return_value)
                del(keyword[max_cata])
            else:
                wrong_num+=1
                return_value = 'others|'+ str(random.random())+'|'
                return_value_list.append(return_value)
            # 删除当前最大概率种类，重新寻找
            del(ensemble[max_cata])

######################################################################################################            
            # 输出结果
            if(i == 0):
                print(str(item1)+'|'+max_cata+'|'+truelabel[item1])
                file_out.write(str(item1)+'|'+max_cata+'|'+truelabel[item1]+'\n')
######################################################################################################                  
                
                
                
        # 得到返回值
        
        return_result[item1] = return_value_list
    print('wrong num:'+str(wrong_num))

    return json.dumps(return_result)  # 输出Json结果



prd(1)