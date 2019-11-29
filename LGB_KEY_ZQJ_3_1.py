# encoding:utf-8
# 训练分类器.
import lightgbm as lgb
import numpy as np
import vsm

path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_1\\'


def lgb_key_train():
    clabel =1
    data = vsm.vsmbuild(clabel)

    np.random.shuffle(data) # 打乱数据顺序
    print('data', data.shape)

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_classes': 4,
            'metric': 'multi_error',
            'num_leaves': 55,
            'learning_rate': 0.01,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'verbose': 1,
            # 'num_threads':4,
            }

    for i in range(5):
        print(i)
        train_data=np.concatenate([data[0:i*len(data)/5],data[(i+1)*len(data)/5:]])
        valid_data=data[i*len(data)/5:(i+1)*len(data)/5]
        train_d = lgb.Dataset(train_data[:,:-1], train_data[:,-1])
        valid_d = lgb.Dataset(valid_data[:, :-1], valid_data[:, -1])
        lis={}
        clf = lgb.train(params, train_d, evals_result=lis, num_boost_round=200000,
                        valid_sets=[valid_d], early_stopping_rounds=100, verbose_eval=10)
        clf.save_model(path+"models\\key_cv"+str(i)+".m")
        clf = lgb.Booster(model_file=path+"models\\key_cv"+str(i)+".m")
        # print clf.feature_importance()
        r=clf.predict(valid_data[:, :-1])
        for k in range(4):
            ct0=0
            ct1=0
            for j,v in enumerate(r):
                if np.where(v==max(v))[0]==k:
                    ct0+=1
                    if valid_data[j,-1]==k:
                        ct1+=1
            if ct0!= 0 :
                print(k,ct0,ct1,ct1*1.0/ct0)
            else:
                print(k,ct0,ct1,0)

# lgb_key_train()
