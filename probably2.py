#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_classify
   Description :
   Author :       menghuanlater
   date：          2019/11/1
-------------------------------------------------
   Change Activity:
                   2019/11/1:
-------------------------------------------------
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import modeling as bert_model
import optimizer as bert_op
import tokenization  # Bert源码中的模块
import os
import re

scene = "Full model"
label_set = ["dcxz", "dkxz", "dtvxz", "fscpjb", "jz", "kdgz",
             "kdts", "kdwsm", "kdwzy", "khwl", "lw", "rhcf",
             "rhlw", "rhxz", "sjhx", "tcqy", "ydwlxhc", "yjqx",
             "zjxy", "znzw","others"]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(allow_soft_placement=True)

config.gpu_options.allow_growth = True
bert_source_dir = "./Chinese/chinese_L-12_H-768_A-12"
# bert_source_dir = "G:/Natural Language Processing/OpenSource_Deep_Contextual_Representation/Bert/Base"
project_dir = "./data"
# project_dir = "G:/河北电信项目/三期项目/Project/Multi-Label_Classification"
is_on_local = False

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X))
class Bert_Classify:
    def __init__(self, scene: str, label_set: list, mini_batch=1, dropout_rate=0.1, learning_rate=2e-5, max_steps=30000,
                 warm_up_steps=10000, max_sequence_length=512):
        self.__log_file = open(project_dir + "/log_Bert_Chinese.txt", "w", encoding="UTF-8")
        self.__mini_batch = mini_batch
        self.__keep_prob = 1.0 - dropout_rate
        self.__learning_rate = learning_rate
        self.__max_steps = max_steps
        self.__warm_up_steps = warm_up_steps
        self.__max_sequence_length = max_sequence_length

        self.__scene = scene
        self.__label_set = label_set
        self.__num_labels = len(label_set)
        self.__data_obj = DataObj(max_words=max_sequence_length, scene=scene, label_set=label_set)

        self.__session = tf.Session(config=config)
        self.__input_ids_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="input_ids")
        self.__input_mask_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="input_mask")
        self.__input_seg_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="input_seg")
        self.__y_out_holder = tf.placeholder(dtype=tf.int32, shape=[None], name="y_out")
        self.__input_real_mask_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="real_mask")
        self.__is_training_holder = tf.placeholder(dtype=tf.bool, name="is_training")

    def load(self):
        bert_encoder_config = bert_model.BertConfig.from_json_file(
            json_file=bert_source_dir + "/bert_config.json"
        )
        with tf.variable_scope("share", reuse=tf.AUTO_REUSE):
            bert_encoder_eval = bert_model.BertModel(
                config=bert_encoder_config, is_training=False,
                input_ids=self.__input_ids_holder, input_mask=self.__input_mask_holder,
                token_type_ids=self.__input_seg_holder,
                use_one_hot_embeddings=False, scope="bert"
            )
        bert_encoder_t_vars = tf.trainable_variables()

        eval_logits = self.__inference_model(cls_input=bert_encoder_eval.get_pooled_output(), sequence_input=bert_encoder_eval.get_sequence_output())

        self.__for_dev_tensor = eval_logits#tf.nn.softmax(eval_logits)


        # 加载预训练模型
        with self.__session.as_default():
            # saver.restore(self.__session, save_path=bert_source_dir + "/bert_model.ckpt")  # 再将bert预训练的参数导入计算图中
            init_checkpoint = bert_source_dir + "/bert_model.ckpt"
            assignment_map = bert_model.get_assignment_map_from_checkpoint(bert_encoder_t_vars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            self.__session.run(tf.global_variables_initializer())  # 先初始化所有参数


        saver = tf.train.Saver(max_to_keep=1)
        module_file = tf.train.latest_checkpoint("./Single_Label/%s/" % self.__scene)
        saver.restore(self.__session, module_file)

        
    # 验证集检测
    def outputdata(self,key,text):
        pre_y = []
        tru_y = []
        data_y = [] 
        max_label=[0,0,0]
        max_prob=[0.,0.,0.]
        dic = self.__data_obj.get_dev_batch(text)
#        while dic is not None:
        predict_out = self.__session.run(
            self.__for_dev_tensor, feed_dict={
                self.__input_ids_holder: dic["input_ids"],
                self.__input_mask_holder: dic["input_mask"],
                self.__input_seg_holder: dic["input_seg"],
                self.__input_real_mask_holder: dic["real_mask"],
                self.__is_training_holder: False,
            }
        )
        tru_y.extend(dic["y_out"].tolist())  # ground_truth
        pre_y.extend(predict_out.tolist())  # predict
        data_y.extend(dic["text"])
        
        softpre=softmax(pre_y[0])
        max_label[0]=np.argmax(softpre)
        max_prob[0]=round(softpre[max_label[0]],2)
        softpre[max_label[0]]=0
        max_label[1]=np.argmax(softpre)
        max_prob[1]=round(softpre[max_label[1]],2)
        softpre[max_label[1]]=0
        max_label[2]=np.argmax(softpre)
        max_prob[2]=round(softpre[max_label[2]],2)
        softpre[max_label[2]]=0
        for i in range(len(pre_y)):
            self.__log_file.write(str(key)+'|'+label_set[max_label[0]]+'|'+str(max_prob[0])+'|'+label_set[max_label[1]]+'|'+str(max_prob[1])+'|'+label_set[max_label[2]]+'|'+str(max_prob[2])+'\n')
            
            self.__log_file.flush()
            print(pre_y[i])
            print(softmax(pre_y[i]))
        print(key)
        print(text)
        return float(0)
    
    def __inference_model(self, cls_input, sequence_input):
        """
        :param cls_input: bert的CLS输出
        :param sequence_input: bert序列输出
        :return: 三logits, 一个是CLS加上一层线性变换的logits, 一个是sequence上加soft-attention的logits, 一个是贯穿sequence各维度max-pool的logits
                三者concat, 768*3， 再引入一层非线性变换, 降到768, 最后接一层分类层
                后续可以考虑门控机制的信息融入
        """
        # 针对[CLS]的输出做线性变换
        with tf.variable_scope("extend_cls", reuse=tf.AUTO_REUSE):
            cls = layers.fully_connected(inputs=cls_input, num_outputs=768, activation_fn=None)
        # 针对序列输出做soft-attention summary
        with tf.variable_scope("extend_sequence_summary", reuse=tf.AUTO_REUSE):
            dense_Ut = layers.fully_connected(inputs=sequence_input, num_outputs=768, activation_fn=tf.nn.tanh)
            vector_Uw = tf.get_variable(name="vector_for_attention_weights", shape=[768], dtype=tf.float32,
                                        initializer=layers.xavier_initializer(dtype=tf.float32))
            attention_logits = tf.reduce_sum(tf.multiply(dense_Ut, vector_Uw), axis=2)
            attention_logits = tf.exp(attention_logits)
            attention_logits_mask = tf.multiply(attention_logits, tf.cast(self.__input_real_mask_holder, dtype=tf.float32))
            attention_logits_mask_sum = tf.reduce_sum(attention_logits_mask, axis=1)
            attention_logits_mask_sum = tf.expand_dims(attention_logits_mask_sum, axis=1)  # 2-D
            attention_weights = tf.divide(attention_logits_mask, attention_logits_mask_sum)
            attention_weights = tf.expand_dims(attention_weights, axis=2)  # 3-D
            summary = tf.reduce_sum(tf.multiply(attention_weights, sequence_input), axis=1)
        # 针对序列输出做dimension-level的max-pool, CLS和SEP可以加入计算
        with tf.variable_scope("extend_sequence_max_pool", reuse=tf.AUTO_REUSE):
            pool = tf.reduce_max(sequence_input, axis=1)
            concat = tf.concat([cls, summary, pool], axis=1)

        with tf.variable_scope("ffn", reuse=tf.AUTO_REUSE):
            ffn = layers.fully_connected(inputs=concat, num_outputs=768, activation_fn=self.__gelu)
            ffn = layers.dropout(inputs=ffn, keep_prob=self.__keep_prob, is_training=self.__is_training_holder)

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            logits = layers.fully_connected(inputs=ffn, num_outputs=self.__num_labels, activation_fn=None)
        return logits

    # 自定义GELU激活函数 --> 替换ReLu
    @staticmethod
    def __gelu(x):
        """
        Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
            x: float Tensor to perform activation.

        Returns:
            `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    # 最优化操作
    def __train_op(self, loss):
        with tf.variable_scope("optimizer"):
            train_op = bert_op.create_optimizer(loss=loss, init_lr=self.__learning_rate, num_train_steps=self.__max_steps,
                                                num_warmup_steps=self.__warm_up_steps, use_tpu=False)
        return train_op

    # 损失函数
    @staticmethod
    def __losses(logits, ground_truth):
        with tf.variable_scope("loss"):
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=ground_truth, scope="my_loss")
        return loss

    # 评价函数-->训练的评价函数
    @staticmethod
    def __evaluation(logits, ground_truth):
        with tf.variable_scope("eval_acc"):
            logits = tf.cast(logits, dtype=tf.float32)
            correct = tf.nn.in_top_k(logits, ground_truth, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct, name="my_acc")
        return accuracy




class DataObj:
    def __init__(self, scene: str, label_set: str, max_words=512):
        """
        整体的设计为动态RNN以及Transformer等结构考虑, 方便移植
        """
        self.__tokenizer = tokenization.FullTokenizer(
            vocab_file=bert_source_dir + "/vocab.txt", do_lower_case=False
        )
        self.__scene = scene
        self.__label_set = label_set
        print(label_set)
        self.__max_words = max_words
        self.__p = re.compile("(\d+_\d+_A_)|(\s+)|([？！。，])|([0-9]{8,})")
        self.__travel_index_train = 0  # 训练集的index
        self.__travel_index_dev = 0  # 验证集的index


    def get_dev_batch(self,text):
#        if self.__travel_index_dev >= len(self.__all_test_data):
#            self.__travel_index_dev = 0
#            return None
#        batch_end = self.__travel_index_dev + batch_size
#        x = self.__all_test_data[self.__travel_index_dev: batch_end]
#        self.__travel_index_dev = batch_end
#
        text_list = []
        label_list = []
        lines=text
        lines=lines.replace('_','')
        lines=lines.replace('A','')
        lines=lines.replace('|','')
        lines=lines.replace('1','')
        lines=lines.replace('2','')
        lines=lines.replace('3','')
        lines=lines.replace('4','')
        lines=lines.replace('5','')
        lines=lines.replace('6','')
        lines=lines.replace('7','')
        lines=lines.replace('8','')
        lines=lines.replace('9','')
        lines=lines.replace('0','')
        text_list.append(lines)
        label_list.append('others')
        print(text_list)
        print(label_list)
        print("wenben")
        dic = self.__encoding_data(text_list)
        dic["y_out"] = self.__encoding_y(label_list)
        print(dic["y_out"])
        dic["text"]=text_list
        return dic

    def __encoding_data(self, text_list: list):
        """
        :param text_list: 输入的文本序列
        :return: batch_size; input_mask; segment_ids; input_ids
                由于不是多个文本交互, 所以segment_ids实际全0
                需要注意的是, 真实文本的tokens最多是max_words-2, 因为[CLS]和[SEP]必占据两位
                mask是填充的标记为0, 非填充的标记为1
        """
        batch_size = len(text_list)
        batch_segment_ids = [[0] * self.__max_words] * batch_size
        batch_input_ids = []
        batch_input_mask = []
        batch_real_mask = []
        for text in text_list:
            tokens = ["[CLS]"]
            if text[0] == "\"":
                text = text[1:]
            if text[-1] == "\"":
                text = text[:-1]
            text = re.sub(self.__p, "", text)
            init_tokens = self.__tokenizer.tokenize(text)
            tokens.extend(init_tokens[0:self.__max_words-2])
            tokens.append("[SEP]")
            # 编码成id
            ids = self.__tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(tokens)
            real_mask = [1] * len(tokens)
            real_mask[0] = real_mask[-1] = 0
            if len(ids) < self.__max_words:
                ids.extend([0] * (self.__max_words - len(ids)))
                mask.extend([0] * (self.__max_words - len(mask)))
                real_mask.extend([0] * (self.__max_words - len(real_mask)))
            batch_input_ids.append(ids)
            batch_input_mask.append(mask)
            batch_real_mask.append(real_mask)
        return {
            "batch_size": batch_size, "input_ids": np.array(batch_input_ids, dtype=np.int), "real_mask": np.array(batch_real_mask, dtype=np.int),
            "input_mask": np.array(batch_input_mask, dtype=np.int), "input_seg": np.array(batch_segment_ids, dtype=np.int)
        }

    def __encoding_y(self, label_list: list):
        y = []
        for t in label_list:
            y.append(self.__label_set.index(t))
        return np.array(y, dtype=np.int)


if __name__ == "__main__":


    data = {
        "73317f2f07774431b8ffe87c359f1feb|1807270000056891": "4450_5650_A_比。9410_10640_A_比。11610_17220_A_累哎您好内个先生是由s单连着线测速扣完了吗？18620_20880_A_我我这儿测一下哈。21530_25370_A_哦是最后测出来是多少呀看一下歌词！25860_26650_A_我正在弄。27080_28060_A_哦没弄完呢？28480_29510_A_没有查到是吗？30390_31050_A_幺儿呢？34720_37860_A_都接好了是吗就是光错了？39020_42300_A_我没弄完呢完了我。42620_63100_A_我一会儿我在在在在这廊我在特意我这可以很好也行那你过一会的话要测完或者值偏低的话才几十的话最少一个一百的或者色素扣划到二百兆的网速八是吧吃值低于一百的话您随时还让联系我们也可以这样报道语言包也行好吧？63690_66610_A_我掏心为国家牺牲没热再见！68190_68200_A_",
        "c03d66381a594c6caab0142efcf250f8|1807270000006271": "70_1940_A_！2860_6190_A_。8380_9490_A_哎！10060_10670_A_你好！10970_15520_A_到你们这里是中国电信我工号集中在在这宽带有问题了是吗？16540_20570_A_应亮了午夜我已经不鸟是怎么回事了？21080_22330_A_咱们地址是哪里？23600_24280_A_成员。25640_29100_A_什么原因相容身着时尚荣耀。30030_31490_A_深圳市生源。31960_34210_A_这是个小区是吗？34790_38030_A_这这边有别的交房嘛这个小区别的没有自由吗？38610_41620_A_没有商蝾螈伤蝾螈。42180_51660_A_收入元商业的桑我光荣地中这个提示您报修的账号和您说的地址不符。51990_55160_A_您名下就这一条宽带嘛是不是有别的宽带？55610_56920_A_就这一条。58090_60890_A_收入的人伤园小区。61280_62160_A_二号楼。62660_63390_A_一单元。64870_65900_A_。66260_69860_A_那你不重要的时候光猫有红灯吗？71280_75290_A_的时候猫上应该一直老三绿灯吗？75690_79030_A_没没没没红灯是就是就是不是按？79580_81790_A_不是不上的那个灯是亮的是吧？82110_83130_A_对网口那。83490_87440_A_两个话说那线路通着不上一般是因为没有数据传输？87870_93470_A_呃我这个的话也就是您的现在一直通着我们这儿也没有给您去修过的想。93810_95830_A_线路通里面数据不稳定？96150_101180_A_这种的话一般就是先重启一下重启不行就把路由器这重设置。103640_106580_A_对我就是我就是保了关了。107060_107830_A_我的一个。108220_115840_A_然后插到了以后插上它好啦是吧转操场呃方面量那我充了不行。116140_138410_A_应采用保保电话报保修回来呲白跑了拜托埃默这他这个的话一般现在都市有路由器上网的路由器里面他会自动进行拨号了才能上网像这种情况就是说他里边的数据不稳定拨不了号米大部分都出奇能好重启不行了一般就是重设置？140900_146810_A_重设置怎么设置而就是路由器里边有手机和电脑新路由器重输账号密码。149140_156270_A_呃这个得设那您这会我看看又恢复了您要不先关注着又在网的话您可以联系我们？156890_161030_A_诶好的行打扰你请稍后服务评价再见！161840_161850_A_",
        "75fdb8dadef14dd3b4655f23c4df679b|1807270000006167": "70_1080_A_！3060_6040_A_好。6930_10860_A_哎你好你好很高兴为您这是上不了网是吧？11560_12340_A_对。12870_15570_A_现在咱们这光猫指示灯是怎么样的呢我说一下？16270_18940_A_主要是那个被子那个猫什么？19670_20320_A_对。20930_22250_A_是绿色的。22830_27080_A_好您看她那张旁边一般会写这些数字姆汉森您这儿有吗？29010_29590_A_！29910_31350_A_业者的是吗尤？31920_33390_A_帮您看一下他写的是什么？34710_35340_A_行哦。35770_38500_A_开了。40360_41960_A_！43660_45030_A_哈！45390_50620_A_行显示故障。51420_53640_A_报修电话是吗？53970_58540_A_不是报修电话就看一下他上面写的那个字母汉字是什么灯旁边的。61650_62960_A_刚刚捐没有？63980_64580_A_没有。65800_68730_A_哎好恐怖。70440_76370_A_说书者你是说后面那个就跟着老师的帖的人。76880_80520_A_由中国电信嘛结果号码还是然后再买。81350_85940_A_他上面有没有写着中国电信天翼宽带或者天翼网关又写着的嘛？88680_89420_A_没有。90400_91240_A_这也没写。91930_93190_A_我看看时候不用。93790_94600_A_好嘞。95470_96440_A_对呀。100620_101820_A_行。102490_106060_A_没有双眼泪手机的玩的是那个。106370_107090_A_帖子。108670_111160_A_这么天翼宽带天翼网关都没写着是吗？112040_112790_A_！113400_114710_A_没有看见。116080_118020_A_哦网关。118570_120540_A_数控网管什么的。121350_127480_A_他亲眼观的这个的话是查一下他说的是不是有一个咱们电信标志一样当有吗？129030_134260_A_哦就是侧面有个电信标志那个的您要开那个一个呀这个号码是吗？134710_141210_A_两毛钱看一下他不是下载一个二维码二维码线嘛还有个灯又常关了灯这个亮吗？141690_142290_A_郭玲。142720_143780_A_中间的两。144840_145750_A_中间的两。146290_146950_A_好的。147690_149450_A_哦就把信号的那个是吧？149750_150430_A_对。151350_153920_A_行那就不对了这个猫重启过吗？154240_158460_A_铝好重启过了拉行么行行行我知道。158760_162950_A_那您这地址是在什么位置悲哀小学这边。164000_165070_A_对杨庄小学。165440_167910_A_行那您稍等我给你找人过去看一下。168580_169720_A_哦行行行。170040_173570_A_好那我不打扰您稍后服务做评价好嘞恩？174040_174050_A_",
        "48b823b558ae432cb82371e9e7117460|1807270000006276": "2000_3580_A_哦比。4100_4640_A_哎你好！5070_9880_A_您好打扰了中国电信我工号决赛中反映宽带有问题了是吗？10600_11960_A_对这两天总是掉线。12560_13910_A_咱们地址是哪里？14610_15660_A_就是。16030_18160_A_文化中心六号地上国际化。18970_24080_A_婆婆出气儿这几天老断带三的时候光猫上那个登录变化吗？24660_26060_A_红灯。26880_28520_A_有红灯断线。29100_34130_A_对能有时候断了半个小时有一个多小时之后他最近有好好儿。35740_39400_A_而屡你稍等给您好几天了？39920_40640_A_诶。41270_44460_A_猫上得需要您自己检查过是吗确实查询了？45700_48170_A_没问题他因为他自己过一会儿自己又好了。48740_49320_A_对对。49670_51820_A_但是我通话这个月轮。52220_56810_A_光信号又是亮红灯我知道客那个光信号只有光信号才会有红灯。57190_64770_A_这个的话一般是线路问题他又自己好了也就是说您看一下猫上了一条光纤线确实查询了吗？65160_70600_A_有没有可能是他说没听到吗他不行的话他不可能我们都冻帽子和那？71280_75020_A_谋杀都熟悉了如果是说不是说得我得现吗？76040_79530_A_鬼行那您这个维修员跟您联系。80250_83110_A_行好勒好勒那那再见！84740_84750_A_"
    }
    M = Bert_Classify(max_sequence_length=512, label_set=label_set, scene=scene)
    M.load()  
    for k, v in data.items():
        M.outputdata(k,v)
        
