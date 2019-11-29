# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:12:12 2019

@author: Tyqnn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import modeling as bert_model
import optimizer as bert_op
import tokenization  # Bert源码中的模块
import re
import os





def softmax(X):
    return np.exp(X)/np.sum(np.exp(X))
class Bert_Classify:
    def __init__(self,config,Modelname,mini_batch=1, dropout_rate=0.1, learning_rate=2e-5, max_steps=30000,
                 warm_up_steps=10000, max_sequence_length=512):
        label_set = ["dcxz", "dkxz", "dtvxz", "fscpjb", "jz", "kdgz",
             "kdts", "kdwsm", "kdwzy", "khwl", "lw", "rhcf",
             "rhlw", "rhxz", "sjhx", "tcqy", "ydwlxhc", "yjqx",
             "zjxy", "znzw","others"]
        self.__mini_batch = mini_batch
        self.__keep_prob = 1.0 - dropout_rate
        self.__learning_rate = learning_rate
        self.__max_steps = max_steps
        self.__warm_up_steps = warm_up_steps
        self.__max_sequence_length = max_sequence_length

        self.__scene = Modelname
        self.__label_set = label_set
        self.__num_labels = len(label_set)
        self.__data_obj = DataObj(max_words=max_sequence_length, scene=self.__scene, label_set=label_set)

        self.__session = tf.Session(config=config)
        self.__input_ids_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="input_ids")
        self.__input_mask_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="input_mask")
        self.__input_seg_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="input_seg")
        self.__y_out_holder = tf.placeholder(dtype=tf.int32, shape=[None], name="y_out")
        self.__input_real_mask_holder = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length], name="real_mask")
        self.__is_training_holder = tf.placeholder(dtype=tf.bool, name="is_training")

    def load(self):
        bert_encoder_config = bert_model.BertConfig.from_json_file(
            json_file="./Chinese/chinese_L-12_H-768_A-12" + "/bert_config.json"
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

        self.__for_dev_tensor = tf.nn.softmax(eval_logits)


        # 加载预训练模型
        with self.__session.as_default():
            init_checkpoint = "./Chinese/chinese_L-12_H-768_A-12" + "/bert_model.ckpt"
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
        
        softpre=pre_y[0]
        max_label[0]=np.argmax(softpre)
        max_prob[0]=round(softpre[max_label[0]],2)
        softpre[max_label[0]]=0
        max_label[1]=np.argmax(softpre)
        max_prob[1]=round(softpre[max_label[1]],2)
        softpre[max_label[1]]=0
        max_label[2]=np.argmax(softpre)
        max_prob[2]=round(softpre[max_label[2]],2)
        softpre[max_label[2]]=0
        data_out=str(key)+'|'+self.__label_set[max_label[0]]+'|'+str(max_prob[0])+'|'+self.__label_set[max_label[1]]+'|'+str(max_prob[1])+'|'+self.__label_set[max_label[2]]+'|'+str(max_prob[2])+'\n'
        return data_out
    
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
            vocab_file="./Chinese/chinese_L-12_H-768_A-12" + "/vocab.txt", do_lower_case=False
        )
        self.__scene = scene
        self.__label_set = label_set
        self.__max_words = max_words
        self.__p = re.compile("(\d+_\d+_A_)|(\s+)|([？！。，])|([0-9]{8,})")
        self.__travel_index_train = 0  # 训练集的index
        self.__travel_index_dev = 0  # 验证集的index


    def get_dev_batch(self,text):

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
        dic = self.__encoding_data(text_list)
        dic["y_out"] = self.__encoding_y(label_list)
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


