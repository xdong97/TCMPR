 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/7 14:59
# @Author  : dx
# @File    : TCMPR_model.py
# @Software: PyCharm
# @Note    :
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers.core import Dense
from keras import models
from gensim.models import Word2Vec
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import MaxPooling1D, AveragePooling1D
import networkx as nx
from sklearn.model_selection import train_test_split


class TcmprModel(object):
    def __init__(self, recorded_pro=True, degree_filter=2):
        self.main_path = 'D:/Work/Project/tcmpr_demo'
        self.recorded_flag = recorded_pro
        self.degree_flag = degree_filter
        self.symptom_network, self.embed_list, self.symptom_network_data = self.load_data()

    def load_data(self):
        """
        load related data
        :return: embedding, embed_list(), network
        """
        embedding = Word2Vec.load(self.main_path + '/data/Symptom_Embedding_200.model')
        embed_list = embedding.wv.index_to_key
        network = pd.read_csv(self.main_path + '/data/Symptom_network.txt')
        return embedding, embed_list, network

    def get_entity_from_graph(self, graph: nx.Graph):
        """
        from graph to entity list with filtered degree
        :param graph: connected graph
        :return: filtered entity list
        """
        degree_dict = dict(graph.degree)
        temp_list = []
        temp_list_append = temp_list.append
        for node in degree_dict.keys():
            # 20211130 add degree flag
            if degree_dict[node] > self.degree_flag:  # degree filter
                temp_list_append(node)
        return temp_list

    def sstm_filter(self, symptom_list):
        """
        Subnetwork-based Symptom Term Mapping (SSTM)
        input: Symptom list
        :return: SSTMed Symptom list
        """
        # sl = ['手脚麻木', '手酸疼', '腰膝酸软']
        result_sstm_for_list = []
        for symptom in symptom_list:
            # 20211130 judge
            if self.recorded_flag is True and symptom in self.embed_list:
                # print(symptom, 'in w2v')
                result_sstm_for_list.append(symptom)
            else:
                # 1. Symptom list -> symptom word set list
                symptom_word = set([word for word in symptom])
                # print(symptom_word)  # {'麻', '木', '手', '脚'}

                # 2. For set, query edges in symptom df, and construct graph
                subgraph_df = self.symptom_network_data.query('Source in @symptom_word or Target in @symptom_word')
                if len(subgraph_df) > 0:
                    subgraph = nx.from_pandas_edgelist(subgraph_df, 'Source', 'Target')

                    # 3. for "connected graph", get the symptom list with nodes' degree > degree_filter
                    if nx.is_connected(subgraph):  # connected graph
                        result_sstm_for_list += self.get_entity_from_graph(subgraph)
                    else:  # not connected graph
                        # print(f'Not Connected graph of {symptom} !')
                        extract_networks = sorted(nx.connected_components(subgraph), key=len, reverse=True)
                        for subnetwork in extract_networks:
                            graph_sub = subgraph.subgraph(subnetwork)
                            assert nx.is_connected(graph_sub) == True
                            result_sstm_for_list += self.get_entity_from_graph(graph_sub)
                else:
                    print(f"The symptom '{symptom}' doesn't have its subgraph in symptom network.")

        result = list(set(result_sstm_for_list))
        # print(len(result), len(result_sstm_for_list))
        return result

    def form_symptom_feature(self, sample_df: pd.DataFrame, symptom_maximum=10):
        """
        form embedding for symptom (both train and test)
        :param symptom_maximum:
        :param sample_df: train/test dataframe
        :return:
        """
        embedding_list = []

        for row in sample_df.itertuples():
            try:
                temp_symptom = getattr(row, 'Symptom').split(';')  # att!
                temp_sstm_list = self.sstm_filter(temp_symptom)  # The core step
                temp_embedding = [
                    list(self.symptom_network.wv[symptom]) for symptom in temp_sstm_list if symptom in self.embed_list
                ]
                temp_symptom_len = len(temp_embedding)
                # Filtering with max_symptom_number
                if temp_symptom_len >= symptom_maximum:
                    # If symptom number is sufficient, then slice the symptom embedding list.
                    embedding_list.append(temp_embedding[:symptom_maximum])
                else:
                    # If symptom number isn't sufficient, then padding.
                    need_symptom_num = symptom_maximum - temp_symptom_len
                    # Padding what? padding zero vectors.
                    temp_embedding += [200 * [0.0] for _ in range(need_symptom_num)]
                    embedding_list.append(temp_embedding)
                    assert len(temp_embedding) == symptom_maximum

            except AssertionError:
                print(row)
                continue

        return np.array(embedding_list)

    @staticmethod
    def form_herb_feature(df: pd.DataFrame):
        """
        form the one-hot feature matrix for herb
        :param train_df: train data, pd.Dataframe
        :param test_df: test data, pd.Dataframe
        :return:
        Note: combine train and test data, then ont-hot embedding! (For test herb)
        """
        # change column name
        df.columns = ['Index', 'Symptom', 'Herb']

        # obtain all herb data in dataset, then form the herb list.
        total_list = df['Herb'].tolist()

        total_herb_sample = list()
        for herb in total_list:
            temp_herb = herb.split(';')  # split herbs
            total_herb_sample += temp_herb

        herb_list = list(set(total_herb_sample))  # drop duplicates
        herb_amount = len(herb_list)
        print('Herb amount:', herb_amount)  # Herb amount, 2827

        # form herb one-hot embedding
        embedding_list = []
        embedding_list_append = embedding_list.append
        for herb_str in total_list:
            split_herb = herb_str.split(';')
            herb_mid = len(herb_list) * [0]  # initialize, zero vector
            # for the actual data, set the corresponding value to 1
            for herb in split_herb:
                herb_mid[herb_list.index(herb)] = 1
            embedding_list_append(herb_mid)

        return np.array(embedding_list), herb_list, herb_amount

    @staticmethod
    def training_and_testing(x_train, y_train, x_test, herblong, max_sym_num, symlong, fusion, layer1, layer2):
        """
        training model for x_train, y_train and predict for x_test
        :param x_train:
        :param y_train:
        :param x_test:
        :param herblong:
        :param max_sym_num:
        :param symlong:
        :param fusion:
        :param layer1:
        :param layer2:
        :return:
        """
        print("/------------- Training and Predicting --------------/")
        model = models.Sequential()
        model.add(Conv1D(filters=10, kernel_size=2, padding='valid', kernel_initializer='uniform', strides=1))
        if fusion == 'Avg':
            model.add(AveragePooling1D())
        elif fusion == 'Max':
            model.add(MaxPooling1D())
        else:
            pass
        model.add(Flatten())
        model.add(Dense(layer1, activation="relu"))
        model.add(Dense(layer2, activation="relu"))
        model.add(Dense(herblong, activation="softmax"))
        model.build(input_shape=(2, max_sym_num, symlong))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        model.fit(x=x_train, y=y_train, epochs=50, shuffle=True, batch_size=2)
        predictions = model.predict(x_test)
        return predictions

    def evaluate(self, predictions, y_test, herb_number, top_k=20):
        """
        evaluate Top@K metric with y_test and prediction result
        :param predictions:
        :param y_test:
        :param herb_number:
        :param top_k:
        :return:
        """
        print("/------------- Evaluate --------------/")
        final_pre = []
        result_df = pd.DataFrame(columns=['k', 'Precision', 'Recall', 'F1_score'])
        for pre in predictions:
            final_pre.append(np.argsort(-pre))
        for k in range(1, top_k+1):
            pr, re, f1 = 0, 0, 0
            for i in range(len(y_test)):
                check_list = list(final_pre[i])[0:k]
                all_num_y = 0
                all_num_p = 0
                count = 0
                check_ = []
                for j in range(herb_number):
                    if y_test[i][j] == 1:
                        check_.append(j)
                        all_num_y += 1
                    if j in check_list:
                        all_num_p += 1
                        if y_test[i][j] == 1:
                            count += 1
                pr += count / all_num_p
                re += count / all_num_y
            pr = pr / len(y_test)
            re = re / len(y_test)
            f1 = 2 * pr * re / (pr + re)
            print(str(k), ':', pr, ':', re, ':', f1)
            result_df.loc[len(result_df.index)] = [k, pr, re, f1]

        result_df.to_excel(self.main_path+"/result/Evaluation.xlsx", index=False)

    def main(self):
        """
        prepare training
        :return:
        """
        # 1. load experimental data
        print('/------------ 1. load experimental data ------------/')
        df = pd.read_excel(self.main_path + '/data/input_example.xlsx')

        # 2. form feature and label for training set and testing set
        print('/------------ 2. form feature and label ------------/')
        # form y_train and y_test embedding
        herb_embedding, herb_list, herb_amount = self.form_herb_feature(df)

        # form symptom feature
        max_symptom = 10
        features = self.form_symptom_feature(df)
        x_train, x_test, y_train, y_test = train_test_split(features, herb_embedding, test_size=0.2, random_state=2022)

        # 3. training
        print('/--------------- 3. Training and Testing ---------------/')
        symptom_dimension = 200
        fusion_way = 'Avg'
        layer1 = 256
        layer2 = 64

        predictions = self.training_and_testing(
            x_train, y_train, x_test, herb_amount, max_symptom, symptom_dimension, fusion_way, layer1, layer2
        )

        # 4. evaluate
        print('/-------------------- 4. Evaluating --------------------/')
        self.evaluate(predictions, y_test, herb_amount)


if __name__ == '__main__':
    tcmpr = TcmprModel()
    tcmpr.main()

