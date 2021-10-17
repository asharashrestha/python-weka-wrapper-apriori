from os import replace
from google.protobuf.symbol_database import Default
import pandas as pd
import sys
import traceback
import numpy as np
import weka.core.jvm as jvm
import helper as helper
from weka.core.converters import Loader
from weka.associations import Associator
import javabridge
from javabridge import JWrapper
import re
from itertools import permutations
from treelib import Node, Tree
from statistics import  mean, median
import streamlit as st
import os
import execute_sql as db
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from datetime import datetime

start_time = datetime.now()
print("Start Time: ", start_time)

st.set_page_config(layout="wide")
rule_parameter = ""
var_seq_order = dict()
def is_order(lst):
    if lst == sorted(lst, reverse=False):
        return 'Ascending'
    elif lst == sorted(lst, reverse=True):
        return 'Descending'
    else:
        return 'Neither ascending nor descending'


def perm_elems(lst):
    p_list = []
    perm_list = permutations(lst, len(lst))
    for i in perm_list:
        p_list.append(', '.join(i))
    return p_list
    
def calculate_asc_consistency_ratio(conf_seq):
    index = 0
    sum_asc = 0
    ascend_count = 0
    while index + 1 < len(conf_seq):
        b = conf_seq[index + 1] 
        a = conf_seq[index]
        if b - a >= 0:
            sum_asc += b-a
            ascend_count+=1
        index += 1
    return sum_asc / ascend_count

def mean_asc_change(conf_seq):
    index = 0
    ap = 0
    while index + 1 < len(conf_seq):
        if conf_seq[index + 1] - conf_seq[index] >= 0:
            ap += 1
        index += 1
    return ap / (len(conf_seq) - 1)
    
#displays tree after user selects number of attributes
def show_tree(num_of_attr):
    rows = db.extract_rules(num_of_attr)
    for i in rows:
        st.markdown(i[0].replace("└──","\n└──"))

# create Tree structure for blocks of rules to show rule progression.
def createRulesTree(num_attr, df, r_param):
    df.to_csv("Dataframe.csv")
    
    for index, row in df.iterrows():
        if "," not in row['LHS']: # to start with root node having only two attributes
            lhs = row['LHS']
            rhs = row['RHS']
            conf = row['Conf']

            tree = Tree()
            rules_queue = []
            tree.create_node(lhs + "->" + rhs + "[Conf: " + conf + "]", "root")  # root node
            l_list = lhs.split(",")
            l_list.append(rhs)
            a = perm_elems(l_list)
            for index, row in df.iterrows():
                if any(row['LHS'] == elem for elem in a):
                    left = row['LHS']
                    right = row['RHS']
                    a = left + ", " + right
                    tree.create_node(left + "->" + right + "[Conf:" + row["Conf"] + "]", a, parent="root")
                    rules_queue.append(left + ", " + right)

            while len(rules_queue) != 0:
                elem = rules_queue.pop(0)
                for index, row in df.iterrows():
                    if (row['LHS'] == elem):
                        left = row['LHS']
                        right = row['RHS']
                        a = left + ", " + right  # concatenating lhs and rhs to find the result in lhs of dataframe
                        tree.create_node(left + "->" + right + "[Conf:" + row["Conf"] + "]", a, parent=left)
                        rules_queue.append(left + ", " + right)
            # tree.show(line_type="ascii-em")
            # tree.save2file("Result_Tree.txt")
            # tree.to_graphviz("graph", shape="circle", graph='digraph')
            # tree.to_json(with_data=True)
            # tree.save2file("Result_Tree.txt")
            st.text_area("RULE: ", tree)

            for path in tree.paths_to_leaves():
                rule_sequence = ""
                conf_sequence = ""
                left_attr = ""
                for i in path:
                    
                    a = tree.get_node(i).tag.replace("'","")
                    # special_characters = '[[|]|(|)]'  # removing "[", "]" from confidence
                    # conf = re.sub(special_characters, '', a.split("Conf:")[1])
                    # conf = conf.strip().replace("(", "").replace(")", "")
                    right = a.split("[Conf:")[0].split("->")[1].split("=")[0]
                    left = a.split("->")[0].split(",")
                    left_attr = ", ".join(l.strip().split("=")[0] for l in left)
                    conf = a.split("[Conf:")[1].replace("]","").strip()
                    if rule_sequence == "":
                        rule_sequence = left_attr + " -> " + right
                        conf_sequence = conf
                    else:
                        rule_sequence += " -> " + right
                        conf_sequence += " -> " + conf
                st.text("Rule Sequence: " + rule_sequence + "\n")
                st.text("Confidence Sequence: " + conf_sequence + "\n")
                
                conf_seq_list = [float(i) for i in conf_sequence.split("->")]
            
                # db.insert_tree_db(str(tree).replace("'",""),rule_sequence.replace("'",""), conf_sequence)  
                # r_param = "Mean Confidence"
                # When parameter = Mean Confidence
                if r_param == "Occurence":
                    # if len(conf_seq_list) == 1:
                    #     if rule_sequence in var_seq_order.keys():
                    #         var_seq_order[rule_sequence]+=1
                    #     else:
                    #         var_seq_order[rule_sequence] = 1
                    
                    # if len(conf_seq_list) == 2:
                    # high = float(conf_sequence.split("->")[-1])
                    # low = float(conf_sequence.split("->")[0])
                    if rule_sequence in var_seq_order.keys():
                        var_seq_order[rule_sequence]+=1
                    else:
                        var_seq_order[rule_sequence] = 1
                if r_param == "Mean Confidence":
                    if len(conf_seq_list) == 1:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append(conf_seq_list[0])
                        else:
                            var_seq_order[rule_sequence] = [conf_seq_list[0]]
                    high = float(conf_sequence.split("->")[-1])
                    low = float(conf_sequence.split("->")[0])
                    # if (high - low >=0):
                    if len(conf_seq_list) == 2:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append(high - low)
                        else:
                            var_seq_order[rule_sequence] = [high-low]

                    if len(conf_seq_list) > 2:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append( high-low)
                        else:
                            var_seq_order[rule_sequence] = [high-low]


                if r_param == "Ascendingness Consistency Ratio":
                    if len(conf_seq_list) == 1:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append(conf_seq_list[0])
                        else:
                            var_seq_order[rule_sequence] = [conf_seq_list[0]]
                    high = float(conf_sequence.split("->")[-1])
                    low = float(conf_sequence.split("->")[0])
                    # if (high - low >=-5000):
                    if len(conf_seq_list) == 2:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append(1)
                        else:
                            var_seq_order[rule_sequence] = [1]

                    if len(conf_seq_list) > 2:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append(calculate_asc_consistency_ratio(conf_seq_list))
                        else:
                            var_seq_order[rule_sequence] = [calculate_asc_consistency_ratio(conf_seq_list)]

                if r_param == "Mean Ascendingness Change":
                    if len(conf_seq_list) == 1:
                        if rule_sequence in var_seq_order.keys():
                            var_seq_order[rule_sequence].append(conf_seq_list[0])
                        else:
                            var_seq_order[rule_sequence] = [conf_seq_list[0]]

                    high = float(conf_sequence.split("->")[-1])
                    low = float(conf_sequence.split("->")[0])

                    if high - low >=0:
                        if len(conf_seq_list) == 2:
                            if rule_sequence in var_seq_order.keys():
                                var_seq_order[rule_sequence].append(high - low)
                            else:
                                var_seq_order[rule_sequence] = [high - low ]

                        if len(conf_seq_list) > 2:
                            if rule_sequence in var_seq_order.keys():
                                var_seq_order[rule_sequence].append(mean_asc_change(conf_seq_list))
                            else:
                                var_seq_order[rule_sequence] = [mean_asc_change(conf_seq_list)]
    
    order_mean_dict = dict()
    if (len(var_seq_order) > 0):
    #     max_order_freq = max(var_seq_order, key=var_seq_order.get)
    #     st.text("Order with highest frequency is: " + "\n \t" + max_order_freq)
        if r_param == "Occurence":
            st.text("Length of dictionary: " + str(len(var_seq_order)))
            st.text("Frequency of different orders:")
            for k in var_seq_order.keys():
                st.text("Order: " + str(k) + ": " + str(var_seq_order[k]))
        else:
            for k in var_seq_order.keys():
                print("\t Mean: " + k + " => " +  str(mean(var_seq_order[k])))
                order_mean_dict[k] = str(mean(var_seq_order[k]))
            print("\n")
            mean_dict_num_attr = dict()
            st.header("Mean for different sequences: ")
            for k in order_mean_dict.keys():
                if len(k.split("->"))==num_attr:
                    mean_dict_num_attr[k] = order_mean_dict[k]
                    st.text(str(k) + "=> " + str(order_mean_dict[k]))
            if len(mean_dict_num_attr) == 0:
                st.text("No rules of length " + str(num_attr) + " was found.")
            max_order = max(mean_dict_num_attr, key=mean_dict_num_attr.get)
            st.text("Order with highest mean is: " + "\n \t" + max_order)
        # select_attr = [i+1 for i in range(num_attr)]
        # # st.sidebar.selectbox("Choose number of atributes", [int(i) for i in range(0,num_of_attr)])
        # attr = st.sidebar.selectbox("Choose number of atributes", (select_attr))
        # show_tree(attr)
        # os.remove("output.txt")
    # if (len(var_seq_order) > 0):
    #     for k in var_seq_order.keys():
    #         print("\t Mean: " + k + " => " +  str(mean(var_seq_order[k])))
    #         order_mean_dict[k] = str(mean(var_seq_order[k]))
    #     print("\n")
    #     mean_dict_num_attr = dict()
    #     st.header("Mean for different sequences: ")
    #     for k in order_mean_dict.keys():
    #         if len(k.split("->"))==num_attr:
    #             mean_dict_num_attr[k] = order_mean_dict[k]
    #             st.text(str(k) + "=> " + str(order_mean_dict[k]))

    #     print("Length of order_mean_dict is : ", len(order_mean_dict))
    #     print("Length of mean_dict_num_attr is : ", len(mean_dict_num_attr))
    #     max_order = max(mean_dict_num_attr, key=mean_dict_num_attr.get)
    #     st.text("Order with highest mean is: " + "\n \t" + max_order)
    #     select_attr = [i+1 for i in range(num_attr)]
    #     # st.sidebar.selectbox("Choose number of atributes", [int(i) for i in range(0,num_of_attr)])
    #     attr = st.sidebar.selectbox("Choose number of atributes", (select_attr))
    #     show_tree(attr)
    #     # os.remove("output.txt")
    else:
        print("No rule block with 3 attributes are present")
        st.text("No rule block with 3 attributes are present with given support and confidence threshold")

def run_Apriori(num_attr, data, data_file, sup, conf, r_param):

    #FP 
    df= pd.read_csv(data_file)
    df = pd.read_csv(data_file, dtype=str)
    transactions = []
    for sublist in df.values.tolist():
        clean_sublist = [item for item in sublist if item is not np.nan]
        transactions.append(clean_sublist)

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    if sup == 0:
        sup = 0.00000000000000001
    frequent_itemsets_fp=fpgrowth(df, min_support=sup, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=conf)
    rules_fp_3cols = rules_fp[["antecedents","consequents","confidence"]]
    
    #Converting data type of all rows into string
    rules_fp_3cols["antecedents"]=rules_fp_3cols["antecedents"].apply(str)
    rules_fp_3cols["consequents"]=rules_fp_3cols["consequents"].apply(str)
    rules_fp_3cols["confidence"]=rules_fp_3cols["confidence"].apply(str)
    # deleting rows where there are more than 1 element in RHS:
    df_RHS_1item = rules_fp_3cols[~rules_fp_3cols['consequents'].str.contains(',')]
    df_rules = df_RHS_1item.rename({'antecedents': 'LHS', 'consequents': 'RHS', 'confidence':'Conf'}, axis=1)  # renaming columns

    df_rules = df_rules.apply(lambda col: col.str.replace('frozenset', ''))
    df_rules = df_rules.apply(lambda col: col.str.replace("\(\{", ""))
    df_rules = df_rules.apply(lambda col: col.str.replace("\}\)", ""))

    createRulesTree(num_attr, df_rules, r_param)
def main(args):
    # Streamlit Sidebar and dashboard
    st.sidebar.write("Sidebar")
    support_threshold = [i for i in range(0, 101, 1)]
    parameters = ["Occurence","Mean Confidence","Ascendingness Consistency Ratio","Mean Ascendingness Change"]
    rule_parameter = st.sidebar.selectbox("Choose Rules Parameter", (parameters))
    # support = st.sidebar.selectbox("Support Threshold", support_threshold)
    support = st.sidebar.select_slider("Select Support Threshold", options = support_threshold)

    support = support / 100

    conf_threshold = [i for i in range(0, 101, 1)]
    confidence = st.sidebar.select_slider("Confidence Threshold", conf_threshold)
    confidence = confidence / 100
    # support = 0.005
    # confidence = 0.10
    # rule_parameter = "Occurence"
    data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/Arff Dataset"

    data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/CSV"
    list_of_files = [f for f in os.listdir(data_folder)]
    filename = st.sidebar.selectbox("Select source data", list_of_files)
    data_file = data_folder + "/" + filename
    attr_num = 3
    # data_file = data_folder + "/" + "Grouped.csv"
    # data_file = data_folder + "/" +  "test_FPGrowth.csv"
    # data_file = data_folder + "/" +  "Inpatient_Claims_PUF.csv"
    # data_file = data_folder + "/" +  "3_attr_include_notrt_noresp.csv"
    st.text("Data File: " + data_file.split("/")[-1])
    st.text("Rule Parameter: " + rule_parameter)
    st.text("Support: " + str(support))
    st.text("Confidence: " + str(confidence))
    print("Datafile: ", data_file)

    # loader = Loader("weka.core.converters.ArffLoader")
    # data = loader.load_file(data_file)
    # attr_num = data.num_attributes
    data = ""
    
    run_Apriori(attr_num, data,data_file, support, confidence, rule_parameter)

if __name__ == "__main__": 
    try:
        jvm.start()
        main(sys.argv)

    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
        end_time = datetime.now()
        print(end_time)
        print('Duration: {}'.format(end_time - start_time))