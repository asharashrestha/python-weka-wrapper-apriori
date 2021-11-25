import pandas as pd
import sys
import traceback
import numpy as np
import weka.core.jvm as jvm
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
var_seq_order_lift = dict()
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
    dict_rules = dict()
    
    for index, row in df.iterrows():
        if "," not in row['LHS']: # to start with root node having only two attributes
            lhs = row['LHS']
            rhs = row['RHS']
            conf = round(float(row['Conf']),2)
            lift = round(float(row['Lift']), 2)
            rhs_sup = round(float(row['rhs_sup']), 2)

            #Using Certainy Factor
            if conf > rhs_sup:
                cf = (conf -  rhs_sup)/(1-rhs_sup)
                cf = round(cf,4)
            elif conf < rhs_sup:
                cf = (conf - rhs_sup)/rhs_sup
                cf = round(cf,4)
            else:
                cf = 0


            tree = Tree()
            rules_queue = []
            if cf > 0:
                tree.create_node(lhs + "->" + rhs + "[Conf: " + str(conf) + "]; " + "[Lift: " + str(lift) + "]" + "[CF: " + str(cf) + "]"   , "root")  # root node
                l_list = lhs.split(",")
                l_list.append(rhs)
                a = perm_elems(l_list)
                for index, row in df.iterrows():
                    if any(row['LHS'] == elem for elem in a):
                        left = row['LHS']
                        right = row['RHS']
                        conf = round(float(row['Conf']),2)
                        lift = round(float(row['Lift']), 2)
                        if conf > rhs_sup:
                            cf = (conf -  rhs_sup)/(1-rhs_sup)
                            cf = round(cf,4)
                        elif conf < rhs_sup:
                            cf = (conf - rhs_sup)/rhs_sup
                            cf = round(cf,4)
                        else:
                            cf = 0
                        a = left + ", " + right
                        if cf>0:
                            tree.create_node(left + "->" + right + "[Conf:" + str(conf) + "]" + "[Lift:" + str(lift) + "]"+ "[CF:" + str(cf) + "]", a, parent="root")
                            rules_queue.append(left + ", " + right)

                while len(rules_queue) != 0:
                    elem = rules_queue.pop(0)
                    for index, row in df.iterrows():
                        if (row['LHS'] == elem):
                            left = row['LHS']
                            right = row['RHS']
                            conf = round(float(row['Conf']),2)
                            lift = round(float(row['Lift']), 2)
                            if conf > rhs_sup:
                                cf = (conf -  rhs_sup)/(1-rhs_sup)
                                cf = round(cf,4)
                            elif conf < rhs_sup:
                                cf = (conf - rhs_sup)/rhs_sup
                                cf = round(cf,4)
                            else:
                                cf = 0 
                            a = left + ", " + right  # concatenating lhs and rhs to find the result in lhs of dataframe
                            if cf > 0:
                                tree.create_node(left + "->" + right + "[Conf:" + str(conf) + "]"+ "[Lift:" + str(lift) + "]"+ "[CF:" + str(cf) + "]", a, parent=left)
                                rules_queue.append(left + ", " + right)

                st.text_area("RULE: ", tree)

                for path in tree.paths_to_leaves():
                    rule_sequence = ""
                    conf_sequence = ""
                    left_attr = ""
                    for i in path:
                        
                        a = tree.get_node(i).tag.replace("'","")
                        right = a.split("[Conf:")[0].split("->")[1].split("=")[0]
                        left = a.split("->")[0].split(",")
                        left_attr = ", ".join(l.strip().split("=")[0] for l in left)
                        conf = a.split("[Conf:")[1].split("]")[0].strip()
                        lift = a.split("Lift:")[1].split("]")[0].strip()
                        if rule_sequence == "":
                            rule_sequence = left_attr + " -> " + right
                            conf_sequence = conf
                            lift_sequence = lift
                        else:
                            rule_sequence += " -> " + right
                            conf_sequence += " -> " + conf
                            lift_sequence += " -> " + lift
                    st.text("Rule Sequence: " + rule_sequence + "; Confidence Sequence: " + conf_sequence + "; Lift Sequence: " + lift_sequence +"\n")
                    # st.text("Confidence Sequence: " + conf_sequence + "\n")
                    # st.text("Lift Sequence: " + lift_sequence + "\n")
                    k = rule_sequence
                    v = conf_sequence
                    if k in dict_rules.items():
                        dict_rules.append(v)
                    else:
                        dict_rules[k] = [v]
                    
                    conf_seq_list = [float(i) for i in conf_sequence.split("->")]
                    lift_seq_list = [float(i) for i in lift_sequence.split("->")]
                
                    # db.insert_tree_db(str(tree).replace("'",""),rule_sequence.replace("'",""), conf_sequence)  
                    # r_param = "Mean Confidence"
                    # When parameter = Mean Confidence
                    if r_param == "Mean Confidence":
                        if len(conf_seq_list) == 1:
                            if rule_sequence in var_seq_order.keys():
                                var_seq_order[rule_sequence].append(conf_seq_list[0])
                            else:
                                var_seq_order[rule_sequence] = [conf_seq_list[0]]

                            if rule_sequence in var_seq_order_lift.keys():
                                var_seq_order_lift[rule_sequence].append(lift_seq_list[0])
                            else:
                                var_seq_order_lift[rule_sequence] = [lift_seq_list[0]]
                        high = float(conf_sequence.split("->")[-1])
                        low = float(conf_sequence.split("->")[0])

                        high_lift = float(lift_sequence.split("->")[-1])
                        low_lift = float(lift_sequence.split("->")[0])

                        if (high - low)>=0 and (high_lift - low_lift) >= 0:
                            if len(conf_seq_list) == 2:
                                if rule_sequence in var_seq_order.keys():
                                    var_seq_order[rule_sequence].append(round((high - low),2))
                                else:
                                    var_seq_order[rule_sequence] = [round((high-low),2)]
                                if rule_sequence in var_seq_order_lift.keys():
                                    var_seq_order_lift[rule_sequence].append(round((high - low),2))
                                else:
                                    var_seq_order_lift[rule_sequence] = [round((high-low),2)]
    
    order_mean_dict = dict()
    order_mean_dict_lift = dict()
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
                print("\t Mean Conf: " + k + " => " +  str(mean(var_seq_order[k])))
                order_mean_dict[k] = str(mean(var_seq_order[k]))
            print("\n")

            for k in var_seq_order_lift.keys():
                print("\t Mean Lift: " + k + " => " +  str(mean(var_seq_order_lift[k])))
                order_mean_dict_lift[k] = str(mean(var_seq_order_lift[k]))
            print("\n")


            mean_dict_num_attr = dict()
            st.header("Mean for different sequences: ")
            if num_attr ==0:
                for k in order_mean_dict.keys():
                    if len(k.split("->")) > 2:
                        st.text(str(k) + "=> " + "Conf Difference: "+ str(order_mean_dict[k]) + " Lift Difference: "+ str(order_mean_dict_lift[k]))
                        continue
                    for l in order_mean_dict_lift.keys():
                        if k == l:
                            st.text(str(k) + "=> " + "Conf: "+ str(order_mean_dict[k]) + " Lift: " + str(order_mean_dict_lift[k]))
                            break
                max_order = max(order_mean_dict, key=order_mean_dict.get)
                st.text("Order with highest mean is: " + "\n \t" + max_order)
                dict_rules = dict(sorted(dict_rules.items(), key=lambda item: item[1])) 
                st.text(dict_rules)
                print(dict_rules)
                
            else:
                for k in order_mean_dict.keys():
                    for l in order_mean_dict_lift.keys():
                        if len(k.split("->"))==num_attr and k ==l:
                            mean_dict_num_attr[k] = order_mean_dict[k]
                            st.text(str(k) + "=> " + "Conf: "+ str(order_mean_dict[k]) + " Lift: " + str(order_mean_dict_lift[k]))
                            break
            if len(mean_dict_num_attr) == 0:
                st.text("No rules of length " + str(num_attr) + " was found.")
            else:
                max_order = max(mean_dict_num_attr, key=mean_dict_num_attr.get)
                st.text("Order with highest mean is: " + "\n \t" + max_order)
    else:
        print("No rule block with 3 attributes are present")
        st.text("No rule block with 3 attributes are present with given support and confidence threshold")

def run_Apriori(num_attr, data_file, sup, conf, r_param):

    df = pd.read_csv(data_file, dtype=str)
    # df = df.sample(frac = 0.1) # take random 10% of dataframe
    

    transactions = []
    for sublist in df.values.tolist():
        clean_sublist = [item for item in sublist if item is not np.nan]
        transactions.append(clean_sublist)

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    
    if sup == 0:
        sup = 0.001
    frequent_itemsets_fp=fpgrowth(df, min_support=sup, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=conf)
    print(rules_fp.head())
    rules_fp_3cols = rules_fp[["antecedents","consequents","confidence","lift","consequent support"]]
    
    # #Converting data type of all rows into string
    rules_fp_3cols["antecedents"]=rules_fp_3cols["antecedents"].apply(str)
    rules_fp_3cols["consequents"]=rules_fp_3cols["consequents"].apply(str)
    rules_fp_3cols["confidence"]=rules_fp_3cols["confidence"].apply(str)
    rules_fp_3cols["lift"]=rules_fp_3cols["lift"].apply(str)
    rules_fp_3cols["consequent support"]=rules_fp_3cols["consequent support"].apply(str)
    
    
    # deleting rows where there are more than 1 element in RHS:
    df_RHS_1item = rules_fp_3cols[~rules_fp_3cols['consequents'].str.contains(',')]
    df_rules = df_RHS_1item.rename({'antecedents': 'LHS', 'consequents': 'RHS', 
    'confidence':'Conf', 'lift':'Lift',  'consequent support':'rhs_sup'}, axis=1)  # renaming columns
    print(df_rules.head())  
    
    df_rules["LHS"] = df_rules["LHS"].str.replace("frozenset", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("frozenset", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\(\{", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\(\{", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\}\)", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\}\)", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("'", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("'", "").astype(str)
    
    # df_rules["RHS"].str.replace("frozenset", "").astype(str)
    print(df_rules.head())

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
    support = 0
    confidence = 0
    rule_parameter = "Mean Confidence"
    data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/CSV"
    list_of_files = [f for f in os.listdir(data_folder)]
    filename = st.sidebar.selectbox("Select source data", sorted(list_of_files))
    data_file = data_folder + "/" + filename
    num_of_attr = [i for i in range(11)]
    attr_num = st.sidebar.selectbox("Number of Attributes", num_of_attr)
    attr_num = 2
    # data_file = data_folder + "/" + "Grouped.csv"
    # data_file = data_folder + "/" +  "test_FPGrowth.csv"
    # data_file = data_folder + "/" +  "Inpatient_Claims_PUF.csv"
    # data_file = data_folder + "/" +  "LDS_Sym_Proc_Diag.csv"
    st.text("Data File: " + data_file.split("/")[-1])
    print("Datafile: ", data_file)
    
    run_Apriori(attr_num, data_file, support, confidence, rule_parameter)

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