# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright (C) 2014-2018 Fracpete (pythonwekawrapper at gmail dot com)
from os import replace
import pandas as pd
import sys
import traceback
import weka.core.jvm as jvm
import helper as helper
from weka.core.converters import Loader
from weka.associations import Associator
import javabridge
from javabridge import JWrapper
import re
from itertools import permutations
from treelib import Node, Tree
from statistics import mean, median
import streamlit as st
import os

st.set_page_config(layout="wide")

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

# create Tree structure for blocks of rules to show rule progression.
def createRulesTree(df):
    var_seq_order = dict()
    for index, row in df.iterrows():
        lhs = row['LHS']
        rhs = row['RHS']
        if ',' not in lhs:  # keeping only those rules where # of elem in lhs and rhs is 1. To find the root node0
            tree = Tree()
            rules_queue = []
            tree.create_node(lhs + "->" + rhs + "[Conf: " + row["Conf"] + "]", "root")  # root node
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
            # print("==================================BLOCK==================================")
            
            tree.show(line_type="ascii-em")
            tree.save2file("Result_Tree.txt") 
            tree.to_graphviz("graph", shape="circle", graph='digraph')
            # tree.to_dict()
            tree.save2file("Ashara.txt")
            tree.to_json(with_data=True)
            tree.save2file("Result_Tree.txt") 
            st.text_area("RULE: ",tree)            
            
            
            # f = open("output.txt", "a")
            
            for path in tree.paths_to_leaves():
                # print("-----------------Confidence Progression-----------------")
                rule_sequence = ""
                conf_sequence = ""
                for i in path:
                    left_attr = ""
                    a = tree.get_node(i).tag
                    special_characters = '[[|]|(|)]'  # removing "[", "]" from confidence
                    conf = re.sub(special_characters, '', a.split("Conf:")[1])
                    conf = conf.strip().replace("(", "").replace(")", "")
                    right = a.split("Conf:")[0].split("->")[1].split("=")[0]
                    left = a.split("->")[0].split(",")
                    left_attr = ", ".join(l.split("=")[0] for l in left)
                    if rule_sequence == "":
                        rule_sequence = left_attr + " -> " + right
                        conf_sequence = conf
                    else:
                        rule_sequence += " -> " + right
                        conf_sequence += " -> " + conf
                # f.write("##################BLOCK##############################" + "\n")
                print("Rule Sequence: " + rule_sequence + "\n")
                print("Confidence Sequence: " + conf_sequence + "\n")
                st.text("Rule Sequence: " + rule_sequence + "\n")
                st.text("Confidence Sequence: " + conf_sequence + "\n")
                # f.write("Rule Sequence: " + rule_sequence + "\n")
                # f.write("Confidence Sequence: " + conf_sequence + "\n")
                conf_seq_list = [float(i) for i in conf_sequence.split("->")]
                if len(conf_seq_list) >1:
                    if rule_sequence in var_seq_order.keys():
                        var_seq_order[rule_sequence].append(
                            float(conf_sequence.split("->")[1]) - float(conf_sequence.split("->")[0]))
                    else:
                        var_seq_order[rule_sequence] = [float(conf_sequence.split("->")[1]) - float(conf_sequence.split("->")[0])]
                # f.write(is_order(conf_seq_list) + "\n")
                # f.write("####################################################" + "\n")
                    
    # f = open("output.txt")         
    # st.text_area("Rules",f.read(), height = 500)  
    # f.close()          
    order_mean_dict = dict()
    for k in var_seq_order.keys():
        print(k)
        print("\t Mean: "+  str(mean(var_seq_order[k])))
        # print("\t Median: "+ str(median(var_seq_order[k])))
        # st.text(k)
        order_mean_dict[k] = str(mean(var_seq_order[k]))
        # st.text(k + " =>" + " Mean: "+  str(mean(var_seq_order[k])))
        # st.text("\t Median: "+ str(median(var_seq_order[k])))
    print("\n")

    for k in order_mean_dict.keys():
        st.text(str(k) + "=> "+ str(order_mean_dict[k]))
    max_order = max(order_mean_dict, key=order_mean_dict.get) 
    st.text("Order with highest mean is: " + "\n \t" + max_order)
    num_of_attr =  len(df.columns)
    # st.sidebar.selectbox("Choose number of atributes", [int(i) for i in range(0,num_of_attr)])
    attr = st.sidebar.selectbox("Choose number of atributes", (1,2,3))
    # os.remove("output.txt")


def run_Apriori(data, sup, conf):
    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori",
                         options=["-M", str(sup), "-c", "-1", "-C", str(conf), "-N", "1000"])
    apriori.build_associations(data)
    # iterate association rules (low-level)
    helper.print_info("****** Rules List ******")
    # make the underlying rules list object iterable in Python
    rules = javabridge.iterate_collection(apriori.jwrapper.getAssociationRules().getRules().o)
    cols = ['LHS', 'RHS', 'Conf']
    df_rules = pd.DataFrame(columns=cols)
    p_conf = re.compile('<conf:(.*)>')
    for i, rule in enumerate(rules):
        # wrap the Java object to make its methods accessible
        rule = JWrapper(rule)
        # print(str(i + 1) + ". " + str(rule))
        rule = str(rule)
        lhs = rule.split("==>")[0]
        rhs = rule.split("==> ")[1]
        conf = p_conf.findall(rule)[0]
        lhs = rule.split(":")[0].replace("[", "").replace("]", "")
        rhs = rhs.split("<conf")[0].split(":")[0].replace("[", "").replace("]", "")
        if "," not in rhs:  # keeping only those rules with one element in right
            df_rules.loc[df_rules.shape[0]] = [lhs, rhs, conf]
    createRulesTree(df_rules)

def main(args):
    # Streamlit Sidebar and dashboard
    st.sidebar.write("Sidebar")
    support_threshold = [i for i in range(0,105,5)]
    support = st.sidebar.selectbox("Support Threshold",support_threshold)
    support = support/100

    conf_threshold = [i for i in range(0,105,5)]
    confidence = st.sidebar.selectbox("Confidence Threshold",conf_threshold)
    confidence = confidence/100
    # """
    # Trains Apriori on the specified dataset
    # """
    data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/Arff Dataset"
    list_of_files = [f for f in os.listdir(data_folder) ]
    filename = st.sidebar.selectbox("Select source data", list_of_files)
    data_file = data_folder + "/"+ filename
    # st.write("Datafile: ", data_file)
    print("Datafile: ", data_file)

    # helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    run_Apriori(data, support, confidence)

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
        
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()

