from os import replace
from google.protobuf.symbol_database import Default
import pandas as pd
import sys
import traceback
import numpy as np
from statistics import  mean
import streamlit as st
import os
import execute_sql as db
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from datetime import datetime

start_time = datetime.now()
print("Start Time: ", start_time)

# st.set_page_config(layout="wide")

# sup = 0.01 
# conf = 0.1
data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/CSV"
# data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/GUI"
# data_file = data_folder + "/" +  "temp_with_drug_2.csv" # Highest rank order is Symtpom => Procedure => Diagnosis => Drug with Support = 10% and Conf = 0
# # Symptom -> diag -> Drug -> proc Sup = 0.01; Conf = 0.1
data_file = data_folder + "/" +  "temp_with_drug_1.csv" # Rank 2 is Symptom-> Diagnosis -> Procedure -> Drug Sup = 0.01; Conf = 0.1
# data_file = data_folder + "/" +  "Synpuf_3attr.csv" # rank 1 is Symptom => Diagnosis => Procedure: Sup = 0.0001; Conf = 0.1
# st.write("DataFile: ", data_file)
var_seq_order_conf = dict()
var_seq_order_lift = dict()
var_seq_order_cf = dict()
lift_mult_conf_dict = dict()
final_dict = dict()


# create Tree structure for blocks of rules to show rule progression.
def createRulesTree(df, metric):
    count_write=0
    for index, row in df.iterrows(): 
        l = row["LHS"]
        r = row["RHS"]
        lhs_attr = l.split("=")[0] 
        # lhs = l.split("=")[1]
        rhs_attr = r.split("=")[0]
        # rhs = r.split("=")[1]
        conf = row["Conf"]
        lift = row["Lift"]
        lhs_sup = row["lhs_sup"]
        rhs_sup =row["rhs_sup"]
        # conf = str(round(float(row["Conf"]),2))
        # lift = str(round(float(row["Lift"]),2))
        
        #Using Certainy Factor
        if conf > rhs_sup:
            cf = (conf -  rhs_sup)/(1-rhs_sup)
            cf = round(cf,4)
        elif conf < rhs_sup:
            cf = (conf - rhs_sup)/rhs_sup
            cf = round(cf,4)
        else:
            cf = 0
        # #Using Conviction
        # cf = (1 - rhs_sup)/(1-conf)
        # if conf > rhs_sup:
        #     cf = (conf -  rhs_sup)/(1-rhs_sup)
        #     cf = round(cf,4)
        # elif conf < rhs_sup:
        #     cf = (conf - rhs_sup)/rhs_sup
        #     cf = round(cf,4)
        # else:
        #     cf = 0

        rule = l + "\n└──" + r + "\n[Conf: " + str(conf) + "] [Lift: " + str(lift) + "] [CF: " + str(cf) +"]"
        st.text_area("RULE: ", rule)
        # st.text_area(rule, height=10)
        rule_sequence = lhs_attr + "->" + rhs_attr
        if rule_sequence in var_seq_order_conf.keys():
            if cf > 0:
                var_seq_order_conf[rule_sequence].append(float(conf*lift))
        else:
            if cf > 0: 
                var_seq_order_conf[rule_sequence] = [float(conf*lift)]

        if rule_sequence in var_seq_order_lift.keys():
            var_seq_order_lift[rule_sequence].append(float(lift))
        else:
            var_seq_order_lift[rule_sequence] = [float(lift)]

        if rule_sequence in var_seq_order_cf.keys():
            # if cf > 0:
            var_seq_order_cf[rule_sequence].append(float(cf))
        else:
            # if cf > 0:
            var_seq_order_cf[rule_sequence] = [float(cf)]

    order_mean_dict_conf = dict()
    order_mean_dict_lift = dict()
    order_mean_dict_cf = dict()
    asc_dict = dict()
    if (len(var_seq_order_conf) > 0):
        for k in var_seq_order_conf.keys():
            order_mean_dict_conf[k] = str(round(mean(var_seq_order_conf[k]),4))

        for k in var_seq_order_lift.keys():
            order_mean_dict_lift[k] = str(round(mean(var_seq_order_lift[k]),4))
            
        for k in var_seq_order_cf.keys():
            order_mean_dict_cf[k] = str(round(mean(var_seq_order_cf[k]),4))

        mean_dict_num_attr = dict()
        # st.header("Mean for different sequences: ")
        # for k in order_mean_dict_conf.keys():
        #     for l in order_mean_dict_lift.keys():
        #         if k == l:
        #             a = float(order_mean_dict_conf[k]) * float(order_mean_dict_lift[k])
        #             st.text(str(k) + "=> " + "; Conf: "+ str(round(float(order_mean_dict_conf[k]),2)) + " ; Lift: " + str(round(float(order_mean_dict_lift[k]),2)) + "; Lift * Confidence: " + str(round(a,2)))
        #             asc_dict[k] = round(a,2)
        #             # print(str(k) + "=> " + "; Conf: "+ str(order_mean_dict[k]) + " ; Lift: " + str(order_mean_dict_lift[k]) + "; Lift * Confidence: " + str(round(a,2)))
        #             lift_mult_conf_dict[k] = a
        #             break
        # max_order = max(order_mean_dict_conf, key=order_mean_dict_conf.get)
        st.subheader("Mean for different sequences: ")
        # for k in order_mean_dict_cf.keys():
        #     a = float(order_mean_dict_cf[k])
        #     st.text(str(k) + "=> " + "; CF: "+ str(round(float(order_mean_dict_cf[k]),2)))
        #     asc_dict[k] = round(a,2)
        #     # print(str(k) + "=> " + "; Conf: "+ str(order_mean_dict[k]) + " ; Lift: " + str(order_mean_dict_lift[k]) + "; Lift * Confidence: " + str(round(a,2)))
        #     final_dict[k] = a
        #     break
        # max_order = max(order_mean_dict_conf, key=order_mean_dict_conf.get)
        # st.text("Order with highest mean is: " + "\n \t" + max_order)
 
        # max_order = max(lift_mult_conf_dict, key=lift_mult_conf_dict.get)
        # st.text("Order with highest score according to lift * conf is: " + "\n \t" + max_order)
        if metric == "Certainty Factor":
            asc_dict =  dict(sorted(order_mean_dict_cf.items(), key=lambda item: item[1])) #sorting by value
            st.write("**Sequences in ascending order of mean(CF)**")
        elif metric == "Confidence":
            asc_dict =  dict(sorted(order_mean_dict_conf.items(), key=lambda item: item[1]))#sorting by value 
            st.write("**Sequences in ascending order of mean(Confidence)**")
        elif metric == "Lift":
            asc_dict =  dict(sorted(order_mean_dict_lift.items(), key=lambda item: item[1]))#sorting by value
        
        
        st.write(asc_dict)
        count = 1
        for k,v,in asc_dict.items():
            # st.write(str(count) + ". " + str(k) + ":" +str(v))
            count+=1
        num_attr = 3
        for key, value in asc_dict.items():
            elems = key.split("->")
            LHS = elems[0]
            RHS = elems[1]
            CL = value
            for key1, value1 in asc_dict.items():
                elems1 = key1.split("->")
                LHS1 = elems1[0]
                RHS1 = elems1[1]
                CL1 = value1
                
                
                if LHS1 == RHS and value1>=value and LHS!=RHS1:
                    # st.header("Order that satisfy Ascendingness: ")
                    if count_write == 0:
                        st.write("**Order that satisfy Ascendingness:** ")
                    st.write(LHS+"->"+RHS+"->"+RHS1)
                    count_write +=1

def run_Apriori(data_file):
    list_of_files = [f for f in os.listdir(data_folder)]
    data_file = st.sidebar.selectbox("Select source data", sorted(list_of_files), key="file_key")
    metric_list = ["Certainty Factor","Confidence", "Lift"]
    metric = st.sidebar.selectbox("Select a metric", metric_list, key="metric_key")
    support_threshold = [i for i in range(0, 101, 1)]
    sup = st.sidebar.select_slider("Min Support", options = support_threshold)
    sup = sup/100

    conf_threshold = [i for i in range(0, 101, 1)]
    conf = st.sidebar.select_slider("Min Confidence", options = conf_threshold)
    conf = conf/100

    data_file = data_folder + "/" + data_file
    df = pd.read_csv(data_file, dtype=str)
    print(data_file)
    # st.write("Data File being processed:", data_file)
    transactions = []
    for sublist in df.values.tolist():
        clean_sublist = [item for item in sublist if item is not np.nan]
        transactions.append(clean_sublist)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # st.write("Data File: ", data_file)
    frequent_itemsets_fp=fpgrowth(df, min_support=sup, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=conf)

    rules_fp.to_csv("raw_FP.csv")
    rules_fp_3cols = rules_fp[["antecedents","consequents","confidence","lift", "antecedent support", "consequent support"]]
    
    # #Converting data type of all rows into string
    rules_fp_3cols["antecedents"]=rules_fp_3cols["antecedents"].apply(str)
    rules_fp_3cols["consequents"]=rules_fp_3cols["consequents"].apply(str)
    rules_fp_3cols["confidence"]=rules_fp_3cols["confidence"].apply(str)
    rules_fp_3cols["lift"]=rules_fp_3cols["lift"].apply(str)
    rules_fp_3cols["antecedent support"]=rules_fp_3cols["antecedent support"].apply(str)
    rules_fp_3cols["consequent support"]=rules_fp_3cols["consequent support"].apply(str)

    rules_fp_3cols["confidence"] = np.round(rules_fp_3cols["confidence"].astype(float), decimals=2)
    rules_fp_3cols["lift"] = np.round(rules_fp_3cols["lift"].astype(float), decimals=2)
    rules_fp_3cols["antecedent support"] = np.round(rules_fp_3cols["antecedent support"].astype(float), decimals=2)
    rules_fp_3cols["consequent support"] = np.round(rules_fp_3cols["consequent support"].astype(float), decimals=2)
    
    # for index, row in rules_fp_3cols.iterrows():
    #     db.insert_FpGrowth_rules(row["antecedents"],row["consequents"],row["confidence"],row["lift"], data_file.split("/")[-1])

    # deleting rows where there are more than 1 element in RHS:
    df_RHS_1item = rules_fp_3cols[~rules_fp_3cols['consequents'].str.contains(',')]
   # deleting rows where there are more than 1 element in LHS:
    df = df_RHS_1item[~rules_fp_3cols['antecedents'].str.contains(',')]

    df_rules = df.rename({'antecedents': 'LHS', 'consequents': 'RHS', 
    'confidence':'Conf', 'lift':'Lift',  'antecedent support':'lhs_sup',  'consequent support':'rhs_sup'}, axis=1)  # renaming columns
    # print(df_rules.head())
    # print(df_rules.dtypes)
    df_rules["LHS"] = df_rules["LHS"].str.replace("frozenset", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("frozenset", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\(\{", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\(\{", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\}\)", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\}\)", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("'", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("'", "").astype(str)
    
    print(df_rules.head())  
    createRulesTree(df_rules, metric)
def main(args):
    print("Datafile: ", data_file)
    run_Apriori(data_file)

if __name__ == "__main__": 
    try:
        # jvm.start()
        main(sys.argv)
        

    except Exception as e:
        print(traceback.format_exc())
    finally:
        # jvm.stop()
        end_time = datetime.now()
        print(end_time)
        print('Duration: {}'.format(end_time - start_time))