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

st.set_page_config(layout="wide")

sup = 0.0001 
conf = 0
data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/CSV"
data_file = data_folder + "/" +  "Synpuf_3attr.csv"
st.write("DataFile: ", data_file)
var_seq_order = dict()
var_seq_order_lift = dict()
lift_mult_conf_dict = dict()

# create Tree structure for blocks of rules to show rule progression.
def createRulesTree(df):
    for index, row in df.iterrows():
        
        l = row["LHS"]
        r = row["RHS"]
        lhs_attr = l.split("=")[0] 
        # lhs = l.split("=")[1]
        rhs_attr = r.split("=")[0]
        # rhs = r.split("=")[1]
        conf = str(round(float(row["Conf"]),2))
        lift = str(round(float(row["Lift"]),2))
        
        rule = l + "->" + r + "[Conf: " + conf + "] [Lift: " + lift + "]"
        st.text_area("RULE: ", rule)
        rule_sequence = lhs_attr + "->" + rhs_attr
        if rule_sequence in var_seq_order.keys():
            var_seq_order[rule_sequence].append(float(conf))
        else:
            var_seq_order[rule_sequence] = [float(conf)]

        if rule_sequence in var_seq_order_lift.keys():
            var_seq_order_lift[rule_sequence].append(float(lift))
        else:
            var_seq_order_lift[rule_sequence] = [float(lift)]

    order_mean_dict = dict()
    order_mean_dict_lift = dict()
    if (len(var_seq_order) > 0):
        for k in var_seq_order.keys():
            order_mean_dict[k] = str(mean(var_seq_order[k]))

        for k in var_seq_order_lift.keys():
            order_mean_dict_lift[k] = str(mean(var_seq_order_lift[k]))

        mean_dict_num_attr = dict()
        st.header("Mean for different sequences: ")
        for k in order_mean_dict.keys():
            for l in order_mean_dict_lift.keys():
                if k == l:
                    a = float(order_mean_dict[k]) * float(order_mean_dict_lift[k])
                    st.text(str(k) + "=> " + "; Conf: "+ str(order_mean_dict[k]) + " ; Lift: " + str(order_mean_dict_lift[k]) + "; Multiply: " + str(round(a,2)))
                    print(str(k) + "=> " + "; Conf: "+ str(order_mean_dict[k]) + " ; Lift: " + str(order_mean_dict_lift[k]) + "; Multiply: " + str(round(a,2)))
                    lift_mult_conf_dict[k] = a
                    break
        max_order = max(order_mean_dict, key=order_mean_dict.get)
        st.text("Order with highest mean is: " + "\n \t" + max_order)
 
        max_order = max(lift_mult_conf_dict, key=lift_mult_conf_dict.get)
        st.text("Order with highest score according to lift * conf is: " + "\n \t" + max_order)

def run_Apriori(data_file):
    df = pd.read_csv(data_file, dtype=str)
    transactions = []
    for sublist in df.values.tolist():
        clean_sublist = [item for item in sublist if item is not np.nan]
        transactions.append(clean_sublist)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    
    frequent_itemsets_fp=fpgrowth(df, min_support=sup, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=conf)
    rules_fp_3cols = rules_fp[["antecedents","consequents","confidence","lift"]]
    
    # #Converting data type of all rows into string
    rules_fp_3cols["antecedents"]=rules_fp_3cols["antecedents"].apply(str)
    rules_fp_3cols["consequents"]=rules_fp_3cols["consequents"].apply(str)
    rules_fp_3cols["confidence"]=rules_fp_3cols["confidence"].apply(str)
    rules_fp_3cols["lift"]=rules_fp_3cols["lift"].apply(str)
    
    for index, row in rules_fp_3cols.iterrows():
        db.insert_FpGrowth_rules(row["antecedents"],row["consequents"],row["confidence"],row["lift"], data_file.split("/")[-1])

    # deleting rows where there are more than 1 element in RHS:
    df_RHS_1item = rules_fp_3cols[~rules_fp_3cols['consequents'].str.contains(',')]
   # deleting rows where there are more than 1 element in LHS:
    df = df_RHS_1item[~rules_fp_3cols['antecedents'].str.contains(',')]

    df_rules = df.rename({'antecedents': 'LHS', 'consequents': 'RHS', 'confidence':'Conf', 'lift':'Lift'}, axis=1)  # renaming columns

    df_rules["LHS"] = df_rules["LHS"].str.replace("frozenset", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("frozenset", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\(\{", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\(\{", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\}\)", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\}\)", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("'", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("'", "").astype(str)
    
    print(df_rules.head())  

    createRulesTree(df_rules)
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