import pandas as pd
import streamlit as st
from treelib import Node, Tree
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import numpy as np

def createRulesTree(df_rules):
        print(df_rules.head())

        # filter df for LHS with the argument given by user.
        # if self.filter != None:
        #     df_rules = df_rules[df_rules['LHS'].str.contains(self.filter)]
        print("Reached Here")
        for index, row in df_rules.iterrows():
            lhs = row['LHS']
            rhs = row['RHS']
            if ',' not in lhs:  # keeping such rules as rules where # of elem in lhs and rhs is 1.
                tree = Tree()
                tree.create_node(lhs + "->" + rhs + "[Conf: " + row["Conf"] + "]", "root")  # root node

                my_queue = []

                for index, row in df_rules.iterrows():
                    left_rule = lhs + ', ' + rhs
                    if (row['LHS'] == left_rule):
                        left = row['LHS']
                        right = row['RHS']
                        a = left + ", " + right
                        tree.create_node(left + "->" + right + "[Conf:" + row["Conf"] + "]", a, parent="root")
                        my_queue.append(left + ", " + right)

                while len(my_queue) != 0:
                    elem = my_queue.pop(0)
                    for index, row in df_rules.iterrows():
                        if (row['LHS'] == elem):
                            left = row['LHS']
                            right = row['RHS']
                            a = left + ", " + right  # concatenating lhs and rhs to find the result in lhs of dataframe
                            tree.create_node(left + "->" + right + "[Conf:" + row["Conf"] + "]", a, parent=left)
                            my_queue.append(left + ", " + right)

                tree.show(line_type="ascii-em")
                if os.path.exists("tree.txt"):
                    os.remove("tree.txt")

                tree.save2file('tree.txt', line_type='ascii-em')
                self.showTree()
@st.cache
def run_FP_Growth(data_file):
    df = pd.read_csv(data_file, dtype=str)   
    transactions = []
    for sublist in df.values.tolist():
        clean_sublist = [item for item in sublist if item is not np.nan]
        transactions.append(clean_sublist)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_fp = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets_fp=fpgrowth(df_fp, min_support=0.0001, use_colnames=True)
    # global rules_fp
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0)
    rules_fp_3cols = rules_fp[["antecedents","consequents","confidence","lift"]]
    rules_fp_3cols["antecedents"]=rules_fp_3cols["antecedents"].apply(str)
    rules_fp_3cols["consequents"]=rules_fp_3cols["consequents"].apply(str)
    rules_fp_3cols["confidence"]=rules_fp_3cols["confidence"].apply(str)
    rules_fp_3cols["lift"]=rules_fp_3cols["lift"].apply(str)

    # deleting rows where there are more than 1 element in RHS:
    df_RHS_1item = rules_fp_3cols[~rules_fp_3cols['consequents'].str.contains(',')]
    df_rules = df_RHS_1item.rename({'antecedents': 'LHS', 'consequents': 'RHS', 'confidence':'Conf', 'lift':'Lift'}, axis=1) 
    df_rules["LHS"] = df_rules["LHS"].str.replace("frozenset", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("frozenset", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\(\{", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\(\{", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("\}\)", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("\}\)", "").astype(str)
    df_rules["LHS"] = df_rules["LHS"].str.replace("'", "").astype(str)
    df_rules["RHS"] = df_rules["RHS"].str.replace("'", "").astype(str)
    return df_rules

def sort_attributes(df_rules):  
    df = df_rules[~df_rules['RHS'].str.contains(',')] 
    for index, row in df.iterrows():
        LHS_list = row['LHS'].split(",")
        LHS_list = [x.strip(' ') for x in LHS_list]
        LHS_list = sorted(LHS_list)
        new_LHS = ', '.join(i for i in LHS_list)
        df.at[index,'LHS']= new_LHS
    return df



data_folder = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/GUI"
data_file = data_folder + "/GUI_Claims_LDS_selected.csv"
df = pd.read_csv(data_file, dtype=str)   
print(list(df))
for attr in list(df):
    df[attr].fillna('UNK', inplace=True)
# df_rules = pd.read_csv(data_folder + "/GUI_Claims_LDS_selected.csv", dtype=str)
df_rules = run_FP_Growth(data_file)

df_rules = sort_attributes(df_rules)
print(df_rules.to_csv("FP_Growth_rules.csv"))

symptoms=list(np.unique(df[['Symptom']])) 
diagnosis=list(np.unique(df[['diagnosis']])) 
procedure=list(np.unique(df[['procedure']]))
TOT_CHRG=list(np.unique(df[['TOT_CHRG']])) 
LOS=list(np.unique(df[['LOS']])) 
STUS_CD=list(np.unique(df[['STUS_CD']])) 

if(len(df_rules)!=0):
    selected_symptom=""
    selected_diagnosis=""
    selected_procedure=""
    clinical_outcomes =set()

    selected_seq = st.sidebar.selectbox("Select sequence: ", ["Symptom->Diagnosis->Procedure", "Symptom->Procedure->Diagnosis"])
    
    selected_symptom = st.sidebar.selectbox("Select Symptom: ", symptoms)
    
    if selected_symptom is not None:
        df_2 = df_rules[(df_rules['LHS'] == selected_symptom)]
        df_2 = df_2[df_2.apply(lambda r: r.str.contains('Diagnosis', case=False).any(), axis=1)]       
        diagnosis_list = list(np.unique(df_2[['RHS']])) 
        selected_diagnosis = st.sidebar.selectbox("Select Diagnosis: ", diagnosis_list)

    if selected_diagnosis is not None:
        df_3 = df_rules[(df_rules['LHS'] == selected_diagnosis + ", " + selected_symptom)]
        df_3 = df_3[df_3.apply(lambda r: r.str.contains('Procedure', case=False).any(), axis=1)]  
        procedure_list = list(np.unique(df_3[['RHS']])) 
        selected_procedure = st.sidebar.selectbox("Select Procedure: ", procedure_list)
   
    if selected_procedure is not None:
        df_4 = df_rules[(df_rules['LHS'] == selected_diagnosis + ", " + selected_symptom + ", " + selected_procedure)]
        for index, row in df_4.iterrows():
            clinical_outcomes.add(row['RHS'].split("=")[0])
        outcome = st.sidebar.selectbox("Select Outcome: ", clinical_outcomes)
        if outcome is not None:
            df_4 = df_4[df_4.apply(lambda r: r.str.contains(outcome, case=False).any(), axis=1)]  
            # charge_list = list(np.unique(df_4[['RHS']])) 
            st.write(df_4)


# createRulesTree(df_rules)