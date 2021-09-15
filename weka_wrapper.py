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
    for index, row in df.iterrows():
        lhs = row['LHS']
        rhs = row['RHS']
        if ',' not in lhs:  # keeping only those rules where # of elem in lhs and rhs is 1.
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
            if tree.depth() > 0:
                print("==================================BLOCK==================================")
                tree.show(line_type="ascii-em")

                for path in tree.paths_to_leaves():
                    print("-----------------Confidence Progression-----------------")
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
                    print("Rule Sequence: " + rule_sequence)
                    print("Confidence Sequence: " + conf_sequence)
                    conf_seq_list = [float(i) for i in conf_sequence.split("->")]
                    print(is_order(conf_seq_list))
                    print("\n")


def main(args):
    """
    Trains Apriori on the specified dataset
    """
    # load a dataset
    if len(args) <= 1:
        data_file = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/processed_claims_3_cond_included_no_treatment_non-resp.arff"
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)

    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori",
                         options=["-M", "0.1", "-c", "-1", "-C", "0.1", "-N", "1000"])
    apriori.build_associations(data)

    # iterate association rules (low-level)
    helper.print_info("****** Rules List ******")
    # make the underlying rules list object iterable in Python
    rules = javabridge.iterate_collection(apriori.jwrapper.getAssociationRules().getRules().o)

    cols = ['LHS', 'LHS_count', 'RHS', 'RHS_count', 'Conf']
    df_rules = pd.DataFrame(columns=cols)
    p_conf = re.compile('<conf:(.*)>')
    for i, rule in enumerate(rules):
        # wrap the Java object to make its methods accessible
        rule = JWrapper(rule)
        print(str(i + 1) + ". " + str(rule))
        rule = str(rule)
        lhs = rule.split("==>")[0]
        lhs_count = lhs.split(": ")[1]
        rhs = rule.split("==> ")[1]
        rhs_count = 0  # LOGIC: Change this and apply logic
        conf = p_conf.findall(rule)[0]
        lhs = rule.split(":")[0].replace("[", "").replace("]", "")
        rhs = rhs.split("<conf")[0].split(":")[0].replace("[", "").replace("]", "")
        if "," not in rhs:  # keeping only those rules with one element in right
            df_rules.loc[df_rules.shape[0]] = [lhs, lhs_count, rhs, rhs_count, conf]
    createRulesTree(df_rules)


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()