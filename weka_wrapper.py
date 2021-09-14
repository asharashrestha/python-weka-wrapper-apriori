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
import itertools
from treelib import Node, Tree


# create Tree structure for blocks of rules to show rule progression.
def createRulesTree(df):
    for index, row in df.iterrows():
        lhs = row['LHS']
        rhs = row['RHS']
        if ',' not in lhs:  # keeping only those rules where # of elem in lhs and rhs is 1.
            tree = Tree()
            rules_queue = []
            conf_queue = []
            tree.create_node(lhs + "->" + rhs + "[Conf: " + row["Conf"] + "]", "root")  # root node
            conf_queue.append((lhs + "->" + rhs, row["Conf"]))
            for index, row in df.iterrows():
                left_rule = lhs + ', ' + rhs
                if (row['LHS'] == left_rule):
                    left = row['LHS']
                    right = row['RHS']
                    a = left + ", " + right
                    # print("Id is: ", a)
                    tree.create_node(left + "->" + right + "[Conf:" + row["Conf"] + "]", a, parent="root")
                    rules_queue.append(left + ", " + right)
                    conf_queue.append((a, row["Conf"]))

            while len(rules_queue) != 0:
                elem = rules_queue.pop(0)
                for index, row in df.iterrows():
                    if (row['LHS'] == elem):
                        left = row['LHS']
                        right = row['RHS']
                        a = left + ", " + right  # concatenating lhs and rhs to find the result in lhs of dataframe
                        tree.create_node(left + "->" + right + "[Conf:" + row["Conf"] + "]", a, parent=left)
                        rules_queue.append(left + ", " + right)
                        conf_queue.append((a, row["Conf"]))
            print("----------------TREE------------------------")
            tree.show(line_type="ascii-em")
            print("----------------TREE------------------------")
            a = conf_queue[-1]
            for node in conf_queue:
                node = node[0]
                if node.count(",") == 0:
                    print(','.join([tree[node].tag for node in tree.rsearch("root")]))
                    print("\n")
                else:
                    print(','.join([tree[node].tag for node in tree.rsearch(node)]))
                    print("\n")
            print("=======================================")
            print("\n")


def main(args):
    """
    Trains Apriori on the specified dataset
    """
    # load a dataset
    if len(args) <= 1:
        data_file = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/preprocessed_4_attr.arff"
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori",
                         options=["-M", "0.1", "-c", "-1", "-C", "0.1", "-N", "1000"])
    apriori.build_associations(data)

    # iterate association rules (low-level)
    helper.print_info("Rules list")
    # make the underlying rules list object iterable in Python
    rules = javabridge.iterate_collection(apriori.jwrapper.getAssociationRules().getRules().o)

    cols = ['LHS', 'LHS_count', 'RHS', 'RHS_count', 'Conf']
    df_rules = pd.DataFrame(columns=cols)
    p_conf = re.compile('<conf:(.*)>')
    for i, rule in enumerate(rules):
        # wrap the Java object to make its methods accessible
        rule = str(JWrapper(rule))
        lhs = rule.split("==>")[0]
        lhs_count = lhs.split(": ")[1]
        rhs = rule.split("==> ")[1]
        rhs_count = 0
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