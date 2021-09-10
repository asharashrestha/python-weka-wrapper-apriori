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

# apriori_output.py
# Copyright (C) 2014-2018 Fracpete (pythonwekawrapper at gmail dot com)
# test comment for github
import pandas as pd
import numpy as np
import sys
import traceback
import weka.core.jvm as jvm
import helper as helper
from weka.core.converters import Loader
from weka.associations import Associator
import javabridge
from javabridge import JWrapper
import re
import statistics
import itertools
import csv
import pprint


csv_file_path ="/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/processed_claims_3_cond_excluded_no-treatment_non-resp.csv"
df = pd.read_csv(csv_file_path,skiprows=0)
attribute_list = list(df.columns)
print(attribute_list)

def find_mean(conf_collection):
    confidences_list = []
    for i in conf_collection:
        confidences_list.append(float(str(i[0]).replace("(","").replace(")","")))
    return statistics.mean(confidences_list)


# takes list of uniques values of single variable and returns pair of values of given length from all pair of variables
def find_pairs_of_variables(comb_len, list_of_lists):
    permutations_of_list = []
    element_pairs = []

    for L in range(0, len(list_of_lists) + 1):
        for subset in itertools.permutations(list_of_lists, L):
            if len(subset)==comb_len:
                permutations_of_list.append(subset)
    # print(list_of_combinations)
    for i in permutations_of_list:
        for j in (list(itertools.product(i[0], i[1]))):
            element_pairs.append(j)
    print(element_pairs)


def read_file_each_chunk(stream, separator):
  buffer = ''
  while True:  # until EOF
    chunk = stream.read(4096)  # I propose 4096 or so
    if not chunk:  # EOF?
      yield buffer
      break
    buffer += chunk
    while True:  # until no separator is found
      try:
        part, buffer = buffer.split(separator, 1)
      except ValueError:
        break
      else:
        yield part



def main(args):

    """
    Trains Apriori on the specified dataset (uses vote UCI dataset if no dataset specified).
    :param args: the commandline arguments
    :type args: list
    """

    # load a dataset
    if len(args) <= 1:
        # data_file = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/processed_claims_3_cond.arff"
        data_file = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/processed_claims_3_cond_excluded_no-treatment_non-resp.arff"

        data_value = np.asarray(data_file[0])
        attributes = data_file[0]
        print(data_value)
        print(attributes)
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    print("==================================================")

    #creating dictionary number of lists equal to number of attributes
    attribute_dictionary = dict()
    for attr in attributes:
        attribute_dictionary[attr] = []

    for attr in attribute_list:
        if attr != "dtype":
            attribute_dictionary[attr] = [attr + "="+df[attr].unique()]

    p=pprint.PrettyPrinter(indent=4)
    p.pprint(attribute_dictionary)
    print(attribute_dictionary)
    print("==================================================")


    #finding combinatons of different attributes
    attributes_pairs = (list(itertools.combinations(attributes, 2)))


    #initialize a dictionary to keep track of confidences:
    conf_dict = {
        "two-only": dict(),
        "three-only": dict(),
        "two-and-three": dict()
    }
    for i in attributes_pairs:
        rule_forward = i[0] + " -> " + i[1]
        rule_backward = i[1] + " -> " + i[0]
        conf_dict["two-only"][rule_forward] = []
        conf_dict["two-only"][rule_backward] = []
        conf_dict["three-only"][rule_forward] = []
        conf_dict["three-only"][rule_backward] = []
        conf_dict["two-and-three"][rule_forward] = []
        conf_dict["two-and-three"][rule_backward] = []

    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori", options=["-M", "0.1", "-c", "-1", "-C", "0.1", "-N", "1000"])
    apriori.build_associations(data)
    apriori.build_associations(data)
    # print(str(apriori))

    # iterate association rules (low-level)
    helper.print_info("Rules (low-level)")
    # make the underlying rules list object iterable in Python
    rules = javabridge.iterate_collection(apriori.jwrapper.getAssociationRules().getRules().o)



    for i, r in enumerate(rules):
        # wrap the Java object to make its methods accessible
        rule = JWrapper(r)
        # print(str(i+1) + ". " + str(rule))

        for i in attributes_pairs:
            attr_1 = i[0]
            attr_2 = i[1]
            rule_forward = attr_1 + " -> " + attr_2
            rule_backward = attr_2 + " -> " + attr_1
            # print("=======================================")
            attr_3 = [x for x in attributes if x not in list(i)][0]


            #Considering all 3 attributes
            if(i[0] in str(rule).split("==>")[0]):
                p = re.compile('<conf:(.*)>') # regular expression to find confidence by finding string "<conf: ***>"
                conf_dict["two-and-three"][rule_forward].append(p.findall(str(rule)))
            elif (i[1] in str(rule).split("==>")[0]):
                p = re.compile('<conf:(.*)>')
                conf_dict["two-and-three"][rule_backward].append(p.findall(str(rule)))

            # Considering all only 2 attributes
            if attr_3 not in str(rule):
                if (i[0] in str(rule).split("==>")[0]):
                    p = re.compile(
                        '<conf:(.*)>')  # regular expression to find confidence by finding string "<conf: ***>"
                    conf_dict["two-only"][rule_forward].append(p.findall(str(rule)))
                elif (i[1] in str(rule).split("==>")[0]):
                    p = re.compile('<conf:(.*)>')
                    conf_dict["two-only"][rule_backward].append(p.findall(str(rule)))

            # Considering all only 3 attributes
            if attr_3 in str(rule):
                if (i[0] in str(rule).split("==>")[0]):
                    p = re.compile(
                        '<conf:(.*)>')  # regular expression to find confidence by finding string "<conf: ***>"
                    conf_dict["three-only"][rule_forward].append(p.findall(str(rule)))
                elif (i[1] in str(rule).split("==>")[0]):
                    p = re.compile('<conf:(.*)>')
                    conf_dict["three-only"][rule_backward].append(p.findall(str(rule)))
    for i in attributes_pairs:
        attr_1 = i[0]
        attr_2 = i[1]
        rule_forward = attr_1 + " -> " + attr_2
        rule_backward = attr_2 + " -> " + attr_1

        # print("+---------------------------------------------------------------------------  ")
        # print("| Relation between " + attr_1 + " and " + attr_2 + ":")
        # print("+---------------------------------------------------------------------------  ")
        #
        # print("| Mean of (three-only)" + rule_forward + "==>"+str(find_mean(conf_dict["three-only"][rule_forward])))
        # print("| Mean of (three-only)" + rule_backward +"==>"+str(find_mean(conf_dict["three-only"][rule_backward])))
        # print("| Mean of (two-only)" + rule_forward+"==>"+str(find_mean(conf_dict["two-only"][rule_forward])))
        # print("| Mean of (two-only)" + rule_backward + "==>" + str(find_mean(conf_dict["two-only"][rule_backward])))
        # print("| Mean of (two-and-three)" + rule_forward + "==>"+str(find_mean(conf_dict["two-and-three"][rule_forward])))
        # print("Mean of (two-and-three)" + rule_backward+"==>"+str(find_mean(conf_dict["two-and-three"][rule_backward])))
        print("+--------------------------------------------------------------------------- ")

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
