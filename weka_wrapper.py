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

import os
import sys
import traceback
import weka.core.jvm as jvm
import helper as helper
from weka.core.converters import Loader
from weka.associations import Associator
import javabridge
from javabridge import JWrapper
import re


def main(args):

    """
    Trains Apriori on the specified dataset (uses vote UCI dataset if no dataset specified).
    :param args: the commandline arguments
    :type args: list
    """
    #maintain a dictionary to keep track of confidences:
    conf_dict = dict()
    conf_dict["Symptoms->Diagnosis"] = [0]
    conf_dict["Diagnosis->Symptoms"] = [0]

    # load a dataset
    if len(args) <= 1:
        data_file = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/claims_sym_diag_preprocessed.arff"
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori", options=["-M", "0.01", "-c", "-1", "-C", "0.01", "-N", "1000"])
    apriori.build_associations(data)
    apriori.build_associations(data)
    # print(str(apriori))
    # print(type(apriori))

    # iterate association rules (low-level)
    helper.print_info("Rules (low-level)")
    # make the underlying rules list object iterable in Python
    rules = javabridge.iterate_collection(apriori.jwrapper.getAssociationRules().getRules().o)
    for i, r in enumerate(rules):
        # wrap the Java object to make its methods accessible
        rule = JWrapper(r)
        print(str(i+1) + ". " + str(rule))
        if ("Symptoms" in str(rule).split("==>")[0]):

            p = re.compile('<conf:(.*)>')
            conf_dict["Symptoms->Diagnosis"].append(p.findall(str(rule)))
        elif("Diagnosis" in str(rule).split("==>")[0]):
            conf_dict["Diagnosis->Symptoms"].append(p.findall(str(rule)))
    print(conf_dict)

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()