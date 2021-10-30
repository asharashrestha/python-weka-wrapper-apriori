import csv
filepath ="/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/results/FP_Growth_test_Drug_1.csv"

# filepath ="/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/results/FP_Growth_LDS.csv"

conf_dict = dict()
count=0
with open(filepath) as csvfile:
    csv_reader = list(csv.reader(csvfile))
    for row_2_attr in csv_reader:
        l_2 = row_2_attr[0]
        r_2 = row_2_attr[1]
        conf_2 = float(row_2_attr[2])
        lift_2 = float(row_2_attr[3])
        c_l_2 = conf_2 * lift_2
        temp_list = []
        count+=1
        #Step 1
        if "Symptom=" in l_2 and "Diagnosis" not in l_2 and "Procedure" not in l_2 and "Drug" not in l_2\
                and "Diagnosis" in r_2 and "Symptom" not in r_2 and "Procedure" not in r_2 and "Drug" not in r_2:
            temp_list.append(l_2)
            temp_list.append(r_2)
            s = sorted(temp_list)
            left = s[0] + ", " + s[1]
            for row_3_attr in csv_reader:
                r_3 = row_3_attr[1]
                #Step 2
                if "Procedure" in r_3 and "Symptom" not in r_3 and \
                 "Diagnosis" not in r_3 and "Drug" not in r_3:
                    conf_3 = float(row_3_attr[2])
                    lift_3 = float(row_3_attr[3])
                    attr = row_3_attr[0].split(",")
                    attr = [i.strip() for i in attr]
                    if len(attr) == 2:
                        s = sorted(attr)
                        l_3 = s[0] + ", " + s[1]
                        if left == l_3:
                            #Step 3:
                            if "Procedure" in r_3 and "Symptom" not in r_3 and \
                                "Drug" not in r_3 and "Diagnosis" not in r_3:
                                left = l_3 + ', ' + r_3
                                attr = left.split(",")
                                attr = [i.strip() for i in attr]
                                s = sorted(attr)
                                left = s[0] + ", " + s[1] + ", " + s[2]
                                for row_4_attr in csv_reader:
                                    r_4 = row_4_attr[1]
                                    if "Symptom" not in r_4 and "Diagnosis" not in r_4 and "Procedure" not in r_4: 
                                        attr = row_4_attr[0].split(",")
                                        attr = [i.strip() for i in attr]
                                        # print(attr)
                                        if len(attr) == 3:
                                            s = sorted(attr)
                                            l_4 = s[0] + ", " + s[1] + ", " + s[2]
                                            if left == l_4:
                                                conf_4 = float(row_4_attr[2])
                                                lift_4 = float(row_4_attr[3])
                                                if (lift_4 * conf_4)>=(lift_3*conf_3)>=(lift_2*conf_2):
                                                    print("Left: ", l_4)
                                                    print("Right: ", r_4)
                                                    print("Conf 1: ", conf_2)
                                                    print("Conf 2: ", conf_3)
                                                    print("Conf 3: ", conf_4)
                                                    print("Success")
                                                    print("\n")