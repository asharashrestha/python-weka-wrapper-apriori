import json
import os
import csv
file_path = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/MIMIC/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4/Symptoms_3cond/notes_output"
symptom_tup = set()
for file in os.listdir(file_path):
    if '.out' in file:
        # Opening JSON file
        f = open(file_path + "/" + file)
        # returns JSON object as a dictionary
        data = json.load(f)
        for entity in data['Entities']:
            try:
                for trait in entity["Traits"]:
                    if trait["Name"] == 'SYMPTOM':
                        for icdconcept in entity["ICD10CMConcepts"]:
                            symptom = icdconcept["Description"]
                            symptom_code = icdconcept["Code"]
                            symptom_tup.add((file, symptom, symptom_code))
                            break
            except Exception as e:
                print(e)
        f.close()
with open('symptoms.csv','w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['Filename','Symptom','ICD-10 Code'])
    for row in symptom_tup:
        csv_out.writerow(row)
