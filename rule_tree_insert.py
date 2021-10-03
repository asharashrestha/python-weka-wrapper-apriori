import sqlite3

db_file = "/Users/ashara/Documents/Study/Research/Dissertation/One Drive/OneDrive - University of Texas at Arlington/Dissertation/data_files/MIMIC/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4/mimic3.db"


def insert_tree_db(rule_tree, rule_seq, conf_seq):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # create table
    c.execute('''CREATE TABLE IF NOT EXISTS students
                (rollno real, name text, class real)''')
    c.execute(
        "INSERT INTO test VALUES('" + rule_tree + "', '" + rule_seq + "', '" + conf_seq + "', '" + conf_seq + "')")

    print(c.lastrowid)

    # commit the changes to db
    conn.commit()
    # close the connection
    conn.close()