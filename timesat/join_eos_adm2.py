import arcpy
import os
import pandas as pd
import glob

#dbf to csv conversion
dbf_folder = "Z:\\Temp\\TIMESAT\\DBF\\adm2\\EOS\\2019_2020"
csv_folder = "Z:\\Temp\\TIMESAT\\DBF\\adm2\\EOS"

arcpy.env.workspace = dbf_folder


print("create folder to stored csv file...")
os.mkdir(os.path.join(csv_folder,'dbf_to_csv'))
print("Folder dbf_to_csv is created...")
print("start processing dbf to csv.....")

for i in os.listdir(dbf_folder):
    if i.endswith(".dbf"):
        print("processing " + i)
        new_name = i.split('.')[0]
        new_name_csv = '{0}.csv'.format(new_name)
        arcpy.TableToTable_conversion(in_rows=i,
                                      out_path=os.path.join(csv_folder,'dbf_to_csv'),
                                      out_name=new_name_csv)
        print("csv file "+i+" is created")

print("DBF to csv are completed...")
print("create folder to stored renaming colomn csv file...")
os.mkdir(os.path.join(csv_folder,'renamed_csv'))
print("Folder renamed_csv is created...")
print("start processing renaming colomn of csv files.....")

renamed_folder = os.path.join(csv_folder, 'renamed_csv')
csv_csv_folder = os.path.join(csv_folder,'dbf_to_csv')
for j in os.listdir(os.path.join(csv_folder,'dbf_to_csv')):
    if j.endswith(".csv"):
        name_split = j.split('_')
        date_name = name_split[1]
        season_name = name_split[6]
        #print(date_name, season_name)
        if season_name == 'season1.csv':
            s_name = 'S1'
        elif season_name == 'season2.csv':
            s_name = 'S2'
        new_colomn_name = 'C{0}{1}'.format(date_name, s_name)
        a = pd.read_csv(os.path.join(csv_csv_folder, j))
        #print(a)
        renamed_csv = a.rename(columns={'COUNT': new_colomn_name})
        #drop_other_colomn = a.drop(['OID', 'ZONE_CODE', 'AREA', 'MEAN'], axis=1)

        renamed_csv.to_csv(os.path.join(renamed_folder, j), index=False)
        print("renamed colomn csv file " + j + " is created")

print("Renaming coloumn are completed...")
print("create folder to stored cleaned csv file...")
os.mkdir(os.path.join(csv_folder,'cleaned_csv'))
print("Folder cleaned_csv is created...")
print("start processing cleaning csv files.....")

renamed_folder = os.path.join(csv_folder, 'renamed_csv')
cleaned_folder = os.path.join(csv_folder,'cleaned_csv')

for k in os.listdir(renamed_folder):
    if k.endswith(".csv"):
        b = pd.read_csv(os.path.join(renamed_folder, k))
        drop_other_colomn = b.drop(['OID', 'ZONE_CODE', 'AREA', 'MEAN'], axis=1)
        drop_other_colomn.to_csv(os.path.join(cleaned_folder, k), index=False)
        print("cleaned colomn csv file " + k + " is created")

print("Clenaing unused coloumn are completed...")
print("start Merging csv files.....")
path = cleaned_folder+"/*.csv"
file_concat = pd.concat([pd.read_csv(f).set_index('A2TEXT') for f in glob.glob(path)],axis=1).reset_index()
file_concat.to_csv(os.path.join(csv_folder, "temp_final_output.csv"), index=False)
csv_to_sort = pd.read_csv(os.path.join(csv_folder, "temp_final_output.csv"))
csv_sorted = csv_to_sort.reindex_axis(sorted(csv_to_sort.columns), axis=1)
csv_sorted.to_csv(os.path.join(csv_folder, "eos.csv"), index=False)
print("final result is created")