import pandas as pd
import random
 
eval= True

file_name='/home/kushal/data/tcgamel_clam/segmentation/process_list_autogen.csv'

df = pd.read_csv(file_name)

print(df.head())
df =df.iloc[:-1]



df['slide_id']=df['slide_id'].str.replace('.svs','',case=False)

if eval== False:
    print(df.head())
    df.to_csv('process_list_autogen.csv',index=False)

if eval == True:
    dest_file ="dataset_csv/tumor_vs_normal_dummy_clean.csv"
    fake_label_df=df.filter(['slide_id'],axis=1)
    print(fake_label_df.head())
    num_array=[random.randint(1,2) for i in range(len(df))]
    print(num_array)
    print(len(num_array))
    label_dict = {1:"tumor_tissue",2:"normal_tissue"}
    labels = [label_dict[i] for i in num_array]
    fake_label_df =fake_label_df.assign(label=labels)
    case_id =[i for i in range(len(df))]
    fake_label_df =fake_label_df.assign(case_id=case_id)
    print(fake_label_df.head())
    fake_label_df.to_csv(dest_file,index=False)