import pandas as pd



file_name='/home/kushal/data/gem_blgm_results_uni_conch_224_path/process_list_autogen.csv'

df = pd.read_csv(file_name)

print(df.head())
df =df.iloc[:-1]



df['slide_id']=df['slide_id'].str.replace('.svs','',case=False)


print(df.head())
df.to_csv('process_list_autogen_refined_uni_224.csv',index=False)
