#%%
import pandas as pd

filepath = './ECG wave data/2017_4.xlsx'

df = pd.read_excel(filepath)

df.columns

#%%
len(df.columns)
# %%
df.columns = ['PTNO','시행일시','진료과코드','진료과명','검사코드(xml)','검사코드(폴더)'
,'WAVE I','WAVE II','WAVE III','WAVE aVR','WAVE aVL','WAVE aVF','WAVE V1','WAVE V2','WAVE V3'
,'WAVE V4','WAVE V5','WAVE V6','장비명','HeartRate','PRInterval','QRSDuration','QTInterval'
,'QTCorrected','PAxis','RAxis','TAxis','Severity','PaceMaker','RRInterval','QTCB','QTCF'
,'RV5','SV1','RS','Dx Statement 1','Dx Comment 1','Dx Statement 2','Dx Comment 2','Dx Statement 3'
,'Dx Comment 3','Dx Statement 4','Dx Comment 4','Dx Statement 5','Dx Comment 5','LeadCount'
,'SamplingRate','Amplitude','Filter HightPass','Filter LowPass','Filter AC','Filter Muscle','Filter Baseline'
,'UserGain limb','UserGain chest']

df.columns

# %%
df['MATCH_KEY'] = df['PTNO'] + df['시행일시']
# %%
df

#%%
filepath[-11:-5]
# %%
# df.to_excel('./WAVE_FIN/2014_1.xlsx'.format(filepath[-11:-5]))
with pd.ExcelWriter('./ECG wave data/WAVE_FIN/{}.xlsx'.format(filepath[-11:-5]),mode='w',engine='openpyxl') as writer:
    df.to_excel(writer,index=False)

#%%
import pandas as pd
import os

dir = './ECG wave data'
files = os.listdir(dir)

filelist = []
for file in files:
    fileinfo = dir + '/'+ file
    filelist.append(fileinfo)

filelist[20][len(dir)+1:-5]
# filelist[21:-3]
# %%
import pandas as pd
import os

dir = './ECG wave data'
files = os.listdir(dir)

filelist = []
for file in files:
    fileinfo = dir + '/'+ file
    filelist.append(fileinfo)
#%%
filelist[49:51]

# %%
# for filepath in filelist[0:49]:
# for filepath in filelist[21:49]:
for filepath in filelist[49:51]:
    df = pd.read_excel(filepath)
    if len(df.columns) == 55:
        df.columns = ['PTNO','시행일시','진료과코드','진료과명','검사코드(xml)','검사코드(폴더)'
                    ,'WAVE I','WAVE II','WAVE III','WAVE aVR','WAVE aVL','WAVE aVF','WAVE V1','WAVE V2','WAVE V3'
                    ,'WAVE V4','WAVE V5','WAVE V6','장비명','HeartRate','PRInterval','QRSDuration','QTInterval'
                    ,'QTCorrected','PAxis','RAxis','TAxis','Severity','PaceMaker','RRInterval','QTCB','QTCF'
                    ,'RV5','SV1','RS','Dx Statement 1','Dx Comment 1','Dx Statement 2','Dx Comment 2','Dx Statement 3'
                    ,'Dx Comment 3','Dx Statement 4','Dx Comment 4','Dx Statement 5','Dx Comment 5','LeadCount'
                    ,'SamplingRate','Amplitude','Filter HightPass','Filter LowPass','Filter AC','Filter Muscle','Filter Baseline'
                    ,'UserGain limb','UserGain chest']
        df['MATCH_KEY'] = df['PTNO'] + df['시행일시']
    elif len(df.columns) == 57:
        df.columns = ['PTNO','SM_DATE','시행일시1','시행일시2','진료과코드','진료과명','검사코드(xml)','검사코드(폴더)'
                     ,'WAVE I','WAVE II','WAVE III','WAVE aVR','WAVE aVL','WAVE aVF','WAVE V1','WAVE V2','WAVE V3'
                     ,'WAVE V4','WAVE V5','WAVE V6','장비명','HeartRate','PRInterval','QRSDuration','QTInterval'
                     ,'QTCorrected','PAxis','RAxis','TAxis','Severity','PaceMaker','RRInterval','QTCB','QTCF'
                     ,'RV5','SV1','RS','Dx Statement 1','Dx Comment 1','Dx Statement 2','Dx Comment 2','Dx Statement 3'
                     ,'Dx Comment 3','Dx Statement 4','Dx Comment 4','Dx Statement 5','Dx Comment 5','LeadCount'
                     ,'SamplingRate','Amplitude','Filter HightPass','Filter LowPass','Filter AC','Filter Muscle','Filter Baseline'
                     ,'UserGain limb','UserGain chest']
        df['MATCH_KEY'] = df['PTNO'] + df['SM_DATE']
    else:
        pass

    with pd.ExcelWriter('./ECG wave data/WAVE_FIN/{}.xlsx'.format(filepath[len(dir)+1:-5]),mode='w',engine='openpyxl') as writer:
        df.to_excel(writer,index=False)
# %%
import pandas as pd
import os

dir = './ECG wave data'
files = os.listdir(dir)

filelist = []
for file in files:
    fileinfo = dir + '/'+ file
    filelist.append(fileinfo)

filelist[:49]

#%%
filepath = filelist[0]

df = pd.read_excel(filepath)

df
#%%
df.columns

#%%
filevarlist = []
filevarlist.append([filepath[len(dir)+1:-5], df.columns, len(df.columns)])

filevarlist

filevarlist_df = pd.DataFrame(filevarlist,columns = ['file_nm','varlist','var_len'])

filevarlist_df
# %%
filevarlist = []

for filepath in filelist[:49]:
    df = pd.read_excel(filepath)
    filevarlist.append([filepath[len(dir)+1:-5], df.columns, len(df.columns)])
    
filevarlist_df = pd.DataFrame(filevarlist,columns = ['file_nm','varlist','var_len'])

with pd.ExcelWriter('./ECG wave data/org_data_var_info.xlsx',mode='w',engine='openpyxl') as writer:
    filevarlist_df.to_excel(writer,index=False)

#%%
import pandas as pd
import os

dir = './ECG wave data/WAVE_FIN'
files = os.listdir(dir)

filelist = []
for file in files:
    fileinfo = dir + '/'+ file
    filelist.append(fileinfo)

filelist

#%%    
filevarlist = []

for filepath in filelist[:49]:
    df = pd.read_excel(filepath)
    filevarlist.append([filepath[len(dir)+1:-5], df.columns, len(df.columns)])
    
filevarlist_df = pd.DataFrame(filevarlist,columns = ['file_nm','varlist','var_len'])

with pd.ExcelWriter('./ECG wave data/WAVE_FIN/trns_data_var_info.xlsx',mode='w',engine='openpyxl') as writer:
    filevarlist_df.to_excel(writer,index=False)

#%%
import pandas as pd
import os

dir = './ECG wave data/WAVE_FIN'
files = os.listdir(dir)

filelist = []
for file in files:
    fileinfo = dir + '/'+ file
    filelist.append(fileinfo)
    
for filepath in filelist:
    df = pd.read_excel(filepath)
    print(filepath, df)
# %%
print(df)