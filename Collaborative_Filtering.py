
# coding: utf-8

# In[430]:

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import random

def read_Books(file_name):
    '''Read the Initial Book Data'''
    cwd=os.getcwd()
    path=cwd+"\\"+file_name
    data_frame=pd.read_csv(path,sep=';',encoding='utf-8',error_bad_lines=False)
    return data_frame

def read_ratings(file_name):
    '''Read the Initial Rating Table Given to use
    as Part of input to Recommender System'''
    cwd=os.getcwd()
    path=cwd+"\\"+file_name
    data_frame=pd.read_csv(path,sep=';',encoding='utf-8',header=0)
    return data_frame

def filter_items(df,min2):
    '''Filter the Data Based on the number of rated items
    to reduce complexity'''
    multi = df.set_index(['Book','User'])
    counts = df.groupby('Book').count()
    ind = counts[(counts.User>min2)].index
    l = []
    ll = []
    for i in ind:
        data = multi.loc[i]
        data = data.assign(Book = np.empty(data.shape[0]))
        data.Book = i
        l.append(data)
    DF = l[0]
    for j in range(1,len(l)):
        DF = DF.append(l[j])
    DF.reset_index(inplace=True)
    return DF

def filter_data(ratings,min1 = 200,min2 = 50):
    '''Filter the Data Based on the number of rated items
    to reduce complexity'''
    multi = ratings.set_index(['User','Book'])
    counts = ratings.groupby('User').count()
    ind = counts[(counts.Book>min1)].index
    l = []
    ll = []
    for i in ind:
        data = multi.loc[i]
        data = data.assign(User = np.empty(data.shape[0]))
        data.User = i
        l.append(data)
    DF = l[0]
    for j in range(1,len(l)):
        DF = DF.append(l[j])
    DF.reset_index(inplace=True)
    DF = filter_items(DF,min2)
    return DF

def find_samples(Data):
    '''Make Training Set & Test Set from Given Data'''
    Train_indices = []
    Test_indices = []
    counter = 0
    c = 0
    rows = (Data.notnull().sum(axis=1).sort_values(ascending = False)>1)[:Data.shape[0]//5].index
#     cols = (Data.notnull().sum().sort_values(ascending = False)>1).index[:20]
    while c != 5:
#         rows = np.random.choice(Data.shape[0],Data.shape[0]//5, replace=False)
        cols = np.random.choice(Data.shape[1],Data.shape[1]//5, replace=False)
        Test = Data.ix[rows,cols]
        Data_Copy = Data.copy()
        Data_Copy.at[Test.index,Test.columns] = np.nan
        Train = Data_Copy
        if (Train.notnull().sum(axis = 1) ==0 ).sum() == 0:
#             if Data.iloc[rows,cols].count().sum() > 10:
            Test_indices.append(Data.loc[Test.index,Test.columns])
            Train_indices.append(Train)
            c += 1
            #print('c = {}'.format(c))       
        counter += 1
        #print(counter, end = ',')
    return Train_indices, Test_indices

def take_input(file_name):
    '''Read From the File, ISBN Per Line (For Online Version only)'''
    isbn = []
    rat = []
    target = open(file_name,mode='r',encoding='utf8')
    lines = target.readlines()
    for line in lines:
        line = line[:-1]
        isbn.append (str(line[:-2]))
        rat.append (int(line[-1]))
    return isbn,rat

def baseline(bd):
    '''Create the Baseline estimators for
    each User and item to make better predictions'''
    mean= np.mean(bd._get_numeric_data().mean())
    col_wise_mean = np.array(bd.mean(skipna=True,axis=0))
    row_wise_mean = np.array(bd.mean(skipna=True,axis=1))
    bd_df= pd.DataFrame(np.zeros(shape=(bd.shape[0],bd.shape[1])))
    bd_df.columns=bd.columns
    bd_df.index = bd.index
    for i in range(len(row_wise_mean)):
        for j in range(len(col_wise_mean)):
            bd_df.iloc[i,j] = mean+(row_wise_mean[i]-mean)+(col_wise_mean[j]-mean)
    return bd_df

def convert_standard_normal(data):
    ''' Subtract Each row from its Mean'''
    return data.sub(data.mean(axis=1), axis=0)

def similar_items(nulldata,df,user,item):
    ''' Find the Most Similar Items wrt the Item of Interest
    alongside their cosine similarity score'''
    related_items =  nulldata.loc[:,str(user)] [nulldata.loc[:,str(user)].notnull()].index
    item_of_interest = df.ix[item]
    item_of_interest = pd.DataFrame([item_of_interest],[item])
    other_items = df.ix[related_items]
    cosine = cosine_similarity(item_of_interest,other_items).ravel()
    ranking = cosine.argsort()[-3:][::-1]
    if len(ranking)>2:
        most_sim_item = [related_items[ranking[0]],related_items[ranking[1]],related_items[ranking[2]]]
        respective_score = [cosine[ranking[0]],cosine[ranking[1]],cosine[ranking[2]]]
    else:
        if len(ranking)>1:
            most_sim_item = [related_items[ranking[0]],related_items[ranking[1]]]
            respective_score = [cosine[ranking[0]],cosine[ranking[1]]]
        else:
            most_sim_item = [related_items[ranking[0]]]
            respective_score = [cosine[ranking[0]]]
    return most_sim_item, respective_score

def predict(org_data,baseline_data,data,user,item):
    '''Predict the rating using the formula'''
    universal_rating = np.nanmean(org_data.values)
    related_items =  org_data.loc[:,str(user)] [org_data.loc[:,str(user)].notnull()].index
    if len(related_items)>1:
        neighbour, sij = similar_items(org_data,data,user,item)
        r_xj = np.zeros(len(neighbour))
        b_xj = np.zeros(len(neighbour))
        counter=0
        for item in neighbour:
            b_xj[counter] = baseline_data.loc[item,user]
            counter+=1
        for i in range(len(neighbour)):
            r_xj[i] =(org_data.loc[neighbour[i],str(user)])
        to=0
        for i in range(len(neighbour)):
            to+= sij[i]*((r_xj[i])-(b_xj[i]))
        return baseline_data.loc[item,user]+(to/sum(sij))
    else:
        return universal_rating
    
def normalize_ratings(filled_data):
    return ((filled_data-np.min(filled_data.values))/(np.max(filled_data.values)-np.min(filled_data.values)))*10

def cal_error(key):
    '''Calculate Error for a Specific Train Set'''
    train = pd.read_csv('Filled_Matrix'+str(key)+'.csv',encoding='utf8')
    train.set_index('Book',inplace=True)
    test = pickle.load(open('Test'+str(key)+'.p','rb'))
    nr = normalize_ratings(train)
    return (np.nansum(abs(nr.loc[test.index,test.columns]-test).values))/test.notnull().sum().sum()/10

def online_pred(null_data,std_normal_data,item):
    neb,sij = similar_items(null_data,std_normal_data,'883',item)
    bxi = baseline_data.loc[item,'883']
    bxj = np.zeros(shape=len(neb))
    rxj = np.zeros(shape=len(neb))
    for i in range(len(neb)):
        bxj [i] = baseline_data.loc[neb[0],'883']
        rxj [i] = std_normal_data.loc[neb[0],'883']
    tot = 0
    for i in range(len(neb)):
        tot+= (sij[i]*(rxj[i]-bxj[i]))/sij[i]
    return tot+bxi

def book_data():
    '''Save the Book Data to print Results'''
    Books_data=read_Books('BX-Books.csv')
    books =Books_data.iloc[:,0:3]
    books.columns = ['ISBN','Title','Author']
    books.set_index('ISBN',inplace=True)
    books.to_csv('Books.csv',encoding='utf8')


# # Offline Version Starts

# In[431]:

'''Read the Initial Data and Reduce it greatly'''
Ratings= read_ratings('BX-Book-Ratings.csv')
Ratings.columns=['User','Book','Rating']
new_data = filter_data(Ratings)
pivot_data = new_data.pivot(index='Book',columns='User',values='Rating')
pivot_data.to_csv('pivot.csv',encoding='utf8')
print ('New Data has been Created')


# In[432]:

'''Generate 5 Training & Test Datasets'''
data = pd.read_csv('pivot.csv',encoding='utf8')
data.set_index('Book',inplace=True)
Trains,Tests = find_samples(data.T)
for i in range(5):
    pickle.dump(Trains[i].T,open('Train{}.p'.format(i),'wb'))
    pickle.dump(Tests[i].T,open('Test{}.p'.format(i),'wb'))
    print('Train{}.p and Test{}.p have been created!'.format(i,i))


# In[433]:

'''Calculate the baseline estimator DF'''
bd_data = baseline(data)
bd_data.to_csv('Baseline.csv',encoding ='utf8')
print ('Baseline Estimators have been Created')


# In[434]:

'''Get a Specific Train Set and Estimate Ratings'''
data = pickle.load(open('Train4.p','rb'))
std_normal_data = convert_standard_normal(data)
org_data = data
std_normal_data.fillna(0,inplace=True)
filled_data = std_normal_data
baselin_data = pd.read_csv('Baseline.csv',encoding='utf8')
baselin_data.set_index('Book',inplace=True)


# In[435]:

'''Find out the Positions of NAN values'''
pos = list(zip(*data.as_matrix().nonzero()))
array= []
for c_pos in pos:
    row_id = data.iloc[c_pos[0],:].name
    col_id = data.iloc[:,c_pos[1]].name
    array.append([row_id,col_id])
print ('Positions to be filled have been found')


# In[ ]:

'''Fill the Matrix with Estimated Predictions'''
for position in array:
    data.loc[position[0],position[1]] = predict(org_data,bd_data,std_normal_data,position[1],position[0])
data.to_csv('Filled_Matrix4.csv',encoding='utf8')
print ('Matrix4 has been Filled')


# In[429]:

'''Calculate MAE for all the training sets'''
MAE = np.ones(5)
for key in range(5):
    MAE[key] = (cal_error(key))
print ('Mean Absolute Error is :') 
np.mean(MAE)


# ## Online Version Starts

# In[426]:

'''Some Pre Processing for Online Version'''
f0 = pd.read_csv('Filled_Matrix0.csv',encoding='utf8')
f1 = pd.read_csv('Filled_Matrix1.csv',encoding='utf8')
f2 = pd.read_csv('Filled_Matrix2.csv',encoding='utf8')
f3 = pd.read_csv('Filled_Matrix3.csv',encoding='utf8')
f4 = pd.read_csv('Filled_Matrix4.csv',encoding='utf8')
f0.set_index('Book',inplace=True)
f1.set_index('Book',inplace=True)
f2.set_index('Book',inplace=True)
f3.set_index('Book',inplace=True)
f4.set_index('Book',inplace=True)
f0 = f0.add(f1)
f0 = f0.add(f2)
f0 = f0.add(f3)
f0 = f0.add(f4)
f0 = f0/5
print ('Pre Processing Done')


# In[428]:

books = pd.read_csv('Books.csv',encoding='utf8')
books.set_index('ISBN',inplace=True)
new_user = pd.DataFrame(np.zeros(f0.iloc[:,0].shape))
new_user.index = f0.index
new_user.columns =['883']
new_user.iloc[:,:] = np.nan
'''Fill the Input File Randomly'''
random_indices = random.sample(range(504), 70)
indices = new_user.iloc[random_indices,:].index
values = np.random.randint(10, size=(len(indices)))
target = open('input.txt',mode='w',encoding='utf8')
for i in range(len(indices)):
    target.write(indices[i]+' '+str(values[i]))
    target.write('\n')
target.close()
#print ('Input File is ready to Use')
isbn, rat = take_input('input.txt')
for i in range(len(isbn)):
    new_user.loc[isbn[i],:] = rat[i]
f0['883'] = new_user.iloc[:,0]
'''Some More Pre Processing'''
baseline_data = baseline(f0)
null_data = f0
std_normal_data = convert_standard_normal(f0)
std_normal_data.fillna(0,inplace=True)
filled_data = f0
'''Find Out NAN'''
nan_indices = []
for i in range(f0.shape[0]):
    if f0.iloc[i,-1]!=f0.iloc[i,-1]:
        nan_indices.append(str(f0.iloc[i,:].name))
for index in nan_indices:
    filled_data.loc[index,'883'] = online_pred(null_data,std_normal_data,index)
estimated = filled_data.loc[nan_indices,'883'].values
sorting = estimated.argsort()[-3:][::-1]
print (books.loc[str(nan_indices[sorting[0]]),'Title']+' Written by '+books.loc[str(nan_indices[sorting[0]]),'Author'])
print (books.loc[str(nan_indices[sorting[1]]),'Title']+' Written by '+books.loc[str(nan_indices[sorting[1]]),'Author'])
print (books.loc[str(nan_indices[sorting[2]]),'Title']+' Written by '+books.loc[str(nan_indices[sorting[2]]),'Author'])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



