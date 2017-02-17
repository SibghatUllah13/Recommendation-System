
# coding: utf-8

# In[3]:

#Preprocessing
import numpy as np
import math 
import pandas as pd
import pickle 

def pre_processing(train_file_names):
    '''Initial Pre Processing'''
    for i in train_file_names:
        Train=(pickle.load(open(i,'rb'))).T
        Train.index.name='Book'
        r,n=Train.shape
        col_mean=np.matrix(Train.mean())       
        Train_col=Train.values-col_mean 
        '''Subtract Column mean from respective columns'''
        row_mean=np.matrix(Train.mean(axis=1))
        Train_processed=Train_col-row_mean.T
        '''Subtract Row mean from resulting matrix'''
        new_data=pd.DataFrame(Train_processed, columns=Train.columns)
        new_data.set_index(Train.index,inplace=True)
        new_data.to_pickle(i+'kl')                              
        pd.DataFrame(col_mean).to_pickle('Column_mean_'+i+'kl')        
        '''Save Column mean for denormalization'''
        pd.DataFrame(row_mean).to_pickle('Row_mean_'+i+'kl') 
        '''Save Row mean for denormalization'''
        del new_data
        
        
def stochastic_descent(U,P,indices,mu,la,M_orig,max_iter):
    '''Performs Stochastic Gradient Descent and Returns 
    Optimized Matrices'''
    total_cost=[]  
    '''total cost after complete one iteration'''
    ''' prev_cost=[]  cost function at every observation before updating new values of U and P for one iteration''' 
    '''new_cost=[]   cost function at every observation after updating new values of U and P for one iteration '''
    
    err=0
    err_new=0


    for x in range(max_iter):
        for i,j in indices:
            err=2*(M_orig[i,j]-(U[i,:]*P[:,j]))
            '''prev_cost.append(np.square(err/2)[0,0]+la*(np.sum(np.square(P[:,j]))+np.sum(np.square(U[i,:])))) '''
            U_temp=U[i,:]+mu*((err[0,0]*P[:,j].T)-(la*U[i,:]))
            P_temp=P[:,j]+mu*(err[0,0]*U[i,:].T-la*P[:,j])
            U[i,:]=U_temp
            P[:,j]=P_temp
            ''' err_new=2*(M_orig[i,j]-(U[i,:]*P[:,j]))'''
            
            '''new_cost.append(np.square(err_new/2)[0,0]+la*(np.sum(np.square(P[:,j]))+np.sum(np.square(U[i,:])))) #cost function at every observation after updating new values of U and P for one iteration '''        
        error=0
        P_sum=0
        U_sum=0
        for i,j in indices:
            error=error+np.square((M_orig[i,j]-(U[i,:]*P[:,j])))
            P_sum=P_sum+np.sum(np.square(P[:,j]))
            U_sum=U_sum+np.sum(np.square(U[i,:]))
        
        error=error+la*(P_sum+U_sum)
        total_cost.append(error[0,0])
    
    return U,P,total_cost


def gen_predictions(Preds,Row_mean,Col_mean,Ratings_index,Ratings_columns):
    '''Returns De Normalized Predictions'''
    Preds=Preds.values
    Row_mean=Row_mean.values
    Col_mean=Col_mean.values
    final_preds=Preds+Row_mean.T+Col_mean                                       #Denormalize the predicted matrix
    Final_predictions=pd.DataFrame(final_preds,index=Ratings_index, columns=Ratings_columns)
    return Final_predictions


def gen_ISBN(Final_predictions,indices,Ratings_index,Ratings_columns):
    '''Generates Sorted ISBN Based on Estimated Predictions'''
    Sorted_Predictions=Final_predictions.values
    Sorted_Predictions[indices[:,0],indices[:,1]]=-float('inf')
    Sorted_Predictions=pd.DataFrame(Sorted_Predictions,index=Final_predictions.index, columns=Final_predictions.columns)
    
    Book_ISBN=pd.DataFrame()
    for i in Sorted_Predictions.columns:
        data=pd.DataFrame(Sorted_Predictions.sort_values(i,ascending=False).index)
        Book_ISBN=Book_ISBN.append([data.T])
    
    Book_ISBN.set_index(Ratings_columns, inplace=True)
    
    return(Book_ISBN)



def gen_final_recommendations(Final_predictions,indices):
    '''Generates Sorted Predictions'''
    Sorted_Predictions=Final_predictions.values
    Sorted_Predictions[indices[:,0],indices[:,1]]=-float('inf')
    Sorted_Predictions=pd.DataFrame(Sorted_Predictions,index=Final_predictions.index, columns=Final_predictions.columns)
    #Sort the dataframe
    Sorted_preds_df=pd.DataFrame([Sorted_Predictions[col].order(ascending = False).reset_index(drop=True) for col in Sorted_Predictions])
    #Map ratings between 0 to 10
    indices_inf=np.argwhere(np.isinf(Sorted_preds_df.values)==True)
    Sorted_preds_df.iloc[indices_inf[:,0],indices_inf[:,1]]=0
    Sorted_preds_df=((Sorted_preds_df-np.min(np.min(Sorted_preds_df)))/(np.max(np.max(Sorted_preds_df))-np.min(np.min(Sorted_preds_df))))*10
    Sorted_preds_df.iloc[indices_inf[:,0],indices_inf[:,1]]=-float('inf')
    return Sorted_preds_df



def gen_indices(Test0,Test1,Test2,Test3,Test4):
    '''Generate indices to index nonzero values in Training and Test set '''
    tmp0=np.transpose(np.nonzero(Test0.values))
    Test0_indices=np.mat((Test0.index[tmp0[:,0]],Test0.columns[tmp0[:,1]])).T
    tmp1=np.transpose(np.nonzero(Test1.values))
    Test1_indices=np.mat((Test1.index[tmp1[:,0]],Test1.columns[tmp1[:,1]])).T
    tmp2=np.transpose(np.nonzero(Test2.values))
    Test2_indices=np.mat((Test2.index[tmp2[:,0]],Test2.columns[tmp2[:,1]])).T
    tmp3=np.transpose(np.nonzero(Test3.values))
    Test3_indices=np.mat((Test3.index[tmp3[:,0]],Test3.columns[tmp3[:,1]])).T
    tmp4=np.transpose(np.nonzero(Test4.values))
    Test4_indices=np.mat((Test4.index[tmp4[:,0]],Test4.columns[tmp4[:,1]])).T
    return Test0_indices,Test1_indices,Test2_indices,Test3_indices,Test4_indices


def mae(Train0,Train1,Train2,Train3,Train4,Test0,Test1,Test2,Test3,Test4):
    '''Mean absolute error function '''
    Test0_indices,Test1_indices,Test2_indices,Test3_indices,Test4_indices=gen_indices(Test0,Test1,Test2,Test3,Test4)
    mae0=0
    for i in Test0_indices:
        x=np.ravel(i)
        mae0=mae0+abs(Train0.loc[x[0],x[1]]-Test0.loc[x[0],x[1]])
    mae0=(mae0/Test0_indices.shape[0])
    mae1=0
    for i in Test1_indices:
        x=np.ravel(i)
        mae1=mae1+abs(Train1.loc[x[0],x[1]]-Test1.loc[x[0],x[1]])
    mae1=(mae1/Test1_indices.shape[0])
    mae2=0
    for i in Test2_indices:
        x=np.ravel(i)
        mae2=mae2+abs(Train2.loc[x[0],x[1]]-Test2.loc[x[0],x[1]])
    mae2=(mae2/Test2_indices.shape[0])
    mae3=0
    for i in Test3_indices:
        x=np.ravel(i)
        mae3=mae3+abs(Train3.loc[x[0],x[1]]-Test3.loc[x[0],x[1]])
    mae3=(mae3/Test3_indices.shape[0])
    mae4=0
    for i in Test4_indices:
        x=np.ravel(i)
        mae4=mae4+abs(Train4.loc[x[0],x[1]]-Test4.loc[x[0],x[1]])
    mae4=(mae4/Test4_indices.shape[0])
    return ((mae0+mae1+mae2+mae3+mae4)/5)


# ## Offline Version Starts

# In[36]:

'''Pre-processing'''
train_file_names=['Train0svd.p','Train1svd.p','Train2svd.p','Train3svd.p','Train4svd.p']
pre_processing(train_file_names)                         


# In[24]:

'''Load the Training Data into RAM'''
Ratings=pickle.load(open('Train0svd.pkl','rb'))         #Pre processed matrix of ratings
Ratings.index.name='Book'
Ratings.fillna(0,inplace=True)

'''SVD Matrix Initialization'''
r,c=Ratings.shape
k=100
U=np.matrix(np.random.rand(r,k)*np.sqrt(5/k))
P=np.matrix(np.random.rand(k,c)*np.sqrt(5/k))
M_orig=Ratings.values
indices=np.transpose(np.nonzero(M_orig))

'''Stochastic Gradient Descent Running'''
User,Items,t_cost=stochastic_descent(U,P,indices,0.003,0.2,M_orig,400)
Preds=User*Items
pd.DataFrame(Preds).to_pickle('Train0Predictions_0.003_0.2_400.pkl')
Preds=pickle.load(open('Train0Predictios_0.003_0.2_400.pkl','rb'))

'''De Normalizing the Estimated Predictions'''
Row_mean=pickle.load(open('Row_mean_Train0svd.pkl','rb'))
Col_mean=pickle.load(open('Column_mean_Train0svd.pkl','rb'))
Final_predictions=gen_predictions(Preds,Row_mean,Col_mean, Ratings.index,Ratings.columns)
Final_predictions.to_pickle('Train0Final_Predictions.pkl') #unsorted

'''Sorting ISBN '''
Book_ISBN=gen_ISBN(Final_predictions,indices,Ratings.index,Ratings.columns)
Book_ISBN.to_pickle('Train0Book_ISBN.pkl')

'''Final Offline Version is completed and Saved into Disk'''
Sorted_preds_df=gen_final_recommendations(Final_predictions,indices)
Sorted_preds_df.to_pickle('Train0Final_Recommendations.pkl')

del Ratings,Preds,Final_predictions,Book_ISBN,Sorted_preds_df


# In[4]:

# Load Training files

Train0=(pickle.load(open('Train0Final_Predictions.pkl','rb'))).T            #Train file 0
Train0.fillna(0,inplace=True)
Train1=(pickle.load(open('Train1Final_Predictions.pkl','rb'))).T            #Train file 1        
Train1.fillna(0,inplace=True)
Train2=(pickle.load(open('Train2Final_Predictions.pkl','rb'))).T            #Train file 2
Train2.fillna(0,inplace=True)
Train3=(pickle.load(open('Train3Final_Predictions.pkl','rb'))).T            #Train file 3
Train3.fillna(0,inplace=True)
Train4=(pickle.load(open('Train4Final_Predictions.pkl','rb'))).T            #Train file 4
Train4.fillna(0,inplace=True)

#Load Test files
Test0=pickle.load(open('Test0svd.p','rb'))         
Test0.fillna(0,inplace=True)
Test1=pickle.load(open('Test1svd.p','rb'))         
Test1.fillna(0,inplace=True)
Test2=pickle.load(open('Test2svd.p','rb'))         
Test2.fillna(0,inplace=True)
Test3=pickle.load(open('Test3svd.p','rb'))         
Test3.fillna(0,inplace=True)
Test4=pickle.load(open('Test4svd.p','rb'))         
Test4.fillna(0,inplace=True)
mae_total=mae(Train0,Train1,Train2,Train3,Train4,Test0,Test1,Test2,Test3,Test4)
mae_total/10


# In[ ]:



