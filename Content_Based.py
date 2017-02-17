
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def convert_lower(sentence):
    '''Lowercases the input'''
    return str.lower(sentence)

def remove_punctuation(sentence):
    '''Remove punctuation from the input'''
    tokenizer = RegexpTokenizer(r'\w+')
    l=tokenizer.tokenize(sentence)
    return (" ".join(l))

def tokenize(sentence):
    '''Tokenizes the input'''
    return nltk.word_tokenize(sentence)

def remove_stop_words(word_list):
    '''Removes Stop Words'''
    array=[]
    for word in word_list:
        if word not in stopwords.words('english'):
            array.append(word)
    return array

def clean_single_Item(Single_recipe):
    '''Uses Natural Language Processing for cleaning the input'''
    new_recipe=[]
    stemmer = PorterStemmer()
    for recipe_element in Single_recipe:
        temp=recipe_element
        temp=convert_lower(temp)
        temp=remove_punctuation(temp)
        temp=tokenize(temp)
        temp=remove_stop_words(temp)
        final_words=[]
        for word in temp:
            final_words.append(stemmer.stem(word))
        final_words=" ".join(final_words)
        new_recipe.append(final_words)
    return new_recipe

def write_to_file(Recipe,file_name):
    '''Writes on the HD the document'''
    target=open(file_name,'a',encoding='utf-8')
    for element in Recipe:
        target.write(str(element))
        target.write('\n')
    target.close()
    
def clean_all_Item(recipe_file,output_file):
    '''Uses subfunctions to clean all the items for TFIDF'''
    target=open(recipe_file,'r',encoding='utf-8')
    counter=0
    Single_Recipe=[]
    for ind_element in target:
        if ind_element=="\n":
            Single_Recipe.append("No Information")
        else:
            Single_Recipe.append(ind_element)
        counter+=1
        if counter>=3:
            nice_recipe=clean_single_Item(Single_Recipe)
            write_to_file(nice_recipe,output_file)
            counter=0
            Single_Recipe[:]=[]
    target.close()

def read_Users(file_name):
    '''Loads Users Dataset'''
    cwd=os.getcwd()
    path=cwd+"\\"+file_name
    data_frame=pd.read_csv(path,sep=';',encoding='utf-8',header=0)
    return data_frame

def read_Books(file_name):
    '''Loads Books Dataset '''
    cwd=os.getcwd()
    path=cwd+"\\"+file_name
    data_frame=pd.read_csv(path,sep=';',encoding='utf-8',error_bad_lines=False)
    return data_frame

def read_ratings(file_name):
    '''Loads Ratings Dataset'''
    cwd=os.getcwd()
    path=cwd+"\\"+file_name
    data_frame=pd.read_csv(path,sep=';',encoding='utf-8',header=0)
    return data_frame

def run():
    '''Loads all the initial data'''
    User_data=read_Users('BX-Users.csv')
    User_data.columns = ['User-ID','Location','Age']
    Books_data=read_Books('BX-Books.csv')
    Ratings= read_ratings('BX-Book-Ratings.csv')
    Ratings.columns = ['User','Book','Rating']
    return User_data,Books_data,Ratings

def clean_data():
    '''Loads and organizes the 3 initial datasets'''
    User,Book,Ratings=run()
    Book=Book.ix[:,0:5]
    Book=Book.dropna()
    Book.columns = ['ISBN','Title','Author','Year','Publisher']
    User.set_index('User-ID',inplace = True)
    Book.set_index('ISBN',inplace=True)
    return User,Book,Ratings

def get_items_of_interest(Train, books):
    '''Takes Train set and books dataframe and returns
    itemset without NaNs inside it ready for creating Item Profiles'''
    books = books.loc[Train.columns,:].dropna()  # Remove NaNs
    mask = (books.Year == '0')
    mask2 = (books.Year == 0)
    books.Year[mask] = round(np.mean(books.Year.values.astype('int'))) # Interpolates missing years
    books.Year[mask2] = round(np.mean(books.Year.values.astype('int'))) # Interpolates missing years
    return books

def write_Items(file_name, Items):
    '''Writes documents to the HD. Each written on 3 new lines'''
    f = open(file_name, 'w', encoding='utf8')
    for i in range(len(Items)):
        f.write(str(Items.Title[i])+'\n')
        f.write(str(Items.Author[i])+'\n')
        f.write(str(Items.Publisher[i])+'\n')
    f.close()

def read_3line_items(file_name):
    '''Takes a file name where the text file contains each document distributed on 3 new lines each and gives
    a list containing strings of documents as elements as output'''
    l = []
    f = open(file_name, 'r', encoding='utf8')
    lines = f.readlines()
    f.close
    for i in range(len(lines)//3):
        l.append(str(lines[i*3])[:-1]+' ' + str(lines[i*3+1])[:-1]+' '+str(lines[i*3+2])[:-1])
    return l

def create_tf_idf(file_name):
    '''Takes a list as input in which each element is a string representing each document and
    gives tfidf np.array as outpput'''
    corpus = read_3line_items(file_name)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X.toarray())
    return tfidf.toarray()

def rated_items(filtered, shape, user_ids):
    '''Takes utility matrix and returns users and the corresponding books he/she has read(ISBN)'''
    user_item = pd.DataFrame(np.zeros(shape=shape))
    for i in range(filtered.shape[1]):
        if len((filtered.iloc[:,i].dropna().index))>1:
            user_item.iloc[i,:len(filtered.iloc[:,i].dropna().index)]= filtered.iloc[:,i].dropna().index.values
        else:
            user_item.iloc[i,:len(filtered.iloc[:,i].dropna().index)]= str(filtered.iloc[:,i].dropna().index.values)[2:-2]
    user_item.index = user_ids
    return user_item

def Year(Read_Books,Items):
    '''Gives the matrix which contains users and corresponding item's publication years '''
    Years = pd.DataFrame(np.zeros(Read_Books.shape), index = Read_Books.index, columns=Read_Books.columns)
    for i in range(Years.shape[0]):
        for j in range((Read_Books.iloc[i] != 0).sum()):
            if Read_Books.iloc[i,j] != 0:
                Years.iloc[i,j] = Items[Read_Books.iloc[i,j]]
    return Years

def AVGYear(Rec_System,Read_Books,Items):
    '''Gives average year for each user'''
    Years = Year(Read_Books, Items)
    Yearavg = pd.DataFrame(np.zeros((Rec_System.shape[0],1)),index = Rec_System.index, columns=['AVGYear'])
    for i in range(Yearavg.shape[0]):
        Yearavg.iloc[i,0] = np.mean(Years.iloc[i][Years.iloc[i]>0])
    return Yearavg

def Diff_Matrix(Rec_System,Read_Books,Items):
    '''Gives User by Item matrix and values are difference of years between average of
    year of publication of read books and all books'''
    AVGYears = AVGYear(Rec_System,Read_Books,Items)
    YearDiff = pd.DataFrame(np.zeros((Read_Books.shape[0],len(Items))),index = Read_Books.index, columns=Items.index)
    for i in range(YearDiff.shape[0]):
        YearDiff.iloc[i] = abs((AVGYears.iloc[i,0] - Items).values)
    YearDiff=(YearDiff-YearDiff.min().min())/(YearDiff.max().max()-YearDiff.min().min())
    return YearDiff

def ISBN_REC_SYS(Rec_System,Read_Books,YearDiff,itz,alfa):
    '''Recommendations for each user. Values are ISBNs (actually
    values are numbers which can be translated to ISBNs with specific table)'''
    l = np.zeros(Rec_System.shape[1])
    for i in range(Rec_System.shape[0]):
        leng = np.array((Rec_System.iloc[i,:].drop(Read_Books.iloc[i,:][Read_Books.iloc[i,:] != 0].values))).shape[0]
        empty = np.ones(Rec_System.shape[1],dtype='int32')*-1
        ISBNS = np.array((Rec_System.iloc[i,:]+alfa*YearDiff.iloc[i,:]).sort_values(ascending = False).
                 drop(Read_Books.iloc[i,:][Read_Books.iloc[i,:] != 0].values).index)
        empty[:leng] = itz.loc[ISBNS].values.flatten()
        l = np.vstack((l,empty))
    return l[1:,:]

def RATINGS_REC_SYS(Rec_System,Read_Books,YearDiff,alfa):
    '''Recommendations for each user. Values are the predicted ratings'''
    l = np.zeros(Rec_System.shape[1])
    for i in range(Rec_System.shape[0]):
        leng = np.array((Rec_System.iloc[i,:].drop(Read_Books.iloc[i,:][Read_Books.iloc[i,:] != 0].values))).shape[0]
        empty = np.ones(Rec_System.shape[1])*-1
        empty[:leng] = np.array((Rec_System.iloc[i,:]+alfa*YearDiff.iloc[i,:]).sort_values(ascending = False).
                 drop(Read_Books.iloc[i,:][Read_Books.iloc[i,:] != 0].values).values)
        l = np.vstack((l,empty))
    return l[1:,:]

def find_samples(Data):
    '''Searches for 5 train and test sets so that when the test set is held out from the train set,
    train sets contains at least one non-NaN value for each user'''
    Train_indices = []
    Test_indices = []
    counter = 0
    c = 0
    rows = (Data.notnull().sum(axis=1).sort_values(ascending = False)>1)[:Data.shape[0]//5].index
    while c != 5:
        cols = np.random.choice(Data.shape[1],Data.shape[1]//5, replace=False)
        Test = Data.ix[rows,cols]
        Data_Copy = Data.copy()
        Data_Copy.at[Test.index,Test.columns] = np.nan
        Train = Data_Copy
        if (Train.notnull().sum(axis = 1) == 0).sum() == 0:
            Test_indices.append(Data.loc[Test.index,Test.columns])
            Train_indices.append(Train)
            c += 1
            print('c = {}'.format(c))       
        counter += 1
        print(counter, end = ',')
    return Train_indices, Test_indices

def filter_items(df,min2):
    '''Filters data so that each book is rated at least by min2 users'''
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

def filter_data(ratings,min1 = 70,min2 = 10):
    '''Filters data so that each user has rated at least min1 book and each book is rated at least by min2 users'''
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

def MAE(filename1,filename2,filename3,filename4):
    '''Takes pickle file names and gives Mean Absolute Error'''
    '''Test.p file, Scores.p file, ISBNS.p file, Items.p file'''
    Test = pickle.load(open(filename1,'rb'))
    RATINGS_RECOMMENDATIONS = pickle.load(open(filename2,'rb'))
    ISBN_RECOMMENDATIONS = pickle.load(open(filename3,'rb'))
    itz = pickle.load(open(filename4,'rb'))
    Test.dropna(axis = 0, how = 'all', inplace = True)
    MAE = 0
    for i in range(Test.shape[0]):
        user = Test.iloc[i].name  #User id
        keys = ISBN_RECOMMENDATIONS.loc[user].values  # ISBNS for that user
        keys = keys[keys!=-1]     # ISBN keys which was not read by that user
        keys = np.array(keys[keys!=-1],dtype='int32')   # Array of ISBN keys which was not read by that user
        predicted_books = itz.reset_index().iloc[keys].ISBN.values  #Books not read (to be recommended)
        scores = RATINGS_RECOMMENDATIONS.loc[user][:len(predicted_books)]
        predicted_scores = pd.Series(scores.values, index = predicted_books)

        mask = Test.iloc[i,:].notnull()
        books_read, actual_ratings = Test.iloc[i][mask].index, Test.iloc[i][mask].values 
        
        MAE += abs((predicted_scores[books_read].values - actual_ratings)).sum()
    return MAE/Test.notnull().sum().sum()

def modif_single_user(ratings_filtered,user_id, items):
    '''Creates user profile for one user'''
    ratings = ratings_filtered.loc[user_id].dropna()
    centered =  ratings - ratings.mean() #centered ratings for the user
    read = centered.index #books read by the user
    call = items.loc[read].loc[:,items.loc[read].sum()>0].columns.astype('int32') #vector containing only positive sum column indices
    user = np.zeros(items.shape[1])
    user[call] = (items.loc[read].iloc[:,call].
                  apply(lambda x: x*centered)/
                  items.loc[read].
                  iloc[:,call].
                  sum()).sum()
    return user
    
def user_profs(Ratings, filename, Items,alfa = 1):
    '''Creates User Profile Matrix'''
    items = item_profile(filename, Items,Ratings,alfa)
    a = pd.DataFrame(np.zeros((Ratings.shape[1],items.shape[1])))
    Rate = Ratings.T
    for i in range(Rate.shape[0]):
        a.iloc[i] = (modif_single_user(Rate, Ratings.columns[i], items))
        print(i, end=',')
    return a, items
    
def predictions(Train,filename, Items, user_ids, book_ids, alfa = 1):
    '''Creates prediction matrix'''
    User_Profiles,item_profiles = user_profs(Train, filename, Items, alfa) 
    Rec_System = pd.DataFrame(cosine_similarity(User_Profiles,item_profiles), index = user_ids, columns=book_ids)
    return Rec_System, item_profiles

def item_profile(filename, Items, Train, alfa=1):
    '''Creates item profiles'''
    tfidf = create_tf_idf(filename)  #Create TFIDF
    item_profiles = pd.DataFrame(tfidf) #Dataframe format
    item_profiles['ISBN'] = Items.index #Add ISBN column
    item_profiles.set_index('ISBN',inplace=True) #ISBNs as index
    avg_rating = np.zeros(Train.shape[0]) #Empty matrix
    avg_all = np.nanmean(Train.values) #Mean Rating
    for i in range(item_profiles.shape[0]):
        book = item_profiles.index[i]
        if len(Train.loc[book].dropna()) >0:
            avg_rating[i] = Train.loc[book].mean()  #average rating of the book
        else:
            avg_rating[i] = avg_all #average rating of all the books
    item_profiles['{}'.format(item_profiles.shape[1])] = alfa*avg_rating
    return item_profiles

def single_user(user):
    '''User profile for online version'''
    ratings = user.Rating
    centered =  ratings - ratings.mean() #centered ratings for the user
    read = centered.index #books read by the user
    call = (items.loc[read].loc[:,items.loc[read].sum()>0].
            columns.astype('int32')) #vector containing only positive sum column indices
    user = np.zeros(items.shape[1]) #empty vector
    user[call] = (items.loc[read].iloc[:,call]. #select only populatable columns
                  apply(lambda x: x*centered)/  #apply function of weighted average
                  items.loc[read].
                  iloc[:,call].
                  sum()).sum()
    return user

def Recommend(sim,Items,user,alfa = 50):
    '''Recommendation output for online version user'''
    readbooks = user.index #Books read by the user
    AVGY = np.mean(Items.loc[readbooks]) #Average year for this user
    Yeardiff = np.abs(Items - AVGY) #Compute year difference for each book
    Yearsim = (1 - ((Yeardiff - Yeardiff.min())/(Yeardiff.max()-Yeardiff.min()))) #Standardize
    Yearsim = Yearsim.drop(readbooks) #Only not read books
    data = pd.Series(sim, index = Items.index).drop(readbooks) #Cosine similarities
    data = data + (alfa*Yearsim) # Total similarity
    data = (data - data.min())/(data.max()-data.min())*10  #Standardize
    data.sort_values(0, ascending = False, inplace = True) 
    print('Recommended books: {}'.format(data[:5].index.values))
    print('Predicted ratings: {}'.format(np.round(data[:5].values.flatten(),1)))


# # Loading the Data

# In[21]:

#Loading Files
users, books, ratings = clean_data()
# ratings.set_index('User',inplace = True)


# In[22]:

# df = filter_data(ratings)


# In[23]:

# df.pivot('User','Book','Rating').to_csv('Ratings.csv',encoding='utf8')


# # Create Train and Test

# In[24]:

# # # Load Data
# Data = pd.read_csv('Ratings.csv',encoding='utf8')
# Data.set_index('User',inplace = True)


# In[25]:

# #Taking only those books in the table that are present in books dataframe.
# #Also removing those rows that have only NaN values inside them
# intersection = Data.columns.intersection(books.index)
# Data = Data.loc[:,intersection]
# Data = Data.loc[((Data.notnull() > 0).sum(axis = 1) > 0),:]


# In[26]:

# Creating Train and Test sets

# Trains, Tests = find_samples(Data)


# In[27]:

# # #Checking if Tests are not fully overlapping

# for i in range(5):
#     for j in range(i+1,5):
#         print((sorted(set(Tests[i].columns))==sorted(set(Tests[j].columns))))


# In[28]:

# # Save Train and Test datasets on the HD
# for i in range(5):
#     pickle.dump(Trains[i],open('Train{}.p'.format(i),'wb'))
#     pickle.dump(Tests[i],open('Test{}.p'.format(i),'wb'))
#     print('Train{}.p and Test{}.p have been created!'.format(i,i))


# In[34]:

# Load Train and Test Datasets
Testing = int(input(prompt = 'Input the number of Train set'))
t = time.clock()
Train, Test = pickle.load(open('Train{}cb.p'.format(Testing),'rb')), pickle.load(open('Test{}cb.p'.format(Testing),'rb'))


# In[35]:

# ratings = 0
# Data = 0


# # Predict the Scores for each User

# In[36]:

Items = get_items_of_interest(Train,books) #Only the books which are in the Train Set
Items.Year = Items.Year.astype('int')


# In[37]:

# #Save Items to the HD
# pickle.dump(Items.reset_index()['index'],open('Items.p','wb'))


# In[38]:

# # Create corpus for TFIDF Transformer

# write_Items('Dirty_Text.txt',Items)
# clean_all_Item('Dirty_Text.txt','Clear_Text.txt')


# In[39]:

Train = Train.T


# In[40]:

del books
del Test


# In[41]:

user_ids, book_ids = Train.columns, Train.index
pickle.dump(user_ids,open('user_ids.p','wb'))
pickle.dump(book_ids,open('book_ids.p','wb')) 


# In[42]:

Rec_System, items = predictions(Train,'Clear_Text.txt',Items, user_ids, book_ids, alfa = 0)  #Create Predicted Matrix


# In[43]:

# Normalize the result
Rec_System = ((Rec_System - np.min(Rec_System.values))/(np.max(Rec_System.values) - np.min(Rec_System.values)))*10
Rec_System.head()


# In[44]:

# Create Table that contains users and their corresponding already read books
Read_Books = rated_items(Train,(Train.shape[1],Train.notnull().sum().max()),user_ids)
Read_Books.shape


# In[45]:

# Create pd.Series of Year from items with integer type
Items = Items.Year.astype('int')


# In[46]:

YearDiff = Diff_Matrix(Rec_System,Read_Books,Items) #Create Matrix which contains year diff. between avg year published for user and each item
YearDiff.head(2)


# In[47]:

# Save table for ISBN to number and vice versa translation
itz = pd.Series(range(len(Items)), index = Items.index)
itz = pd.DataFrame({'ISBN' : itz.index, 'Key' : itz.values}).set_index('ISBN')
pickle.dump(itz,open('Items.p','wb'))


# In[48]:

# Save these three datasets on the HD
pickle.dump(itz,open('Items.p','wb'))
pickle.dump(Rec_System,open('Rec_System_modif.p','wb'))
pickle.dump(Read_Books,open('Read_Books_modif.p','wb'))
pickle.dump(YearDiff,open('YearDiff_modif.p','wb'))


# # Create Recommended ISBNs for each User

# In[49]:

#Load Files
a = pickle.load(open('Rec_System_modif.p','rb'))
b = pickle.load(open('Read_Books_modif.p','rb'))
c = pickle.load(open('YearDiff_modif.p','rb'))
d = pickle.load(open('Items.p','rb'))


# In[50]:

ISBN_RECOMMENDATIONS = ISBN_REC_SYS(a,b,c,d,50) #Create matrix of Recommendations


# In[51]:

ISBN_RECOMMENDATIONS = pd.DataFrame(ISBN_RECOMMENDATIONS) #Create DataFrame


# In[52]:

ISBN_RECOMMENDATIONS.index = pickle.load(open('user_ids.p','rb'))     #Assign labels to rows
ISBN_RECOMMENDATIONS.head(2) #Visualize


# In[53]:

pickle.dump(ISBN_RECOMMENDATIONS, open('CB_RS_ISBNS{}_modif.p'.format(Testing),'wb')) #Save to the HD


# # Create Recommended Ratings for each User

# In[54]:

#Load Files
# a = pickle.load(open('Rec_System_modif.p','rb'))
# b = pickle.load(open('Read_Books_modif.p','rb'))
# c = pickle.load(open('YearDiff_modif.p','rb'))


# In[55]:

RATINGS_RECOMMENDATIONS = RATINGS_REC_SYS(a,b,c,50) #Create matrix of Recommendations


# In[56]:

RATINGS_RECOMMENDATIONS = pd.DataFrame(RATINGS_RECOMMENDATIONS) #Create DataFrame
RATINGS_RECOMMENDATIONS.index = pickle.load(open('user_ids.p','rb'))     #Assign labels to rows


# In[57]:

# Translate to Ratings format
RATINGS_RECOMMENDATIONS = ((RATINGS_RECOMMENDATIONS-np.min(RATINGS_RECOMMENDATIONS.values))/
                            (np.max(RATINGS_RECOMMENDATIONS.values)-np.min(RATINGS_RECOMMENDATIONS.values)))*10


# In[58]:

RATINGS_RECOMMENDATIONS.head(2)  #Visualize


# In[59]:

pickle.dump(RATINGS_RECOMMENDATIONS, open('CB_RS_SCORES{}_modif.p'.format(Testing),'wb')) #Save to the HD


# # Test the model

# In[62]:

error = MAE('Test{}cb.p'.format(Testing),'CB_RS_SCORES{}_modif.p'.format(Testing),  #Computer Mean Absolute Error
            'CB_RS_ISBNS{}_modif.p'.format(Testing),'Items.p') 


# In[65]:

pickle.dump(error, open('cb_error{}.p'.format(Testing),'wb')) #Save to the HD
error/10


# In[67]:

# Display Cross Validation error of all test sets
CV_error = np.zeros(5)
for i in range(5):
    CV_error[i] = pickle.load(open('cb_error{}.p'.format(i),'rb'))
print('Cross Validation Error on average is {}'.format(np.round(np.mean(CV_error),4)))


# # Online Version

# In[3]:

items= pickle.load(open('items.p','rb'))
Items= pickle.load(open('Itemz.p','rb'))


# In[4]:

user = pd.read_csv('input_cb.txt',encoding='utf8', header = None, index_col=0, names=['ISBN','Rating']) #Read input
user_profile = single_user(user) 
sim = cosine_similarity(user_profile.reshape(1,-1), items).flatten()
Recommend(sim,Items,user)


# In[75]:

print('Time of whole process is {} minutes.'.format(round(time.clock()-t)//60,0))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



