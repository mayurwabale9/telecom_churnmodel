from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def preprocessing(df):

    #Step 1: Dropping Duplicated Columns
    df.drop_duplicates()

    #Step 2: Imputing Missing Values
    '''
    No Missing Values present in Dataset
    '''

    #Step 3: Data Cleaning : Drop unwanted Columns from the Dataset

    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

    # Checking Descriptive Statistics

    numerical_data = df.select_dtypes(exclude = 'object')
    categorical_data = df.select_dtypes(include = 'object')

    # Using Encoding Technique 

    le = LabelEncoder()
    for i in categorical_data.columns:
        df[i] = le.fit_transform(df[i])   

    # Droping Unwantesd Columns 
    df.drop(columns = 'customerID',inplace = True)


    # Split the dataset into train and test : Seen and Unseen Data 

    # Split the dataset into input fetures and target column i.e. X and y

    X = df.drop(columns = 'Churn')
    y = df['Churn']
    
    X_train,X_test,y_train,y_test = train_test_split( X,y,
                                                    test_size = 0.3,
                                                    random_state = 1)

    
    sm = SMOTE()

    X_train,y_train = sm.fit_resample(X_train,y_train)

    return X_train,X_test,y_train,y_test
