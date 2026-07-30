from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocessing
from src.model_build import model_build




def main():
    # step1: Data Ingestion 
    df = data_ingestion()
    print(df.shape)
    
    # data preprocessing 
    X_train,X_test,y_train,y_test = preprocessing(df)
    print(X_train.shape)
    
    # Model Building 
    model, accuracy = model_build(X_train,X_test,y_train,y_test)
main()


    