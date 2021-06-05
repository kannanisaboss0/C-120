#------------------------------------------------------------------------------------LogisticAndGaussianRegression.py------------------------------------------------------------------------------------#
'''
Importing Modules:
-GaussianNB (GNB) :-sklearn.naive_bayes
-LogisticRegression (LogReg) :-sklearn.linear_model
-train_test_split (tts) :-sklearn.model_selection
-StandardScaler (StSc) :-sklearn.preprocessing
-accuracy_score (a_s) :-sklearn.metrics
-datasets (dt) :-sklearn
-pandas (pd)
-random (rd)
-time (tm)
-sys
'''

from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as StSc
from sklearn.metrics import accuracy_score as a_s
from sklearn import datasets as dt
import pandas as pd
import plotly.express as px
import random as rd
import time as tm
import sys


#Defining a function to print the ending message and the error in case of an abrupt error
def PrintRequestTerminationMessage(additional_message_param,syntax_param):
  print("Request Terminated.")

  if(syntax_param!="None"):
    print(syntax_param)

  print(additional_message_param)

  #Printing the ending message
  print("Thank You for using LogisticAndGaussianRegression.py")


#Defining a function to choose specefic fields from the given dataset
def ChooseFieldsFromData(features_arg,number_arg,features_length_arg):

  #Verifying whether the user's input on the number of features is lesser than zero or greater than the total number of featrues in the dataset
  #Case-1
  if(number_arg<0 or number_arg>features_length_arg):
     PrintRequestTerminationMessage("The value should be within a range of 1 and {}".format(features_length),"None")

     #Terminating the program due to the error
     return sys.exit("Invalid Value")

  #Case-2   
  else:  
    factors_param=[]

    for loop in range(1,(number_arg+1)):
      feature_count_param=0

      for feature_param in features_arg:
        feature_count_param+=1
        print("{}:{}".format(feature_count_param,feature_param))

      user_input_param=int(input("Enter number-{}:".format(loop)))

      #Verifying whether the user's input for each feature number is lesser than the total number of features and greater than 0
      #Case-1
      if(user_input<=features_length_arg and user_input>0):
        feature_choice_param=features_arg[user_input_param-1] 
        factors_param.append(feature_choice_param)  

      #Case-2
      else:
        PrintRequestTerminationMessage("The value should be within a range of 1 and {}".format(features_length),"None")

        #Terminating the program due to the error
        return sys.exit("Invalid Value")

    return factors_param 


#Defining a function to model and predict the values from the stipulated data
def ModelAndPredictData(classifier_arg,train_arg,test_arg,result_arg):
  clf_arg_param=classifier_arg

  train_param,test_param=SS.fit_transform(train_arg),SS.fit_transform(test_arg)

  clf_arg_param.fit(factors_train,result_arg)

  prediction=clf_arg_param.predict(test_param)
  return prediction

print("Welcome to LogisticAndGaussianRegression.py. We provide determination on the validity of Logistic and Gaussian Naive Bayes Regression on a dataset,and recommend ether of them. ")
view_information=input("Do not know what Gaussian Naive Bayes Regression is?(:- I Don't Know or I Know)")

#Verifying the user's choice whether they have pre-requisiste knowledge of Gaussian Naive Bayes Regression
#Case-1
if(view_information=="I Don't Know" or view_information=="i don't know" or view_information=="I don't know" or view_information=="I don't Know" or view_information=="I Don't know" or view_information=="I Dont Know" or view_information=="i dont know" or view_information=="I dont know" or view_information=="I dont Know" or view_information=="I Dont know"):
  print("What is Gaussian Naive Bayes Regression?")
  tm.sleep(1.2)
  print("In regression, espically calsifier-based regression, the various factors which contribute to the final value are greatly influenced by each other.")
  tm.sleep(2.3)
  print("In several cases, where the classes of the data are correlated with each other naturally, logistic regression is an adequate method.")
  tm.sleep(2.9)
  print("However, sometimes, classes of a dataset are naturally independent of each other.")
  tm.sleep(2.1)
  print("That is, each value in a class contributes to the resultant value without being affected by any peripheral values of other classes.")
  tm.sleep(2.3)
  print("Logistic regression, in such paramters, would be considered obsolete as it calculates factors with relation to other factors.")
  tm.sleep(2.23)
  print("Hence, in order to resolve this flaw, Binary Naive Regression is used.")
  tm.sleep(1.2)
  print("The most popular type of Naive Bayes Regression is the Gaussian Naive Bayes Regression.")
  tm.sleep(2.3)
  print("An example of Gaussian Naive Regression expressing its superiority in these conditions over Logistic Regression would be:")
  tm.sleep(2.3)
  print("Suppose a dataframe is created. It depicts pass and fail factor of several students in a test and also they hours they studied for the exam and the hours failed to do so.")
  print("Hours Studied    Hours Not Studied   Pass")
  print("    {}                   {}            {}  ".format(rd.randint(5,7),rd.randint(1,2),"1"))
  print("    {}                   {}            {}  ".format(rd.randint(0,1),rd.randint(7,8),"0"))
  print("    {}                   {}            {}  ".format(rd.randint(4,5),rd.randint(4,5),rd.randint(0,1)))
  tm.sleep(3.8)
  print("From the above dataset, it can be concluded that 'Hours Studied' and 'Hours Not Studied' are likely to be linearly regressed.")
  print("Hence, it is applicable for Logistic Regression.")
  tm.sleep(4.5)
  print("However, if 'Hours Not Studied' is replaced with 'Number Of Revison Conducted', the correlation between fields is lost.")
  print("Hours Studied    Number Of Revisions Conducted   Pass")
  print("    {}                   {}                        {}  ".format(rd.randint(5,7),rd.randint(1,2),"1"))
  print("    {}                   {}                        {}  ".format(rd.randint(0,1),rd.randint(7,8),"0"))
  print("    {}                   {}                        {}  ".format(rd.randint(4,5),rd.randint(4,5),rd.randint(0,1)))
  tm.sleep(2.1)
  print("To know more about Naive Bayes Regression, visit 'https://en.wikipedia.org/wiki/Naive_Bayes_classifier' ")

#Case-2
else:
  print("Request Accepted.")
  tm.sleep(0.2)

print("Loading Data...")
tm.sleep(2.3)
  
list_samples=["Unusable_Element","Wine","Iris"]
list_sample_count=0


for list_sample in list_samples[1:]:
  list_sample_count+=1
  print(str(list_sample_count)+":"+list_sample)

loaded_dataset=None
user_input=int(input("Enter the index of dataset desired to analyse:"))


user_input=int(user_input)
#Verifying whether the input provided by the user is either 1 or 2
#Case-1
if (user_input==1 or user_input==2): 

    #Assessing the user's choice on the dataset desired to analyse upon
    #Case-1
    if (user_input==1):

      #Loading dataset-1
      loaded_dataset=dt.load_wine()

    #Case-2  
    elif (user_input==2):

      #Loading dataset-2
      loaded_dataset=dt.load_iris() 
    user_choice=list_samples[user_input]   
#Case-2    
else:
  PrintRequestTerminationMessage("The value of the index should be a whole number or integer","None")
  sys.exit("Invalid Value")
  


#Extracting the keys from the loaded dataset
dataset_keys=loaded_dataset.keys()

print("About the dataset:")
tm.sleep(1.2)
print(loaded_dataset["DESCR"])

df_feature_list=[]
df_target_list=[]

for value in dataset_keys:

  #Verifying whether the the element in the loop is "data" or not
  #Case-1
  if(value=="data"):
    df_feature_list.append(loaded_dataset[value])
    
df_target_list.append(loaded_dataset["target"])

columns_list=loaded_dataset["feature_names"]

df=pd.DataFrame(df_feature_list[0],columns=columns_list)

abstract_input=input("Abstract fields?(:-Yes or No)")

func_factors=None

#Verifying the user's choice on whether they want to abstract the fields or not
#Case-1
if(abstract_input=="Yes" or abstract_input=="yes"):
  features_length=(len(loaded_dataset["feature_names"]))
  number_features_input=int(input("Please enter the number of features(total:{}) desired to predict".format(features_length)))
 
  func_factors=ChooseFieldsFromData(columns_list,number_features_input,features_length)
  
#Case-2  
else:
  print("Request Accepted.")
  func_factors=loaded_dataset["feature_names"]

factors=df[func_factors]
result=loaded_dataset["target"]


random_state_input=int(input("Should the random state vallue be 42 or 0?"))

#Assessing the user's input on the random_state_input value which is to be assgined to the random_state parameter
if(random_state_input>42):
  random_state_input=42
elif(random_state_input<=24):
  random_state_input=0  
factors_train,factors_test,result_train,result_test=tts(factors,result,train_size=0.75,random_state=random_state_input)

lr=LogReg(random_state=0)
gnb=GNB()
SS=StSc()

prediction_log=ModelAndPredictData(lr,factors_train,factors_test,result_train)
logistic_prediction=(a_s(result_test,prediction_log)*100)

print("The acuuracy of the prediction using the Logistic method is {}%".format(round(logistic_prediction,2)))

prediction_gnb=ModelAndPredictData(gnb,factors_train,factors_test,result_train)
gaussian_prediction=(a_s(result_test,prediction_gnb)*100)

print("The acuuracy of the prediction using the Gaussian Naive Bayes method is {}%".format(round(gaussian_prediction,2)))

print("Conclusion:")

tm.sleep(1.3)

#Assessing the accuracy of both the regression types, and to deduce their values comapritively
#Case-1
if(gaussian_prediction>logistic_prediction):
  print("The fields of the dataset are able to contribute indpendently to the resultant value, and hence, are more suitable to Gaussian Naive Bayes Regression or GNB Regression")

#Case-2
elif(logistic_prediction>gaussian_prediction):
  print("The fields of the dataset are unable to contribute indpendently to the resultant value,hence, they are affected by other fields and are more suitable to Logistic Regression or LR Regression")

#Case-3
elif(logistic_prediction==gaussian_prediction):
  print("The fields of the dataset are partially unable and partially able to contribute to the resultant value, and hence, are suitable to both Gaussian Naive Bayes Regression or GNB Regression and Logistic Regression or LR Regression ")    

#Printing the ending message
print("Thank You for using LogisticAndGaussianRegression.py")
#------------------------------------------------------------------------------------LogisticAndGaussianRegression.py------------------------------------------------------------------------------------#  