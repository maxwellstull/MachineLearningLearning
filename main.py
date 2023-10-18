from sklearn import linear_model
import pandas as pd

def main():
    data = pd.read_csv(f'flowers.csv', sep=",")
    # Change strings to numbers for classification
    data.replace("Iris-setosa",1,inplace=True)
    data.replace("Iris-versicolor",2,inplace=True)
    data.replace("Iris-virginica",3,inplace=True)
    # Shuffle the data
    data = data.sample(frac=1)

    # Percentage of set to be training data
    DATA_SPLIT_PERC = 50
    cutoff = (int)(len(data)*(DATA_SPLIT_PERC/100))
    # Isolate training data
    x_train = data[['SepalL','SepalW','PetalL','PetalW']][:cutoff]
    y_train = data['Class'][:cutoff]
    # Isolate testing data
    x_test = data[['SepalL','SepalW','PetalL','PetalW']][cutoff+1:]
    y_test = pd.DataFrame(data['Class'][cutoff+1:]) #this gets cast to df since its a series

    # Make and train the model
    regr = linear_model.LogisticRegression(max_iter=1000)
    regr.fit(x_train, y_train)
    # Predict and put that in the result df
    y_test['Predicted'] = regr.predict(x_test)

    correct = 0
    incorrect = 0
    for _index, row in y_test.iterrows():
        if row['Class'] != row['Predicted']:
            print("Error")
            print(row)
            incorrect += 1
        else:
            correct += 1
    print("Correct: {a}/{c}, Incorrect: {b}/{c}".format(a=correct, b=incorrect, c=len(y_test)))

if __name__ == "__main__":
    main()