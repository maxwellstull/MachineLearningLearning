from sklearn import linear_model
import pandas as pd

def main():
    data = pd.read_csv(f'winequality-red.csv', sep=";")
    print(data)





if __name__ == "__main__":
    main()