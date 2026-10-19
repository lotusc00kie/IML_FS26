import pandas as pd
from sklearn.metrics import mean_squared_error


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train["mean_x"] = train[[f"x{i}" for i in range(1,11)]].mean(axis=1)
test["y"] = test[[f"x{i}" for i in range (1,11)]].mean(axis=1)


RMSE = mean_squared_error(train["y"], train["mean_x"])**0.5
print(RMSE)

submission = test[["Id", "y"]]
submission.to_csv("submission.csv", index=False)