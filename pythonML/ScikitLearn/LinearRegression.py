import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')

X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수 - 원인)
y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터 (종속 변수 - 결과)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)                   # 학습 (모델 생성)

y_pred = reg.predict(X) # X 에 대한 예축 값

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='green')
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

print('9시간 예상점수', reg.predict([9])) #[9], [8], [7]