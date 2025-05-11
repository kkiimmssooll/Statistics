# 통계학 5주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_5th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

5주차는 `2부. 데이터 분석 준비하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.


## Statistics_5th_TIL

### 2부. 데이터 분석 준비하기
### 11.데이터 전처리와 파생변수 생성



## Study Schedule

|주차 | 공부 범위     | 완료 여부 |
|----|----------------|----------|
|1주차| 1부 p.2~56     | ✅      |
|2주차| 1부 p.57~79    | ✅      | 
|3주차| 2부 p.82~120   | ✅      | 
|4주차| 2부 p.121~202  | ✅      | 
|5주차| 2부 p.203~254  | ✅      | 
|6주차| 3부 p.300~356  | 🍽️      | 
|7주차| 3부 p.357~615  | 🍽️      | 

<!-- 여기까진 그대로 둬 주세요-->

# 11.데이터 전처리와 파생변수 생성

```
✅ 학습 목표 :
* 결측값과 이상치를 식별하고 적절한 방법으로 처리할 수 있다.
* 데이터 변환과 가공 기법을 학습하고 활용할 수 있다.
* 모델 성능 향상을 위한 파생 변수를 생성하고 활용할 수 있다.
```


## 11.1. 결측값 처리
```
❓ 내가 평소에 궁금했던 거!
-> 어느 정도의 결측치가 있을 때 무시하거나/처리하는지
```
1. 완전 무작위 결측
순수하게 결측값이 무작위로 발생한 경우
-> 결측값을 포함한 데이터 제거해도 편향이 거의 발생하지 않음
2. 무작위 결측
다른 변수의 특성에 의해 결측치가 체계적으로 발생한 경우
3. 비무작위 결측
결측값들이 해당 변수 자체의 특성을 갖고 있는 경우
> '고객 소득' 변수는 공개를 꺼려해서 결측이 발생할 수 있음

### 표본 제거법
전체 데이터에서 결측값 비율이 10% 미만인 경우우
### 평균 대치법
⚠️ 평균 사용->표준오차가 작아짐->pvalue가 부정확
### 보간법
데이터가 시계열적 특성이 있을 때 효과적
### 회귀 대치법
해당 변수와 다른 변수 사이의 관계성 고려(연령-연 수입)
⚠️ 분산을 과소 추정->확률적 회귀대치법(회귀식에 확률 오차법 추가)
### 다중 대치법
단순대치를 여러 번 수행->n개의 가상 데이터를 생성해 이들의 평균으로 대치
가상 데이터는 5개 내외 정도만 생성해도 문제가 없음
![statweek_1](/git_stat/11.1.png)

```
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df.isnull().sum() # 결측값 수 확인

# 결측값 영역 표시
msno.matrix(df)
plt.show()

# 결측값 막대그래프
msno.bar(df)
plt.show()

# 결측값 표본 제거
df.dropna(how='all') # 모든 컬럼이 결측값인 행 제거
df.dropna(thresh=3) # 세 개 이상 컬럼이 결측값인 행 제거
df.dropna(subset=['temp']) # 특정 컬럼(temp)이 결측값인 행 제거
df.dropna(how='any') # 한 컬럼이라도 결측치가 있는 행 제거

# 결측값 보간 대치
df['temp'].fillna(method='pad', inplace=True) # 전 시점 값으로 대치+컬럼 지정
df.fillna(method='bfill') # 뒤 시점 값으로 대치+전체 컬럼
df.interpolate(method='values') # 단순 순서 보간

df.set_index('dteday') # 시계열 다루니까 인덱스로 설정정
df.interpolate(method='time') # 시점에 따른 보간

# 다중대치
imputer = IterativeImputer(imputation_order='assending',
                        max_iter=10, random_state=42,
                        n_nearest_features=5)
df_imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed) # 다중 대치를 적용하면 넘파이로 변환->다시 판다스 df로 변환
df_imputed.columns = [~~~~]                       
```

## 11.2. 이상치 처리
> 이상치를 결측값으로 대체한 후 결측값 처리 or 이상치 제거가 가장 간단!

> ⚠️ 분산은 감소하지만 편향 발생시킴

> 관측값 변경(하한/상한을 기준으로 대체) / 가중치 조정(이상치 영향을 감소시키는 가중치를 부여) 많이 사용

### 이상치 식별
- 평균으로부터 +-n 표준편차 이상 떨어진 값(n은 보통 3)
- 중위수 절대 편차(MAD)

```
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df['BMI'].describe() # 분포 확인

# 박스 플롯으로 시각화하여 이상치 확인
plt.figure(figsize=(8,6))
sns.boxplot(y='BMI', data=df)
plt.show()

# 이상치 제거
Q1 = df['BMI'].quantile(0.25) # Q1 범위 정의
Q3 = df['BMI'].quantile(0.75) # Q3 범위 정의
IQR = Q3-Q1 # IQR 범위
rev_range = 3 # 제거 범위 조절 변수 설정

# 이상치 범위 설정
filter = (df['BMI'] >= Q1 - rev_range * IQR) & (df['BMI'] <>= Q3 + rev_range * IQR)
df_rmv = df.loc[filter]
```

## 11.3. 변수 구간화
> 이산형 변수->범주형 변수(나이대 등)

- 평활화
    - 변수를 구간화한 후 각 구간의 통계값으로 변환환
- 클러스터링
    - 타깃 변수 필요 없이 값들을 유사한 수준끼리 묶음
- 의사결정나무
    - 타깃 변수를 설정해 값들을 예측에 가장 적합한 구간으로 나눔

```
구간화가 효과적으로 되었는지 측정
1. WOE
2. IV
```
```
from xverse.transformer import WOE

df1.loc[df1['BMI']<=20, 'BMI_bin'] = 'a' # 단순 구간화
pd.cut(df1.BMI, bins=[0,20,30,40,70,95], labels=['a','b','c','d','e]) # cut
pd.cut(df1.BMI, q = 5, labels=['a','b','c','d','e]) # qcut
```
## 11.4. 데이터 표준화와 정규화 스케일링
### 표준화
- 각 관측치의 값이 전체 평균을 기준으로 어느 정도 떨어져 있는지 나타낼 때 사용
- 평균을 0으로 변환

### 정규화
- 데이터의 범위를 0부터 1까지로 변환하여 데이터 분포를 조정
- 전체 데이터 중 해당 값이 어떤 위치에 있는지 파악할 때 유용

### RobustScaler
- 이상치의 영향력을 최소화

![statweek_2](/git_stat/11.4.png)
![statweek_3](/git_stat/11.4.1.png)

```
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

df_stand = StandardScaler.fit_transform(df) # 평균은 0에, 분산은 1에 가까워짐
df_minmax = MinMaxScaler.fit_transform(df) # 모든 데이터 값이 0과 1 사이로 변환
df_robust = RobustScaler.fit_transform(df) # 평균과 분포가 유사하게 변환
```

## 11.5. 모델 성능 향상을 위한 파생 변수 생성
> EDA 및 도메인 이해를 통해 생성
![statweek_4](/git_stat/11.5.png)
![statweek_5](/git_stat/11.5.1.png)
```
⚠️ 기존 변수들을 활용하여 생성하므로 다중공선성 문제가 발생할 수 있음
    -> 파생변수 생성 후 상관분석을 통해 상관성을 확인하자
```

<br>
<br>

# 확인 문제

## 문제 1. 데이터 전처리

> **🧚 한 금융회사의 대출 데이터에서 `소득` 변수에 결측치가 포함되어 있다. 다음 중 가장 적절한 결측치 처리 방법은 무엇인가?**

> **[보기]   
1️⃣ 결측값이 포함된 행을 모두 제거한다.  
2️⃣ 결측값을 `소득` 변수의 평균값으로 대체한다.  
3️⃣ `연령`과 `직업군`을 독립변수로 사용하여 회귀 모델을 만들어 `소득` 값을 예측한다.  
4️⃣ 결측값을 보간법을 이용해 채운다.**

> **[데이터 특징]**     
    - `소득` 변수는 연속형 변수이다.  
    - 소득과 `연령`, `직업군` 간에 강한 상관관계가 있다.  
    - 데이터셋에서 `소득` 변수의 결측 비율은 15%이다.

```
3️⃣
```

## 문제 2. 데이터 스케일링

> **🧚 머신러닝 모델을 학습하는 과정에서, `연봉(단위: 원)`과 `근속연수(단위: 년)`를 동시에 독립변수로 사용해야 합니다. 연봉과 근속연수를 같은 스케일로 맞추기 위해 어떤 스케일링 기법을 적용하는 것이 더 적절한가요?**

<!--표준화와 정규화의 차이점에 대해 고민해보세요.-->

```
표준화
```

### 🎉 수고하셨습니다.