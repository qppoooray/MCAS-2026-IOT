import numpy as np 
# from sklearn import sklearn # 這行多餘，可以註解掉或刪除
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 載入資料 (這會略過第一行的欄位名稱)
data = np.loadtxt("Bank_Customer.csv", skiprows=1, dtype=str, delimiter=',')

'''
Your work:
1. 非數值型資料需轉換成數值型資料 (country, gender)
2. 去除掉無意義的特徵類別 (customer_id)
3. 標準化特徵 (StandardScaler)
'''
# 讀取 CSV 的第一行(Header)，以便動態找到 customer_id 等欄位的位置
with open("Bank_Customer.csv", "r", encoding="utf-8-sig") as f:
    header = f.readline().strip().split(",")
header_lower = [h.lower() for h in header]

# 1. 轉換非數值型資料為數值 (使用 LabelEncoder)
le = LabelEncoder()
for i in range(data.shape[1]):
    try:
        # 嘗試轉為 float，若失敗則代表該欄位是文字 (如 country, gender)，則進行數值編碼
        data[:, i].astype(float)
    except ValueError:
        data[:, i] = le.fit_transform(data[:, i])

# 2. 去除無意義的特徵類別 (customer_id)
if 'customer_id' in header_lower:
    cid_idx = header_lower.index('customer_id')
    data = np.delete(data, cid_idx, axis=1) # 刪除 customer_id 該欄
else:
    # 如果找不到名稱，防呆預設刪除第 0 欄
    data = np.delete(data, 0, axis=1)

# 將所有資料正式轉換為 float 數值型態，以利後續計算
data = data.astype(float)

# 3. 標準化特徵 (StandardScaler)
# 假設最後一欄是預測目標 y (客戶是否流失)，前面的欄位是特徵 X
X = data[:, :-1]
y = data[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 僅對特徵 X 進行標準化，不標準化標籤 y

# 將標準化後的特徵與目標標籤重新合併為 pre_data
pre_data = np.column_stack((X_scaled, y))

# 儲存處理後的資料
np.savetxt("Bank_Customer_Preprocessed.csv", pre_data, delimiter=",")


from sklearn.linear_model import LogisticRegression 
# 邏輯迴歸分類器
clf = LogisticRegression()

'''
Your work:
切割 data 為 x_train, x_test, y_train, y_test
'''
# 從預處理完畢的 pre_data 取出特徵 (X) 與標籤 (y)
X_processed = pre_data[:, :-1]
y_processed = pre_data[:, -1]

# 切割資料集，此處將 80% 作為訓練集(train)，20% 作為測試集(test)
x_train, x_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)


# 使用訓練集來訓練我們的分類器
clf.fit(x_train, y_train)
# 測試資料集中進行預測
pred = clf.predict(x_test)

#print (pred) # 預測分類，即客戶是否流失
#print (y_test) # 分類
print("模型準確率: " + str(accuracy_score(pred, y_test)*100) + "% ")   # 預測精度