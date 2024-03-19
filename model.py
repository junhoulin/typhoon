import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# 定義模型類
class TyphoonModel(nn.Module):
    def __init__(self, input_size):
        super(TyphoonModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# 加載和預處理數據
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, encoding="big5")
    encoder = OneHotEncoder(handle_unknown="ignore")  # 添加 handle_unknown='ignore'
    features = encoder.fit_transform(data.drop("台北是否有放假", axis=1))
    labels = data["台北是否有放假"].values
    return features, labels, encoder


# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 加載模型
def load_model(path, input_size):
    model = TyphoonModel(input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# 進行預測
def predict(model, encoder, input_data):
    input_data_df = pd.DataFrame(
        [input_data],
        columns=["颱風名稱", "侵台路徑", "強度", "近中心最低氣壓", "近中心最大風速", "7級風暴風半徑", "10級風暴風半徑"],
    )
    input_data_encoded = encoder.transform(input_data_df).toarray()
    input_tensor = torch.tensor(input_data_encoded, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).squeeze()
        return prediction.round().item()


# 主函數
def main():
    # 加載和預處理數據
    features, labels, encoder = load_and_preprocess_data("number2.csv")

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # 轉換為 PyTorch 張量
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 創建模型
    model = TyphoonModel(X_train.shape[1])

    # 訓練模型（此處省略訓練過程的詳細代碼）

    # 保存模型
    save_model(model, "typhoon_model.pth")

    # 加載模型
    model = load_model("typhoon_model.pth", X_train.shape[1])


if __name__ == "__main__":
    main()
