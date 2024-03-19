# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from model import (
    TyphoonModel,
    predict,
    load_and_preprocess_data,
)  # 假設您的模型相關代碼在 model.py 文件中

app = Flask(__name__)
CORS(app)  # 啟用 CORS

# 加載模型和編碼器
features, labels, encoder = load_and_preprocess_data("number2.csv")
model = TyphoonModel(features.shape[1])
model.load_state_dict(torch.load("typhoon_model.pth"))
model.eval()


@app.route("/")
def index():
    return render_template("index.html")  # 返回 index.html


@app.route("/predict", methods=["POST"])
def predict_api():
    # 从 JSON 中提取数据
    input_data = request.json
    # 转换为适合模型的格式
    input_data_processed = [
        input_data.get("颱風名稱", ""),
        int(input_data.get("侵台路徑", 0)),
        int(input_data.get("強度", 0)),
        int(input_data.get("近中心最低氣壓", 0)),
        int(input_data.get("近中心最大風速", 0)),
        int(input_data.get("7級風暴風半徑", 0)),
        int(input_data.get("10級風暴風半徑", 0)),
    ]
    prediction = predict(model, encoder, input_data_processed)
    return jsonify({"prediction": "放假" if prediction == 1 else "不放假"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
