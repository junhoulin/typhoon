// 當表單被提交時，此函數將被調用
function sendPredictRequest(event) {
    event.preventDefault(); // 防止表單的預設提交行為

    // 從表單中收集數據
    var formData = new FormData(event.target);
    var object = {};
    formData.forEach(function (value, key) {
        object[key] = value;
    });
    var json = JSON.stringify(object);

    // 創建 AJAX 請求
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://192.168.7.167/predict", true);
    xhr.setRequestHeader("Content-Type", "application/json");

    // 定義 resultElement 變數
    var resultElement = document.getElementById("result");

    // 處理響應數據
    xhr.onload = function () {
        if (xhr.status == 200) {
            var response = JSON.parse(xhr.responseText);

            if (response.prediction === "放假") {
                // 如果放假，顯示1.jpg圖片
                resultElement.textContent = "預測結果: 放假";
                resultElement.innerHTML += '<img src="../static/4.jpg" alt="放假">';
            } else {
                // 如果不放假，顯示2.jpg圖片
                resultElement.textContent = "預測結果: 不放假";
                resultElement.innerHTML += '<img src="../static/3.png" alt="不放假">';
            }
        } else {
            resultElement.textContent = "發生錯誤: " + xhr.status;
        }
    };

    // 發送請求
    xhr.send(json);
}
