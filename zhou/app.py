import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

app = Flask(__name__)

# 路由：主页
@app.route('/')
def index():
    return render_template('index.html')

# 路由：上传文件并训练模型
@app.route('/train_model', methods=['POST'])
def train_model():
    # 获取上传的文件
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # 读取CSV文件
    data_train = pd.read_csv(file)
    
    # 选择需要的变量
    simvars = ['anti_after', 'creatinine_0','plt_1', 'creatinine_1', 'charlson', 'wbc_1', 'imv_after','hb_1', 'pct_0', 'bilirubin_total_1']
    x_train = data_train[simvars]
    y_train = data_train['death_28day']
    
    # 创建DMatrix和XGBoost模型
    dtrain = xgb.DMatrix(x_train, label=y_train)
    model = xgb.XGBClassifier()
    
    # 网格搜索调参
    bst = GridSearchCV(model, {
        'max_depth': [5, 7],
        'n_estimators': [30, 40],
        'learning_rate': [0.01, 0.1],
        'gamma': [0.001, 0.01],
        'min_child_weight': [5, 7, 9]
    }, verbose=1, cv=2, scoring='roc_auc', n_jobs=-1, refit=True)
    
    bst.fit(x_train, y_train)
    params = bst.best_params_
    
    # 保存模型
    bst.best_estimator_.save_model("model.json")
    
    # 训练模型并生成ROC曲线
    cv_res = xgb.cv(params, dtrain, num_boost_round=100, early_stopping_rounds=10, metrics='auc', show_stdv=True)
    bst = xgb.train(params, dtrain, num_boost_round=cv_res.shape[0])
    ypred_train = bst.predict(dtrain)
    
    fpr, tpr, threshold = roc_curve(y_train, ypred_train)
    auc_score = auc(fpr, tpr)

    # 绘制ROC曲线并保存为PDF
    plt.plot(fpr, tpr, color="#0984e3", lw=2, label='XGBoost in train dataset (AUROC = %0.2f)' % auc_score)
    plt.grid(0)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("static/roc_curve.pdf")
    plt.close()

    return render_template('index.html', auc_score=auc_score, roc_image='static/roc_curve.pdf')

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

@app.route('/predict', methods=['POST'])
def predict():
    # 加载训练好的模型
    bst = xgb.Booster()
    bst.load_model("model.json")
    
    # 获取输入的患者数据
    patient_data = [float(request.form[variable]) for variable in [
        'anti_after', 'creatinine_0','plt_1', 'creatinine_1', 'charlson', 
        'wbc_1', 'imv_after','hb_1', 'pct_0', 'bilirubin_total_1']]

    patient_df = pd.DataFrame([patient_data], columns=['anti_after', 'creatinine_0','plt_1', 'creatinine_1', 'charlson', 
        'wbc_1', 'imv_after','hb_1', 'pct_0', 'bilirubin_total_1'])

    # 使用训练好的XGBoost模型进行预测
    dpatient = xgb.DMatrix(patient_df)
    death_prob = bst.predict(dpatient)[0]

    # 将预测结果转换为普通float类型
    death_prob = float(death_prob)

    # 返回JSON响应，包含死亡概率
    return jsonify({
        'death_probability': death_prob
    })

if __name__ == '__main__':
    app.run(debug=True)
