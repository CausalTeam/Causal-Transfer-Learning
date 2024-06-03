
import numpy as np
import pandas as pd

def generate_data(n, bias_smoking=0, age=50):
    smoking = np.random.binomial(1, 0.3, n)
    exercise = np.random.binomial(1, 0.5, n)
    age = np.random.normal(age, 5, n)  # 平均年龄50，标准差10
    gender = np.random.binomial(1, 0.5, n)  # 0为女性，1为男性
    diet = np.random.binomial(1, 0.4, n)  # 健康饮食为1，否则为0
    cholesterol = np.random.normal(200, 30, n)  # 均值200，标准差30

    blood_pressure = np.random.normal(120 + bias_smoking * smoking + 0.5 * age, 10, n)
    # 计算心脏病发病概率，确保其在0和1之间
    p_heart_disease = 0.25 * (blood_pressure - 120) / 10 + 0.2 * (1 - exercise) + 0.05 * gender + 0.05 * (1 - diet) + \
                      0.02 * (cholesterol - 200) / 30 - 0.3
    #print(p_heart_disease)
    p_heart_disease = np.clip(p_heart_disease, 0, 1)  # 将概率限制在0和1之间
    heart_disease = np.random.binomial(1, p_heart_disease, n)

    return pd.DataFrame({
        'smoking': smoking,
        'exercise': exercise,
        'age': age,
        'gender': gender,
        'diet': diet,
        'cholesterol': cholesterol,
        'blood_pressure': blood_pressure,
        'heart_disease': heart_disease
    })