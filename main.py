import numpy as np
from sklearn.neural_network import MLPClassifier
import statsmodels.api as sm
import data_generator as dg
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


data_source = dg.generate_data(10000, bias_smoking=35,age=50)
data_target = dg.generate_data(5, bias_smoking=5,age=30)
data_target2 = dg.generate_data(1000, bias_smoking=5,age=30)

def train_model(data):
    model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
    X = data[['smoking', 'exercise', 'age', 'gender', 'diet', 'cholesterol', 'blood_pressure']]
    y = data['heart_disease']
    model.fit(X, y)
    return model

def evaluate_model(model, data):
    X = data[['smoking', 'exercise', 'age', 'gender', 'diet', 'cholesterol', 'blood_pressure']]
    y = data['heart_disease']
    score = model.score(X, y)
    return score

model_source = train_model(data_source)


print("调整前源域模型在目标域的表现：", evaluate_model(model_source, data_target2))


# 估计源域中吸烟对血压的影响
X_source = sm.add_constant(data_source['smoking'])
bp_model_source = sm.OLS(data_source['blood_pressure'], X_source).fit()
smoking_effect_on_bp_source = bp_model_source.params['smoking']

# 估计源域中血压对心脏病的影响
X_source = sm.add_constant(data_source['blood_pressure'])
heart_disease_model_source = sm.OLS(data_source['heart_disease'], X_source).fit(disp=False)#Logit
bp_effect_on_heart_disease_source = heart_disease_model_source.params['blood_pressure']

# 估计目标域中吸烟对血压的影响
X_target = sm.add_constant(data_target['smoking'])
bp_model_target = sm.OLS(data_target['blood_pressure'], X_target).fit()
smoking_effect_on_bp_target = bp_model_target.params['smoking']

# 调整目标域模型
def adjust_model(data, effect_difference, effect2):
    model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
    X = data[['smoking', 'exercise', 'age', 'gender', 'diet', 'cholesterol', 'blood_pressure']]
    X['blood_pressure'] = data['blood_pressure'] + effect_difference * data['smoking'] - 10
    X['age'] = data['age'] - 20

    y = data['heart_disease']

    p_heart_disease = 0.25 * (X['blood_pressure'] - 120) / 10 + 0.2 * (1 - X['exercise']) + 0.05 * X['gender'] + 0.05 * (1 - X['diet']) + \
                      0.02 * (X['cholesterol'] - 200) / 30 - 0.3

    p_heart_disease = np.clip(p_heart_disease, 0, 1)  # 将概率限制在0和1之间
    y = np.random.binomial(1, p_heart_disease, len(y))

    model.fit(X, y)
    return model

effect_difference = smoking_effect_on_bp_target - smoking_effect_on_bp_source
# print(effect_difference)
# print(bp_effect_on_heart_disease_source)

model_adjusted = adjust_model(data_source, effect_difference, bp_effect_on_heart_disease_source)

print("源域数据调整后模型在目标域的表现：", evaluate_model(model_adjusted, data_target2))

model3=train_model(data_target)

print("目标域样本训练模型在目标域的表现：", evaluate_model(model3, data_target2))

