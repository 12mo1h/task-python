#mohamed ashrtaf salah
# final task
# مكتبات
import numpy as np              # مكتبة العمليات الحسابية
import pandas as pd             # مكتبة معالجة البيانات
import matplotlib.pyplot as plt # مكتبة الرسم البياني

# تحميل البيانات
# تأكد أن ملف student_score.csv موجود في نفس المجلد
data = pd.read_csv("student_score.csv")

# عرض أول 5 صفوف من البيانات
print(data.head())

# فحص القيم المفقودة (Preprocessing)
print("\nMissing Values:")
print(data.isnull().sum())

# في حال وجود قيم ناقصة، نقوم بتعويضها بالمتوسط
data.fillna(data.mean(), inplace=True)

#  تصور البيانات (Visualization)
plt.scatter(data["Hours"], data["Scores"], color="blue")
plt.title("Study Hours vs Student Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Student Score")
plt.grid(True)
plt.show()

#  تحديد المتغيرات
#DataFrame
X = data[["Hours"]]   # المتغير المستقل (المدخل)
y = data["Scores"]    # المتغير التابع (المخرج)

# تقسيم البيانات إلى تدريب واختبار
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% اختبار
    random_state=42     # لضمان نفس النتائج كل مرة
)

#  إنشاء وتدريب نموذج Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # إنشاء النموذج
model.fit(X_train, y_train) # تدريب النموذج

#  التنبؤ باستخدام بيانات الاختبار
y_pred = model.predict(X_test)

#  تقييم النموذج
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)  # حساب MSE
r2 = r2_score(y_test, y_pred)              # حساب R²

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)#كل ما كان أقل → النموذج أفضل
print("R-Squared (R²):", r2)#كل ما كان أقرب إلى 1 → النموذج أفضل

#  رسم خط الانحدار الخطي
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Linear Regression - Student Score Prediction")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
