# ==========================================
# Aprendizaje automático - Supervisado - Clasificación
# Algoritmo: Regresión logística
# ==========================================

# Importación de librerías
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

print("Librerías importadas correctamente")

# Carga de datos
df = pd.read_csv("telco_Customer_Churn.csv")

print(df.head(5))
print("Cantidad de registros totales: ", len(df))
print(df.shape)
df.info()  # tipo de dato
print(df.isnull().sum())  # trae los valores nulos
print("Mostrar cuántos clientes abandonaron y cuántos permanecieron: ")
print(df["Churn"].value_counts())

# Realizar la separación de nuestras variables
# Limpieza y conversión de TotalCharges (porque tiene espacios vacíos)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Separar las variables
X = df[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "TechSupport",
]]
y = df["Churn"]

# Convertir variables categóricas de X en variables numéricas (dummies)
X = pd.get_dummies(X, drop_first=True)

# Codificar la variable objetivo y (Yes -> 1, No -> 0)
y = y.map({"Yes": 1, "No": 0})

print("Variables en X:", X.shape)
print("Variable objetivo y:", y.shape)

# Dividir el dataset en entrenamiento y prueba (80% / 20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Registro utilizados para el entrenamiento: ", len(X_entrenamiento))
print("Registros utilizados para la prueba: ", len(X_prueba))
print("")
print(
    "Registro utilizados para el entrenamiento - Variable objetivo ",
    len(y_entrenamiento),
)
print(
    "Registros utilizados para la prueba - Variable objetivo ", len(y_prueba)
)

# Creación de nuestro modelo de aprendizaje automatizado
modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print("Modelo creado de forma correcta")

# Entrenar a nuestro modelo
modelo.fit(X_entrenamiento, y_entrenamiento)
print("Modelo entrenado de forma correcta")

# Comprobar como aprendió nuestro modelo - Probabilidades
probabilidad = modelo.predict(X_prueba)
print("Datos de prueba:", X_prueba)
print("Probabilidad de aprendizaje: ", probabilidad)

# Evaluación del modelo - Rendimiento y Métricas
exactitud = accuracy_score(y_prueba, probabilidad)
print(f"Exactitud: {round(exactitud * 100, 2)}%")

# Predicción con un nuevo valor - Externo
nuevo_cliente = pd.DataFrame({
    "tenure": [71],
    "MonthlyCharges": [85.0],
    "TotalCharges": [5500.0],
    "Contract": ["Two year"],
    "InternetService": ["DSL"],
    "TechSupport": ["Yes"],
})

# Procesar y predecir para el Cliente 1
cliente_1_dummies = pd.get_dummies(nuevo_cliente, drop_first=True).reindex(
    columns=X.columns, fill_value=0
)
prediccion_1 = modelo.predict(cliente_1_dummies)[0]

print("--- RESULTADO CLIENTE 1 ---")
print(f"Predicción: {prediccion_1}")
if prediccion_1 == 0:
  print("Resultado: El cliente NO abandonará la compañía.")
else:
  print("Resultado: El cliente abandonará la compañía.")