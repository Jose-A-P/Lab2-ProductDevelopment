import pandas as pd 
import numpy as np
import yaml
import optuna
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

params = yaml.safe_load(open("config.yaml"))
data = pd.read_csv("data/hotel_bookings.csv")

var_numericas = ['lead_time', 'stays_in_weekend_nights', 'adults', 'babies', 'adr'] #Posiblemente escalado
var_nominales = ['hotel', 'market_segment', 'distribution_channel'] #Requiere de etiquetado
var_alta_cardinalidad = ['country'] #Etiquetado
var_ordinales = ['reserved_room_type', 'assigned_room_type'] #Etiquetado pero que ofrezca jerarquia
var_fechas = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']

var_objective = ['is_canceled']

# Juntando las variables de fecha para obtener dia de la semana (Lunes a Domingo) y determinar si es o no fin de semana
data['arrival_date'] = pd.to_datetime(data['arrival_date_year'].astype(str) + '-' + data['arrival_date_month'].astype(str) + '-' + data['arrival_date_day_of_month'].astype(str))
data['day_of_week'] = data['arrival_date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

X = data.drop(var_objective, axis=1)
y = data[var_objective]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Generando las categorias utilizadas por el ordinal encoder
ordinal_room_type = sorted(data['reserved_room_type'].astype(str).dropna().unique().tolist())
ordinal_assigned_room_type = sorted(data['assigned_room_type'].astype(str).dropna().unique().tolist())
ordinal_dia_semana = sorted(data['day_of_week'].astype(str).dropna().unique().tolist(), reverse=True)

# Pipeline numericas
numericas_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#Pipeline nominales, usado tambien en fechas para determinar si es o no fin de semana
nominales_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline alta cardinalidad
alta_cardinalidad_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target', TargetEncoder())
])

def convert_to_str(x):
    return x.astype(str)

#Pipeline ordinales, usado en los dias para determinar el dia de la semana
ordinales_transformer_1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_str', FunctionTransformer(convert_to_str)),
    ('ordinal',OrdinalEncoder(categories=[ordinal_room_type]))
])

ordinales_transformer_2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_str', FunctionTransformer(convert_to_str)),
    ('ordinal',OrdinalEncoder(categories=[ordinal_assigned_room_type]))
])

ordinales_transformer_3 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_str', FunctionTransformer(convert_to_str)),
    ('ordinal',OrdinalEncoder(categories=[ordinal_dia_semana]))
])

# Pipeline final
preprocessor = ColumnTransformer(
    transformers=[
        ('numericas', numericas_transformer, var_numericas),
        ('nominales', nominales_transformer, var_nominales),
        ('weekends', nominales_transformer, ['is_weekend']),
        ('cardinalidad', alta_cardinalidad_transformer, var_alta_cardinalidad),
        ('ordinales_1', ordinales_transformer_1,['reserved_room_type']),
        ('ordinales_2', ordinales_transformer_2,['assigned_room_type']),
        ('ordinales_3', ordinales_transformer_3,['day_of_week'])
    ]
)

def objective(trial):
    
    n_estimators = trial.suggest_int("n_estimators", params['search_space']["n_estimators"][0], params['search_space']["n_estimators"][1])
    max_depth = trial.suggest_int("max_depth", params['search_space']["max_depth"][0], params['search_space']["max_depth"][1])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    # Pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Validación cruzada
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y.values.ravel(), cv=cv, scoring=params['optimizer']['metric'])
    return scores.mean()

study = optuna.create_study(direction=params['optimizer']['direction'])
study.optimize(objective, n_trials=params['optimizer']['n_trials'])

pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=study.best_params['n_estimators'], max_depth=study.best_params['max_depth']))
    ])

pipeline.fit(X_train, y_train.values.ravel())

#Exportando el modelo como joblib para ser utilizado por 
final_model = pipeline
filename = '/output/last_model.joblib'
joblib.dump(final_model, filename)