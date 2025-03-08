#!/usr/bin/env python
# coding: utf-8

# # Config


# In[1]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
from datetime import datetime

import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix,
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from openpyxl import load_workbook
import itertools





def evaluate_model(y_true, y_pred):
    """
    :param y_true: ground truth values
    :param y_pred: predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(y_true, y_pred)
    print("Classification Report\n", report)

    cm = confusion_matrix(y_true, y_pred)
    
def calculate_age_in_days(row):
    if pd.isna(row["DOB Month"]) or pd.isna(row["DOB Day"]) or pd.isna(row["DOB Year"]):
        return None
    birth_date = datetime(
        year=int(row["DOB Year"]), month=int(row["DOB Month"]), day=int(row["DOB Day"])
    )
    age_in_days = (row["date"] - birth_date).days
    return age_in_days


def calculate_elo(winner_elo, loser_elo, k=32):
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loss = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
    new_winner_elo = winner_elo + k * (1 - expected_win)
    new_loser_elo = loser_elo + k * (0 - expected_loss)
    return new_winner_elo, new_loser_elo

def calculate_elo_v2(winner_elo, loser_elo, method, base_k=32):
    # Define K-factor multipliers based on method
    method_multiplier = {
        'SUB': 1.5,          # Submission
        'M-DEC': 1.2,        # Majority Decision
        'KO/TKO': 1.5,       # Knockout/Technical Knockout
        'U-DEC': 1.1,        # Unanimous Decision
        'S-DEC': 0.8,        # Split Decision
        'Overturned': 0,     # Overturned, no rating change
        'CNC': 0,            # No Contest, no rating change
        'DQ': 0.75           # Disqualification
    }

    # Get the multiplier based on the method of victory
    k = base_k * method_multiplier.get(method, 1)

    # Calculate the expected win/loss probabilities
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    
    # Update Elo ratings
    new_winner_elo = winner_elo + k * (1 - expected_win)
    new_loser_elo = loser_elo - k * expected_win
    return new_winner_elo, new_loser_elo

# In[2]


# Update historical data

historical_file_path = "historical_data 2.csv"

# script_runner()

hist = pd.read_csv(historical_file_path, encoding="latin1")


hist.rename(columns={'date': 'Event Date'}, inplace=True)
hist['Date of Birth'] = hist['Date of Birth'].apply(lambda x: "-".join(x.split("-")[::-1]))
hist['Event Date'] = pd.to_datetime(hist['Event Date'])
hist['Date of Birth'] = pd.to_datetime(hist['Date of Birth'])
data = hist

data['Event Date Month'] = data['Event Date'].dt.month
data['Event Date Day'] = data['Event Date'].dt.day
data['Event Date Year'] = data['Event Date'].dt.year
data['odds'] = data['odds'].apply(lambda x: x if x>0 else round(-10000/x,0))

data['DOB Month'] = data['Date of Birth'].dt.month
data['DOB Day'] = data['Date of Birth'].dt.day
data['DOB Year'] = data['Date of Birth'].dt.year

data.sort_values('Fight ID')
data = data.drop(['Event Date','Date of Birth'], axis=1)

print(data.shape)

data.tail(6)

# In[3]
    
# Create datetime feature to sort by event timeline
data["date"] = (
    data["Event Date Day"].astype(int).astype(str)
    + "-"
    + data["Event Date Month"].astype(int).astype(str)
    + "-"
    + data["Event Date Year"].astype(int).astype(str)
)
data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y")

# Concat full name fighter
data["Winner Full Name"] = (
    data["Winner First Name"].fillna("").str.upper() + " " + data["Winner Last Name"].fillna("").str.upper()
)
data["Fighter Full Name"] = (
    data["Fighter First Name"].fillna("").str.upper() + " " + data["Fighter Last Name"].fillna("").str.upper()
)

data["Age (in days)"] = data.apply(calculate_age_in_days, axis=1)
data["Height"] = data["Height Feet"] * 12 + data["Height Inches"]

def time_converter(a):
    ans = 0
    if np.isnan(a):
        ans = 0 
    elif a<1:
        ans = a*100
    else:
        ans = np.floor(a)*60+((a*100)%100)          

    return ans

data["Ground and Cage Control Time"] = data["Ground and Cage Control Time"].apply(lambda x:time_converter(x))
data["Winning Time"] = data["Winning Time"].apply(lambda x:time_converter(x))

fillna_features = [
    "Winning Time",
    "Ground and Cage Control Time",
    "Knockdown Total",
    "Takedown Total Attempted",
    "Takedown Total Landed",
    "Significant Strike Total Attempted",
    "Significant Strike Total Landed",
    "Significant Strike Head Attempted",
    "Significant Strike Head Landed",
    "Significant Strike Body Attempted",
    "Significant Strike Body Landed",
    "Significant Strike Leg Attempted",
    "Significant Strike Leg Landed",
    "Significant Strike Clinch Attempted",
    "Significant Strike Clinch Landed",
    "Significant Strike Ground Attempted",
    "Significant Strike Ground Landed",
]
data[fillna_features] = data[fillna_features].fillna(0)

data = (
    data.drop_duplicates(subset = ["date", "Winner Full Name", "Fighter Full Name"])
    .sort_values(["date", "Fight ID", "Winner Full Name", "Fighter Full Name"])
    .reset_index(drop=True)
)



# # Feature Engineer
########################################################### Add your feature engineered columns here ###################################################################################################

data["NumberOf_Fight"] = data.groupby(["Fighter Full Name"], as_index=False)[
    "Fight ID"
].transform(lambda x: x.shift(1).rolling(100, min_periods=1).count())
data["IS_WIN"] = np.where(data["Fighter Full Name"] == data["Winner Full Name"], 1, 0)

data["NumberOf_WIN"] = data.groupby(["Fighter Full Name"])["IS_WIN"].transform(
    lambda x: x.shift(1).rolling(window=100, min_periods=1).sum()
)
data["NumberOf_LOSE"] = data["NumberOf_Fight"] - data["NumberOf_WIN"]
data[["NumberOf_Fight", "NumberOf_WIN"]] = data.groupby(["Fighter Full Name"])[
    ["NumberOf_Fight", "NumberOf_WIN"]
].transform(lambda x: x.ffill())
data[["NumberOf_Fight", "NumberOf_WIN", "NumberOf_LOSE"]] = data[
    ["NumberOf_Fight", "NumberOf_WIN", "NumberOf_LOSE"]
].fillna(0)
data["WIN_RATE"] = pd.to_numeric(
    data["NumberOf_WIN"] / data["NumberOf_Fight"], errors="coerce"
).fillna(0)
data["WIN_RATE"] = data["WIN_RATE"].astype(float)
data = data.drop(["IS_WIN"], axis=1)


data["ELO_fighter"] = 1500
data["ELO_winner"] = 1500
for index, row in data.iterrows():
    if (row["Fighter Full Name"] != row["Winner Full Name"]) & (
        row["Winner Full Name"] != "Draw Draw"
    ):
        jump_step = 0
        try: 
            if (
                (data.loc[index, "Winner Full Name"] == data.loc[index + 1, "Fighter Full Name"])
                & (data.loc[index, "Fight ID"] == data.loc[index + 1, "Fight ID"])
            ):
                jump_step = 1
        except: pass
            
        winner_name = row["Winner Full Name"]
        loser_name = row["Fighter Full Name"]
        new_winner_elo, new_loser_elo = calculate_elo(
            data.loc[index, "ELO_winner"], data.loc[index, "ELO_fighter"]
        )
        data.loc[
            (data["Fighter Full Name"] == winner_name) & (data.index > index+jump_step),
            "ELO_fighter",
        ] = new_winner_elo
        data.loc[
            (data["Winner Full Name"] == winner_name) & (data.index > index+jump_step),
            "ELO_winner",
        ] = new_winner_elo

        data.loc[
            (data["Fighter Full Name"] == loser_name) & (data.index > index+jump_step),
            "ELO_fighter",
        ] = new_loser_elo
        data.loc[
            (data["Winner Full Name"] == loser_name) & (data.index > index+jump_step),
            "ELO_winner",
        ] = new_loser_elo



data["new_ELO_fighter"] = 1500
data["new_ELO_winner"] = 1500

for index, row in data[:-1].iterrows():
    if (row["Fighter Full Name"] != row["Winner Full Name"]) & (
        row["Winner Full Name"] != "Draw Draw"
    ):
        jump_step = 0
        if (
            (data.loc[index, "Winner Full Name"] == data.loc[index + 1, "Fighter Full Name"])
            & (data.loc[index, "Fight ID"] == data.loc[index + 1, "Fight ID"])
        ):
            jump_step = 1
            
        winner_name = row["Winner Full Name"]
        loser_name = row["Fighter Full Name"]
        method = row["Winning Method"]  # Assuming you have a 'Win Method' column

        new_winner_elo, new_loser_elo = calculate_elo_v2(
            data.loc[index, "new_ELO_winner"], data.loc[index, "new_ELO_fighter"], method
        )

        # Update Elo ratings for winner and loser
        data.loc[
            (data["Fighter Full Name"] == winner_name) & (data.index > index+jump_step),
            "new_ELO_fighter",
        ] = new_winner_elo
        data.loc[
            (data["Winner Full Name"] == winner_name) & (data.index > index+jump_step),
            "new_ELO_winner",
        ] = new_winner_elo

        data.loc[
            (data["Fighter Full Name"] == loser_name) & (data.index > index+jump_step),
            "new_ELO_fighter",
        ] = new_loser_elo
        data.loc[
            (data["Winner Full Name"] == loser_name) & (data.index > index+jump_step),
            "new_ELO_winner",
        ] = new_loser_elo


# In[4]

data['original_index'] = data.index

# Step 1: Sort the data by Fighter Full Name and date to ensure calculations are done chronologically within each fighter
data = data.sort_values(by=['Fighter Full Name', 'date'])

# Step 2: Create the is_win column (1 if the fighter won, 0 if lost)
data['is_win'] = (data["Fighter Full Name"] == data["Winner Full Name"]).astype(int)

# Step 3: Shift the is_win and Knockdown Total columns to use previous fight's data
data['prev_is_win'] = data.groupby('Fighter Full Name')['is_win'].shift(1)


fighter_list = data['Fighter Full Name'].unique()
ddata = pd.DataFrame()
new_feat_list = []
for k in fighter_list:
    data_fighter = data[data['Fighter Full Name']==k]
    data_fighter_odds = data_fighter['odds']
    for i in fillna_features:
        col_name = i
        prev_col_name = "prev_"+i
        total_win_col_name = "total_win_"+i
        total_lose_col_name = "total_lose_"+i
        win_avg_col_name = "win_avg_"+i
        lose_avg_col_name = "lose_avg_"+i

        data_fighter[prev_col_name] = data_fighter.groupby('Fighter Full Name')[col_name].shift(1)

        # Step 4: Calculate total knockdowns for wins (only for previous winning fights)
        data_fighter[total_win_col_name] = data_fighter.groupby("Fighter Full Name")[prev_col_name].transform(
            lambda x: x.where(data_fighter['prev_is_win'] == 1).cumsum()
        )

        # Step 5: Calculate total knockdowns for losses (only for previous losing fights)
        data_fighter[total_lose_col_name] = data_fighter.groupby("Fighter Full Name")[prev_col_name].transform(
            lambda x: x.where(data_fighter['prev_is_win'] == 0).cumsum()
        )

        # Step 6: Calculate win average knockdown total and apply forward fill for missing values
        data_fighter[win_avg_col_name] = data_fighter.groupby("Fighter Full Name")[total_win_col_name].transform(
            lambda x: x / data_fighter['NumberOf_WIN']
        ).ffill()

        # Step 7: Calculate lose average knockdown total and apply forward fill for missing values
        data_fighter[lose_avg_col_name] = data_fighter.groupby("Fighter Full Name")[total_lose_col_name].transform(
            lambda x: x / data_fighter['NumberOf_LOSE']
        ).ffill()
        data_fighter.drop([prev_col_name,total_win_col_name,total_lose_col_name],axis=1,inplace=True)
        
        if win_avg_col_name in new_feat_list:
            continue
        else:
            new_feat_list.append(win_avg_col_name)
            new_feat_list.append(lose_avg_col_name)


ddata = ddata.fillna(0)
ddata.drop(['is_win','prev_is_win'],axis=1,inplace=True)
ddata = ddata.sort_values(by='original_index').drop(columns=['original_index'])

#####################################################################################################################################################################################################
# In[5]

transform_col = [
    'NumberOf_WIN', 'NumberOf_LOSE',
]


features = []
outcomes = []
ddata = ddata.sort_values(["date", "Fight ID"]).reset_index(drop=True)

# Iterate over the data in steps of 2 rows (one for Fighter A, one for Fighter B)
for i in range(0, len(ddata), 2):
    if i + 1 >= len(ddata):
        break
    row_a = ddata.iloc[i]
    row_b = ddata.iloc[i + 1]

    # Randomly decide who is Fighter A and who is Fighter B to avoid bias
    if np.random.rand() > 0.5:
        row_a, row_b = row_b, row_a

    feature_diff = {
        "fight_id": row_a["Fight ID"],  # Adding Fight ID to feature_diff
        "date": row_a["date"],
        "fighter_a_name": row_a["Fighter Full Name"],
        "fighter_b_name": row_b["Fighter Full Name"],
    }
    features.append(feature_diff)

    if row_a["Winner Full Name"] == row_a["Fighter Full Name"]:
        outcomes.append(1)
    else:
        outcomes.append(0)

    row_a, row_b = row_b, row_a
    feature_diff = {
        "fight_id": row_a["Fight ID"],  # Adding Fight ID to feature_diff
        "date": row_a["date"],
        "fighter_a_name": row_a["Fighter Full Name"],
        "fighter_b_name": row_b["Fighter Full Name"],
    }
    features.append(feature_diff)

    if row_a["Winner Full Name"] == row_a["Fighter Full Name"]:
        outcomes.append(1)
    else:
        outcomes.append(0)

df = pd.concat([pd.DataFrame(features), pd.Series(outcomes)], axis=1)
df.columns = ["fight_id", "date", "fighter_x_name", "fighter_y_name", "fighter_x_win"]
df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)

df





# merge biographical characteristics of fighter and historical dynamic features in each fight for training set
data_bio_feat_list = [
        "Fight ID",
        "date",
        "Fighter Full Name",
        "Age (in days)",
        "Height",
        "ELO_fighter",
        "new_ELO_fighter",
        "NumberOf_Fight",
        "NumberOf_WIN",
        "NumberOf_LOSE",
        "WIN_RATE",
        "Height Feet",
        "Height Inches",
        "Weight Pounds",
        "Reach Inches",
        "Stance",
        "DOB Month",
        "DOB Day",
        "DOB Year",
        ] + new_feat_list 

data_bio = ddata[data_bio_feat_list].drop_duplicates()

df = df.merge(
    data_bio,
    left_on=["fight_id", "date", "fighter_x_name"],
    right_on=["Fight ID", "date", "Fighter Full Name"],
    how="left",
)
df = df.merge(
    data_bio,
    left_on=["fight_id", "date", "fighter_y_name"],
    right_on=["Fight ID", "date", "Fighter Full Name"],
    how="left",
)
print(df.shape)
df.tail()


# In[6]


# exclude "fighter_id", "date" features and "Fighter Full Name" as well to avoid duplicate fighter name with "fighter_x/y_name"
exclude_feature = [
    feature
    for feature in df.columns
    # if "id" in feature.lower() or "fighter full name" in feature.lower()
    if "fighter full name" in feature.lower()
]
print(exclude_feature)
df = df.drop(
    [
        *exclude_feature,
        # "date"
    ],
    axis=1,
)

# astype categorical features for fitting ensemble model
categorical_cols = [feature for feature in df.columns if df[feature].dtype == "O"]
df[categorical_cols] = df[categorical_cols].astype("category")
categorical_cols



fixed_col = [
    "Fighter Full Name",'Height Feet', 'Height Inches',
    'Stance', "date","Fight ID",
]

dynamic_col = data_bio.columns.drop(fixed_col)

for col in dynamic_col:
    df[f"{col}_diff"] = df[f"{col}_x"] - df[f"{col}_y"]
    
dynamic_col


# Correlation calculation with fighter_x_win
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
fighter_x_win_correlation = numeric_df.corrwith(df['fighter_x_win']).sort_values(ascending=False)
print("Correlation with fighter_x_win:")
print(fighter_x_win_correlation)


data_with_leakage = df.copy()

data_without_leakage = df.drop(['fighter_x_win'],axis=1)

file_path = "backtest_prediction_202501_v2.csv"

# In[]

df.to_csv('Cleaned and manipulated UFC data_202501_v2.csv',index=False)

# In[7]

# os.remove(file_path)
model_trained = 0

future_dates = df[df['date'] > '2022-06-04']['date'].unique()

for future_date in future_dates:

    print(future_date)

    train_df_with_leakage = data_with_leakage[data_with_leakage['date']<future_date]

    train_df = train_df_with_leakage.copy()

    new_future = data_without_leakage[data_without_leakage['date']==future_date]
    
    fill_0_cols = ['NumberOf_Fight', 'NumberOf_WIN', 'NumberOf_LOSE', 'WIN_RATE']

    fill_0_cols = list(''.join(e) for e in itertools.product(fill_0_cols, ["_x","_y"]))
    train_df[fill_0_cols] = train_df[fill_0_cols].fillna(0)

    elo_cols = [
            "ELO_fighter",
            "new_ELO_fighter"
    ]
    elo_cols = list(''.join(e) for e in itertools.product(elo_cols, ["_x","_y"]))
    train_df[elo_cols] = train_df[elo_cols].fillna(1500)

    bio_cols = [
            'Age (in days)', 'Height','Height Feet', 'Height Inches', 'Weight Pounds', 'Reach Inches',"Stance"] + new_feat_list 
    

    for col in bio_cols:
            if train_df[f"{col}_x"].dtype == "category":
                    mode_category = train_df[f"{col}_x"].mode()[0]
                    train_df[[f"{col}_x",f"{col}_y"]] = train_df[[f"{col}_x",f"{col}_y"]].fillna(mode_category)
            else:
                    mean_imputation = train_df[[f"{col}_x",f"{col}_y"]].mean().mean()
                    train_df[[f"{col}_x",f"{col}_y"]] = train_df[[f"{col}_x",f"{col}_y"]].fillna(mean_imputation)
                    
    train_df.shape

    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    import lightgbm as lgb


    # model = LGBMClassifier()

    if model_trained ==0:
        n_HP_points_to_test = 100
        param_grid = {
        'num_leaves': [5, 20, 31],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150]
        }
        model = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
        gs = RandomizedSearchCV(
        estimator=model, param_distributions=param_grid, 
        n_iter=n_HP_points_to_test,
        scoring='roc_auc',
        cv=3,
        refit=True,
        random_state=314,
        verbose=True)
        model.fit(train_df.drop(["fight_id","Fight ID_x","Fight ID_y","fighter_x_win", "date",'fighter_x_name','fighter_y_name'], axis=1), train_df["fighter_x_win"])

        evaluate_model(train_df["fighter_x_win"],model.predict(train_df.drop(["fight_id","Fight ID_x","Fight ID_y","fighter_x_win", "date",'fighter_x_name','fighter_y_name'], axis=1)))
        model_trained = 1

    y_pred = model.predict(new_future.drop(["fight_id","Fight ID_x","Fight ID_y","date",'fighter_x_name','fighter_y_name'],axis=1))
    y_proba_all = model.predict_proba(new_future.drop(["fight_id","Fight ID_x","Fight ID_y","date",'fighter_x_name','fighter_y_name'],axis=1))[:, 1]

    new_future["x_win"]  = y_pred
    new_future["probability"] = y_proba_all
    

    # Check if the file already exists


    if os.path.exists(file_path):
        # Load the existing CSV into a DataFrame
        existing_df = pd.read_csv(file_path)
        # Append the new data (future_test DataFrame)       
        updated_df = pd.concat([existing_df, new_future], ignore_index=True)
        # Write the updated DataFrame back to the CSV file
        updated_df.to_csv(file_path, index=False)
    else:
        # If the file doesn't exist, write the future_test DataFrame to a new CSV file
        new_future.to_csv(file_path, index=False)

    # new_df = pd.concat([new_df, future_test])

print("File Saved Successfully")
# %%
