{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "38ddd7cf-23a7-44dc-9b95-d5b3507097a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing essential libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import pickle\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.manifold import TSNE\n",
    "from sklearn.cluster import KMeans\n",
    "from matplotlib import pyplot as plt\n",
    "from sklearn.metrics import roc_auc_score, roc_curve, auc\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4d39cbd8-707f-47f8-a8d3-b90dd933b102",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the dataset\n",
    "df = pd.read_csv('diabetes.csv')\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == 'object':\n",
    "        df[col] = pd.to_numeric(df[col], errors='coerce')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3b4a5743-7b4a-47f5-a894-f85cbfc5dc9e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>148</td>\n",
       "      <td>72</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>33.6</td>\n",
       "      <td>0.627</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0            6      148             72             35        0  33.6   \n",
       "1            1       85             66             29        0  26.6   \n",
       "2            8      183             64              0        0  23.3   \n",
       "3            1       89             66             23       94  28.1   \n",
       "4            0      137             40             35      168  43.1   \n",
       "\n",
       "   DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                     0.627   50        1  \n",
       "1                     0.351   31        0  \n",
       "2                     0.672   32        1  \n",
       "3                     0.167   21        0  \n",
       "4                     2.288   33        1  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9d13a55f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 768 entries, 0 to 767\n",
      "Data columns (total 9 columns):\n",
      " #   Column                    Non-Null Count  Dtype  \n",
      "---  ------                    --------------  -----  \n",
      " 0   Pregnancies               768 non-null    int64  \n",
      " 1   Glucose                   768 non-null    int64  \n",
      " 2   BloodPressure             768 non-null    int64  \n",
      " 3   SkinThickness             768 non-null    int64  \n",
      " 4   Insulin                   768 non-null    int64  \n",
      " 5   BMI                       768 non-null    float64\n",
      " 6   DiabetesPedigreeFunction  768 non-null    float64\n",
      " 7   Age                       768 non-null    int64  \n",
      " 8   Outcome                   768 non-null    int64  \n",
      "dtypes: float64(2), int64(7)\n",
      "memory usage: 54.1 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "adf733d6-da72-4fa1-9bce-5381fea4ccab",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Renaming DiabetesPedigreeFunction as DPF\n",
    "df = df.rename(columns={'DiabetesPedigreeFunction':'familyhistory'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7b52d242",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total number of rows : 768\n",
      "number of rows missing Glucose: 5\n",
      "number of rows missing BloodPressure: 35\n",
      "number of rows missing SkinThickness: 227\n",
      "number of rows missing Insulin: 374\n",
      "number of rows missing BMI: 11\n",
      "number of rows missing familyhistory: 0\n",
      "number of rows missing Age: 0\n"
     ]
    }
   ],
   "source": [
    "print(\"total number of rows : {0}\".format(len(df)))\n",
    "print(\"number of rows missing Glucose: {0}\".format(len(df.loc[df['Glucose'] == 0])))\n",
    "print(\"number of rows missing BloodPressure: {0}\".format(len(df.loc[df['BloodPressure'] == 0])))\n",
    "print(\"number of rows missing SkinThickness: {0}\".format(len(df.loc[df['SkinThickness'] == 0])))\n",
    "print(\"number of rows missing Insulin: {0}\".format(len(df.loc[df['Insulin'] == 0])))\n",
    "print(\"number of rows missing BMI: {0}\".format(len(df.loc[df['BMI'] == 0])))\n",
    "print(\"number of rows missing familyhistory: {0}\".format(len(df.loc[df['familyhistory'] == 0])))\n",
    "print(\"number of rows missing Age: {0}\".format(len(df.loc[df['Age'] == 0])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8cc95612-5b52-4998-b3ee-f521cb77f504",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_copy = df.copy(deep=True)\n",
    "df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a78668fb-af9c-44be-a55d-49c78c54965a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>familyhistory</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.845052</td>\n",
       "      <td>121.686763</td>\n",
       "      <td>72.405184</td>\n",
       "      <td>29.108073</td>\n",
       "      <td>140.671875</td>\n",
       "      <td>32.455208</td>\n",
       "      <td>0.471876</td>\n",
       "      <td>33.240885</td>\n",
       "      <td>0.348958</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.369578</td>\n",
       "      <td>30.435949</td>\n",
       "      <td>12.096346</td>\n",
       "      <td>8.791221</td>\n",
       "      <td>86.383060</td>\n",
       "      <td>6.875177</td>\n",
       "      <td>0.331329</td>\n",
       "      <td>11.760232</td>\n",
       "      <td>0.476951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>18.200000</td>\n",
       "      <td>0.078000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>99.750000</td>\n",
       "      <td>64.000000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>121.500000</td>\n",
       "      <td>27.500000</td>\n",
       "      <td>0.243750</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>117.000000</td>\n",
       "      <td>72.202592</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>125.000000</td>\n",
       "      <td>32.300000</td>\n",
       "      <td>0.372500</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.000000</td>\n",
       "      <td>140.250000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>127.250000</td>\n",
       "      <td>36.600000</td>\n",
       "      <td>0.626250</td>\n",
       "      <td>41.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>199.000000</td>\n",
       "      <td>122.000000</td>\n",
       "      <td>99.000000</td>\n",
       "      <td>846.000000</td>\n",
       "      <td>67.100000</td>\n",
       "      <td>2.420000</td>\n",
       "      <td>81.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \\\n",
       "count   768.000000  768.000000     768.000000     768.000000  768.000000   \n",
       "mean      3.845052  121.686763      72.405184      29.108073  140.671875   \n",
       "std       3.369578   30.435949      12.096346       8.791221   86.383060   \n",
       "min       0.000000   44.000000      24.000000       7.000000   14.000000   \n",
       "25%       1.000000   99.750000      64.000000      25.000000  121.500000   \n",
       "50%       3.000000  117.000000      72.202592      29.000000  125.000000   \n",
       "75%       6.000000  140.250000      80.000000      32.000000  127.250000   \n",
       "max      17.000000  199.000000     122.000000      99.000000  846.000000   \n",
       "\n",
       "              BMI  familyhistory         Age     Outcome  \n",
       "count  768.000000     768.000000  768.000000  768.000000  \n",
       "mean    32.455208       0.471876   33.240885    0.348958  \n",
       "std      6.875177       0.331329   11.760232    0.476951  \n",
       "min     18.200000       0.078000   21.000000    0.000000  \n",
       "25%     27.500000       0.243750   24.000000    0.000000  \n",
       "50%     32.300000       0.372500   29.000000    0.000000  \n",
       "75%     36.600000       0.626250   41.000000    1.000000  \n",
       "max     67.100000       2.420000   81.000000    1.000000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Replacing NaN value by mean, median depending upon distribution\n",
    "df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)\n",
    "df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)\n",
    "df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)\n",
    "df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)\n",
    "df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)\n",
    "df_copy.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cc834acd-2e34-4e95-9fd0-6714dd46525e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model Building\n",
    "from sklearn.model_selection import train_test_split\n",
    "X = df_copy.drop(columns='Outcome')\n",
    "y = df_copy['Outcome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9431ea95",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)\n",
    "scaler = StandardScaler().fit(X_train)\n",
    "X_train_scaled = scaler.transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0413e575-9952-4900-bcb1-99cfc99e3d78",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating Random Forest Model\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "3efb71ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Training Score: 1.0\n",
      "Random Forest Testing Score: 0.8020833333333334\n"
     ]
    }
   ],
   "source": [
    "print(f'Random Forest Training Score: {clf.score(X_train_scaled, y_train)}')\n",
    "print(f'Random Forest Testing Score: {clf.score(X_test_scaled, y_test)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "67ff03b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    low_risk       0.83      0.87      0.85       123\n",
      "   high_risk       0.75      0.68      0.71        69\n",
      "\n",
      "    accuracy                           0.80       192\n",
      "   macro avg       0.79      0.78      0.78       192\n",
      "weighted avg       0.80      0.80      0.80       192\n",
      "\n",
      "Training Score: 1.0\n",
      "Testing Score: 0.8020833333333334\n"
     ]
    }
   ],
   "source": [
    "# Fit a model, and then print a classification report\n",
    "y_pred = clf.predict(X_test_scaled)\n",
    "print(classification_report(y_test, y_pred, target_names=[\"low_risk\",\"high_risk\"]))\n",
    "print(f'Training Score: {clf.score(X_train_scaled, y_train)}')\n",
    "print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "61de401f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "feature_importances = clf.feature_importances_\n",
    "features = sorted(zip(X_test.columns, clf.feature_importances_), key = lambda x: x[1])\n",
    "cols = [f[0] for f in features]\n",
    "width = [f[1] for f in features]\n",
    "\n",
    "fig, ax = plt.subplots()\n",
    "\n",
    "plt.margins(y=0.001)\n",
    "\n",
    "ax.barh(y=cols, width=width, height=0.5)\n",
    "plt.savefig(\"img1.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ddcac836",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## Correlation\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "#get correlations of each features in dataset\n",
    "corrmat = df_copy.corr()\n",
    "top_corr_features = corrmat.index\n",
    "plt.figure(figsize=(10,10))\n",
    "mask = np.triu(np.ones_like(corrmat, dtype=bool))\n",
    "#plot heat map\n",
    "g=sns.heatmap(df_copy[top_corr_features].corr(),annot=True,cmap=\"RdYlGn\", mask=mask)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "69618e3d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[<AxesSubplot:title={'center':'Pregnancies'}>,\n",
       "        <AxesSubplot:title={'center':'Glucose'}>,\n",
       "        <AxesSubplot:title={'center':'BloodPressure'}>],\n",
       "       [<AxesSubplot:title={'center':'SkinThickness'}>,\n",
       "        <AxesSubplot:title={'center':'Insulin'}>,\n",
       "        <AxesSubplot:title={'center':'BMI'}>],\n",
       "       [<AxesSubplot:title={'center':'familyhistory'}>,\n",
       "        <AxesSubplot:title={'center':'Age'}>,\n",
       "        <AxesSubplot:title={'center':'Outcome'}>]], dtype=object)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 9 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_copy.hist(figsize=(10,8))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "df569db6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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gV6vl4mJ4a7Taaedll6v13IiAID1UwgBADQwNTWVCy64IElywQUXZNeuXS0nAgBYP4UQAAAAQMcohAAAGlhaWsrVV1+dJLn66quzvLzcciIAgPVTCAEANNDv91NrTZKsrKyk3++3nAgAYP0UQgAADSwsLGQwGCRJBoNB5ufnW04EALB+CiEAgAZcdh4A2E4UQgAADfR6vaysrCQZHjLmsvMAwDhTCAEAAAB0jEIIAKCBfr+fiYnhrtPExISTSgMAY00hBADQwMLCQo4cOZIkOXLkiJNKAwBjTSEEANDA9PR0SilJklKKk0oDAGNNIQQA0MCFF16YWmuSpNaaiy66qOVEAADrpxACAGjgbW9726oZQpdffnnLiQAA1k8hBADQwMLCwqoZQs4hBACMM4UQAEAD09PTmZycTJJMTk46hxAAMNYUQgAADfR6vftmCB0dAwCMK4UQAEADU1NTOf3005MkD33oQ7Nr166WEwEArJ9CCACggY985CO58847kyR33nlnDh061HIiAID1UwgBADRw2WWXrRpfeumlLSUBANg4hRAAQAOLi4trjgEAxolCCACggZ07d645BgAYJwohAIAGjhw5suYYAGCcKIQAABp4znOes2r83Oc+t6UkAAAbpxACAGig1+utOQYAGCcKIQCABj7zmc+sGn/2s59tKQkAwMYphAAAGvj5n//5VeN//+//fUtJAAA2TiEEANDAJz7xiTXHAADjRCEEAAAA0DEKIQCABp761KeuGj/taU9rKQkAwMYphAAAGjjrrLPWHAMAjBOFEABAA9dee+2q8Tvf+c6WkgAAbJxCCACggV27dq05BgAYJwohAIAGbr755jXHAADjRCEEAAAA0DEKIQCABs4555w1xwAA40QhBADQwC233LLmGABgnCiEAAAaWFlZWXMMADBOFEIAAA2UUtYcAwCME4UQAEADtdY1xwAA40QhBADQwMTExJpjAIBxYk8GAKCB6enpVeOZmZmWkgAAbJxCCACggZe85CVrjgEAxolCCACggc985jOrxp/97GdbSgIAsHEKIQCABi677LJV40svvbSlJAAAG6cQAgBoYHFxcc0xAMA4UQgBADSwZ8+eNccAAONEIQQA0MArX/nKVeNXvepVLSUBANg4hRAAQAPnnntudu7cmSTZuXNn9u7d23IiAID1UwgBADSwtLSUe+65J0ny+c9/PsvLyy0nAgBYP4UQAEAD/X4/KysrSZJ77703/X6/5UQAAOunEAIAaGB+fj611iRJrTVXXnlly4kAANZPIQQA0MA555yz5hgAYJwohAAAGrj11lvXHAMAjBOFEABAAzMzM6vGz3nOc1pKAgCwcQohAIAGer1eSilJklJKer1ey4kAANZPIQQAAADQMQohAIAGXv/616+6ytjrX//6lhMBAKyfQggAoIGrrrpq1fgv/uIvWkoCALBxCiEAgAaOzg460RgAYJwohAAAGnjGM56xavzMZz6zpSQAABunEAIAaOChD33ommMAgHGiEAIAaOBd73rXqvG1117bUhIAgI1TCAEANDA9Pb1qPDMz01ISAICNUwgBADRw/3MIPetZz2opCQDAximEAAAa+PVf//VV41/7tV9rKQkAwMYphAAAGlhcXFxzDAAwThRCAAANPO5xj1tzDAAwThRCAAANfM3XfM2q8d69e1tKAgCwcQohAIAG/uZv/mbV+K//+q9bSgIAsHEKIQCABs4555w1xwAA40QhBADQwK233rrmGABgnCiEAAAa+NZv/dZV42/7tm9rKQkAwMYphAAAGrjxxhtXjT/84Q+3lAQAYOMUQgAADThkDADYThRCAAAAAB2jEAIAaOCMM85YcwwAME4UQgAADdRa1xwDAIwThRAAQAP3v6rYU5/61JaSAABsnEIIAKCBf/iHf1g1PnToUEtJAAA2TiEEANDAJz7xiTXHAADjRCEEANDAnj171hwDAIwThRAAQAMvfOELV41f9KIXtZQEAGDjFEIAAA288Y1vXDX+3d/93ZaSAABsnEIIAKCBxcXFNccAAONEIQQA0IBzCAEA24lCCACgge/93u9dNf6+7/u+lpIAAGycQggAoIE3vOENq8a/+Zu/2VISAICNUwgBADRw5513rjkGABgnCiEAgAZ27ty55hgAYJwohAAAGnj1q1+9avya17ymnSAAAJtAIQQA0MAnP/nJNccAAONEIQQA0MCv/uqvrhr/8i//cjtBAAA2gUIIAKCBWuuaYwCAcaIQAgAAAOgYhRAAAABAxyiEAAAa2LFjx5pjAIBxohACAGjgyJEja44BAMbJAxZCpZTHlVKuLqXcWEr5UCnlJ0e3P6KUslBKuWn09exjfudnSymHSil/X0p5zlb+AwAAAAB4cJrMEDqS5P+ptX59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\n",
      "text/plain": [
       "<Figure size 1440x1080 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize=(20,15))\n",
    "sns.boxplot(data=df_copy);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8ba9bb8c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.63994726,  0.86510807, -0.03351824, ...,  0.46849198,\n",
       "         1.4259954 ,  1.36589591],\n",
       "       [-0.84488505, -1.20616153, -0.52985903, ..., -0.36506078,\n",
       "        -0.19067191, -0.73212021],\n",
       "       [ 1.23388019,  2.0158134 , -0.69530596, ...,  0.60439732,\n",
       "        -0.10558415,  1.36589591],\n",
       "       ...,\n",
       "       [ 0.3429808 , -0.0225789 , -0.03351824, ..., -0.68519336,\n",
       "        -0.27575966, -0.73212021],\n",
       "       [-0.84488505,  0.14180757, -1.02619983, ..., -0.37110101,\n",
       "         1.17073215,  1.36589591],\n",
       "       [-0.84488505, -0.94314317, -0.19896517, ..., -0.47378505,\n",
       "        -0.87137393, -0.73212021]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_scaled = StandardScaler().fit_transform(df_copy)\n",
    "df_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "dc1d1412",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(768, 9)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_scaled.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2bfcedb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "pca = PCA(n_components=0.90)\n",
    "scaled_pca = pca.fit_transform(df_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "8c45c4b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.898069</td>\n",
       "      <td>-0.649162</td>\n",
       "      <td>0.319979</td>\n",
       "      <td>0.636566</td>\n",
       "      <td>0.529739</td>\n",
       "      <td>-0.626567</td>\n",
       "      <td>0.667715</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1.704944</td>\n",
       "      <td>0.095397</td>\n",
       "      <td>-0.227434</td>\n",
       "      <td>-0.093522</td>\n",
       "      <td>0.188265</td>\n",
       "      <td>0.428828</td>\n",
       "      <td>0.356478</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.059376</td>\n",
       "      <td>-0.669268</td>\n",
       "      <td>1.941193</td>\n",
       "      <td>0.472333</td>\n",
       "      <td>0.610127</td>\n",
       "      <td>-1.205594</td>\n",
       "      <td>0.640480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-2.281603</td>\n",
       "      <td>0.216062</td>\n",
       "      <td>-0.272205</td>\n",
       "      <td>-0.568943</td>\n",
       "      <td>-0.035082</td>\n",
       "      <td>-0.253182</td>\n",
       "      <td>-0.185080</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.396639</td>\n",
       "      <td>3.624300</td>\n",
       "      <td>2.018026</td>\n",
       "      <td>4.710945</td>\n",
       "      <td>0.971517</td>\n",
       "      <td>0.121820</td>\n",
       "      <td>0.388562</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          0         1         2         3         4         5         6\n",
       "0  1.898069 -0.649162  0.319979  0.636566  0.529739 -0.626567  0.667715\n",
       "1 -1.704944  0.095397 -0.227434 -0.093522  0.188265  0.428828  0.356478\n",
       "2  1.059376 -0.669268  1.941193  0.472333  0.610127 -1.205594  0.640480\n",
       "3 -2.281603  0.216062 -0.272205 -0.568943 -0.035082 -0.253182 -0.185080\n",
       "4  1.396639  3.624300  2.018026  4.710945  0.971517  0.121820  0.388562"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_pca = pd.DataFrame(data = scaled_pca)\n",
    "df_pca.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "67d59c4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.29101077, 0.16644605, 0.13033872, 0.10204294, 0.08548146,\n",
       "       0.08331198, 0.05555137])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pca.explained_variance_ratio_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "23267999",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(768, 7)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaled_pca.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0aff48f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "pca = PCA(n_components=3)\n",
    "\n",
    "X_scaled_pca = pca.fit_transform(scaled_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1da7e831",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pc1</th>\n",
       "      <th>pc2</th>\n",
       "      <th>pc3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.898069</td>\n",
       "      <td>-0.649162</td>\n",
       "      <td>0.319979</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1.704944</td>\n",
       "      <td>0.095397</td>\n",
       "      <td>-0.227434</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.059376</td>\n",
       "      <td>-0.669268</td>\n",
       "      <td>1.941193</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-2.281603</td>\n",
       "      <td>0.216062</td>\n",
       "      <td>-0.272205</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.396639</td>\n",
       "      <td>3.624300</td>\n",
       "      <td>2.018026</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        pc1       pc2       pc3\n",
       "0  1.898069 -0.649162  0.319979\n",
       "1 -1.704944  0.095397 -0.227434\n",
       "2  1.059376 -0.669268  1.941193\n",
       "3 -2.281603  0.216062 -0.272205\n",
       "4  1.396639  3.624300  2.018026"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_scaled_pca = pd.DataFrame(\n",
    "    data=X_scaled_pca, columns=[\"pc1\", \"pc2\", \"pc3\"]\n",
    ")\n",
    "df_scaled_pca.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "51684538",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.31832869, 0.18207076, 0.14257394])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pca.explained_variance_ratio_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "04bff7bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\manifold\\_t_sne.py:800: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(768, 2)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Next, further reduce the dataset dimensions with t-SNE and visually inspect the results. In order to accomplish this task, run t-SNE on the principal components: the output of the PCA transformation. Then create a scatter plot of the t-SNE output. Observe whether there are distinct clusters or not.\n",
    "\n",
    "# Run t-SNE\n",
    "tsne = TSNE(learning_rate = 250)\n",
    "tsne_features = tsne.fit_transform(df_scaled_pca)\n",
    "tsne_features.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "2166deb5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(tsne_features[:,0], tsne_features[:,1])\n",
    "plt.savefig(\"img2.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "ebe808ee",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1334: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.\n",
      "  warnings.warn(\n",
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1334: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.\n",
      "  warnings.warn(\n",
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1334: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.\n",
      "  warnings.warn(\n",
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1334: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.\n",
      "  warnings.warn(\n",
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1334: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.\n",
      "  warnings.warn(\n",
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1334: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>k</th>\n",
       "      <th>inertia</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>4062.842734</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2678.137746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2068.619691</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1722.539820</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>1512.494598</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>1344.017363</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   k      inertia\n",
       "0  1  4062.842734\n",
       "1  2  2678.137746\n",
       "2  3  2068.619691\n",
       "3  4  1722.539820\n",
       "4  5  1512.494598\n",
       "5  6  1344.017363"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create an elbow plot to identify the best number of clusters\n",
    "inertia = []\n",
    "k = list(range(1, 7))\n",
    "\n",
    "# Calculate the inertia for the range of k values\n",
    "for i in k:\n",
    "    km = KMeans(n_clusters=i, random_state=0)\n",
    "    km.fit(df_scaled_pca)\n",
    "    inertia.append(km.inertia_)\\\n",
    "    \n",
    "# Create the Elbow Curve using hvPlot\n",
    "elbow_data = {\"k\": k, \"inertia\": inertia}\n",
    "df_elbow = pd.DataFrame(elbow_data)\n",
    "df_elbow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "5b28f636",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot\n",
    "plt.plot(df_elbow['k'], df_elbow['inertia'])\n",
    "plt.xticks(list(range(7)))\n",
    "plt.title('Elbow Curve')\n",
    "plt.xlabel('Number of clusters')\n",
    "plt.ylabel('Inertia')\n",
    "plt.savefig(\"img3.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "04dcdd92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Training Data Score: 0.7725694444444444\n",
      "Logistic Regression Testing Data Score: 0.78125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\cbarr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "classifier = LogisticRegression(random_state=0).fit(X_train, y_train)\n",
    "print(f\"Logistic Regression Training Data Score: {classifier.score(X_train, y_train)}\")\n",
    "print(f\"Logistic Regression Testing Data Score: {classifier.score(X_test, y_test)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "cb757f33-8417-4ad1-b8e8-1ec455a7306e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating a pickle file for the classifier\n",
    "filename = 'diabetes-prediction-rfc-model.pkl'\n",
    "pickle.dump(clf, open(filename, 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "5ece0c4d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "First 10 predictions:   [0 0 0 0 0 0 0 0 0 0]\n",
      "First 10 actual labels: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
     ]
    }
   ],
   "source": [
    "predictions = classifier.predict(X_test)\n",
    "print(f\"First 10 predictions:   {predictions[:10]}\")\n",
    "print(f\"First 10 actual labels: {y_test[:10].tolist()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "d20442cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[108,  15],\n",
       "       [ 27,  42]], dtype=int64)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_true = y_test\n",
    "y_pred = classifier.predict(X_test)\n",
    "confusion_matrix(y_true, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "7abd4417",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    low_risk       0.80      0.88      0.84       123\n",
      "   high_risk       0.74      0.61      0.67        69\n",
      "\n",
      "    accuracy                           0.78       192\n",
      "   macro avg       0.77      0.74      0.75       192\n",
      "weighted avg       0.78      0.78      0.78       192\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, predictions,\n",
    "                            target_names=[\"low_risk\",\"high_risk\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "1692c0cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.78125\n"
     ]
    }
   ],
   "source": [
    "tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n",
    "accuracy = (tp + tn) / (tp + fp + tn + fn) \n",
    "print(f\"Accuracy: {accuracy}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "cd837858",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Classifier Training Score: 1.0\n",
      "Random Forest Classifier Testing Score: 0.8072916666666666\n"
     ]
    }
   ],
   "source": [
    "clf = RandomForestClassifier(random_state=1).fit(X_train, y_train)\n",
    "print(f'Random Forest Classifier Training Score: {clf.score(X_train, y_train)}')\n",
    "print(f'Random Forest Classifier Testing Score: {clf.score(X_test, y_test)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "0e990907",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    low_risk       0.83      0.88      0.85       123\n",
      "   high_risk       0.76      0.68      0.72        69\n",
      "\n",
      "    accuracy                           0.81       192\n",
      "   macro avg       0.79      0.78      0.79       192\n",
      "weighted avg       0.80      0.81      0.80       192\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = clf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred, target_names=[\"low_risk\",\"high_risk\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "2ed779d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.8072916666666666\n"
     ]
    }
   ],
   "source": [
    "tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n",
    "accuracy = (tp + tn) / (tp + fp + tn + fn) \n",
    "print(f\"Accuracy: {accuracy}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "3202d72e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>familyhistory</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Outcome</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3.298000</td>\n",
       "      <td>110.710121</td>\n",
       "      <td>70.935397</td>\n",
       "      <td>27.726000</td>\n",
       "      <td>127.792000</td>\n",
       "      <td>30.885600</td>\n",
       "      <td>0.429734</td>\n",
       "      <td>31.190000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.865672</td>\n",
       "      <td>142.165573</td>\n",
       "      <td>75.147324</td>\n",
       "      <td>31.686567</td>\n",
       "      <td>164.701493</td>\n",
       "      <td>35.383582</td>\n",
       "      <td>0.550500</td>\n",
       "      <td>37.067164</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \\\n",
       "Outcome                                                                      \n",
       "0           3.298000  110.710121      70.935397      27.726000  127.792000   \n",
       "1           4.865672  142.165573      75.147324      31.686567  164.701493   \n",
       "\n",
       "               BMI  familyhistory        Age  \n",
       "Outcome                                       \n",
       "0        30.885600       0.429734  31.190000  \n",
       "1        35.383582       0.550500  37.067164  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "df_copy.groupby('Outcome').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "7ff96e3a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=[5,5])\n",
    "plt.pie(df_copy['Outcome'].value_counts(),labels = ['Non Diabetic','Diabetic'],autopct = '%.2f',\n",
    "        startangle = 60)\n",
    "plt.title('Proportion of Participant',fontsize = 18)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "f3b1b9ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.utils import resample\n",
    "df_major = df_copy[(df_copy['Outcome']==0)]\n",
    "df_minor = df_copy[(df_copy['Outcome']==1)]\n",
    "upsample = resample(df_minor,\n",
    "                    replace = True,\n",
    "                    n_samples = 500,\n",
    "                    random_state= 42)\n",
    "df_copy = pd.concat([upsample, df_major])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "45825293",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "Outcome\n",
       "0    500\n",
       "1    500\n",
       "dtype: int64"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "plt.figure(figsize=[5,5])\n",
    "plt.pie(df_copy['Outcome'].value_counts(),labels = ['Non Diabetic','Diabetic'],autopct = '%.2f',\n",
    "        startangle = 60)\n",
    "plt.title('Proportion of Participant',fontsize = 18)\n",
    "plt.show()\n",
    "\n",
    "df_copy.groupby('Outcome').size()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "cca3d6e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df_copy.drop(['Outcome'],axis = 1)\n",
    "y = df_copy['Outcome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "ed2e581b",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, \n",
    "                                                 random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "02ce2910",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({1: 402, 0: 398})\n"
     ]
    }
   ],
   "source": [
    "from collections import Counter\n",
    "print(Counter(y_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "46753e23",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = StandardScaler()\n",
    "x_train = sc.fit_transform(x_train)\n",
    "x_test = sc.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "3b1338fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[0.8,\n",
       "  0.8681472589035615,\n",
       "  0.7938144329896908,\n",
       "  0.7857142857142857,\n",
       "  0.8020833333333334],\n",
       " [0.9,\n",
       "  0.9709883953581433,\n",
       "  0.9047619047619048,\n",
       "  0.9693877551020408,\n",
       "  0.8482142857142857]]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "def initiate_clf(modeling):\n",
    "  \n",
    "    y_pred = modeling.fit(x_train, y_train).predict(x_test)\n",
    "    Accuracy = accuracy_score(y_test, y_pred)\n",
    "    Roc = roc_auc_score(y_test, modeling.predict_proba(x_test)[:,1])\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    recall = recall_score(y_test,y_pred)\n",
    "    precision = precision_score(y_test,y_pred)\n",
    "    metrics = [Accuracy, Roc, f1,recall, precision]\n",
    "    return metrics\n",
    "\n",
    "modeling = [LogisticRegression(random_state = 0),\n",
    "      RandomForestClassifier(random_state = 0)]\n",
    "\n",
    "metric_list = []\n",
    "for clf in modeling:\n",
    "    initiate_clf(clf)\n",
    "    metrics = initiate_clf(clf)\n",
    "    metric_list.append(metrics)\n",
    "metric_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "8a412b0d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>ROC-AUC</th>\n",
       "      <th>F1-Score</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>LogisticRegression</th>\n",
       "      <td>0.8</td>\n",
       "      <td>0.868147</td>\n",
       "      <td>0.793814</td>\n",
       "      <td>0.785714</td>\n",
       "      <td>0.802083</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RandomForest</th>\n",
       "      <td>0.9</td>\n",
       "      <td>0.970988</td>\n",
       "      <td>0.904762</td>\n",
       "      <td>0.969388</td>\n",
       "      <td>0.848214</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                    Accuracy   ROC-AUC  F1-Score    Recall  Precision\n",
       "LogisticRegression       0.8  0.868147  0.793814  0.785714   0.802083\n",
       "RandomForest             0.9  0.970988  0.904762  0.969388   0.848214"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model  = pd.DataFrame([[metric_list[0][0],metric_list[0][1],metric_list[0][2],metric_list[0][3],metric_list[0][4]],\n",
    "                      [metric_list[1][0],metric_list[1][1],metric_list[1][2],metric_list[1][3],metric_list[1][4]]],\n",
    "                      columns = ['Accuracy','ROC-AUC','F1-Score','Recall','Precision'],\n",
    "                      index = ['LogisticRegression','RandomForest'])\n",
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "5c89d083",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.93, 0.91, 0.94, 0.88, 0.93])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "clf = RandomForestClassifier(random_state = 42)\n",
    "scores = cross_val_score(clf, x, y, cv=5,scoring='recall')\n",
    "scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ff1dc7a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0652e738",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
