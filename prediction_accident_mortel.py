# -*- coding: utf-8 -*-
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier

##############################
##########################
years = range(2010, 2015)

# Concatenation et lecture caractéristiques
pieces_caracteritiques = []
for year in years:
    path = 'data/caracteristiques_%d.csv' %year
    frame = pd.read_csv(path)
    frame['an'] = year
    pieces_caracteritiques.append(frame)
    
caracteristiques = pd.concat(pieces_caracteritiques, ignore_index=True)

# Concaténation et lecture lieux
pieces_lieux = []
for year in years:
    path = './data/lieux_%d.csv' %year
    frame = pd.read_csv(path)
    pieces_lieux.append(frame)
lieux = pd.concat(pieces_lieux, ignore_index=True)

# Contaténation et lecture usagers
pieces_usagers = []
for year in years:
    path = './data/usagers_%d.csv' %year
    frame = pd.read_csv(path)
    pieces_usagers.append(frame)
usagers = pd.concat(pieces_usagers, ignore_index=True)

# Contaténation et lecture véhicules
pieces_vehicules = []
for year in years:
    path = './data/vehicules_%d.csv' %year
    frame = pd.read_csv(path)
    pieces_vehicules.append(frame)
vehicules = pd.concat(pieces_vehicules, ignore_index=True)

# Base de données 
learning = DataFrame({"Num_Acc": caracteristiques.Num_Acc,
                      "agg": caracteristiques.agg,
                      "int": caracteristiques['int'],
                      "atm": caracteristiques.atm,
                      "catr": lieux.catr,
                      "circ": lieux.circ,
                      "plan": lieux.plan,
                      "surf": lieux.surf,
                      "annee": caracteristiques.an,
                      "lum": caracteristiques.lum,
                      "jour": caracteristiques.jour,
                      "mois": caracteristiques.mois})
                      
# Equipement de sécurité
z = usagers.secu
d =[]
for i in z:
    s = str(i)
    if s[1] == '1':
       d.append(0)
    else:
        d.append(1)

usagers['securite'] = d       
inter = DataFrame(usagers[usagers.securite == 1].groupby('Num_Acc')['securite'].count())
inter.reset_index(level=0, inplace=True)
learning= pd.merge(learning, inter[['Num_Acc', 'securite']], how= 'left', on=['Num_Acc'])
learning['securite'] = learning['securite'].fillna(0)
del z

# Jour de semaine
z = pd.to_datetime(learning.annee*10000+learning.mois*100+learning.jour,format='%Y%m%d')
l =[]
for i in z:
    s = str(i)
    l.append(i.weekday())

learning['jour_de_semaine'] = l
learning['jour_de_semaine'] = (learning.jour_de_semaine >=5).astype(float)

# Conditions d'éclairage    
learning['pas_nuit'] = pd.Series((learning.lum == 1).astype(float), index=learning.index)
learning['nuit_avec_eclairage'] = pd.Series((learning.lum == 5).astype(float), index=learning.index)
learning['nuit_sans_eclairage'] = pd.Series(((learning.lum != 5) & (learning.lum != 1) ).astype(float), index=learning.index)

# Intersection ou pas
learning['int'] = pd.Series((learning.int != 1).astype(float), index=learning.index)

# Ordonnement des conditions atmoshériques
learning['atm'] = learning['atm'].map({1: 1, 2: 7, 3: 8, 4: 2, 5: 5, 6: 3, 7: 4, 8: 6, 9: 9})

# Nbr de piétons impliqués
inter = DataFrame(usagers[usagers.catu == 3].groupby('Num_Acc')['catu'].count())
inter.reset_index(level=0, inplace=True)
learning= pd.merge(learning, inter[['Num_Acc', 'catu']], how= 'left', on=['Num_Acc'])
learning['catu'] = learning['catu'].fillna(0)

# Manoeuvre avant l'accident ou pas
inter = DataFrame(vehicules[(vehicules.manv != 1) & (vehicules.manv != 2) & (vehicules.manv.notnull()) 
                            & (vehicules.manv != 0) ].groupby('Num_Acc')['manv'].count())
inter.reset_index(level=0, inplace=True)
learning= pd.merge(learning, inter[['Num_Acc', 'manv']], how= 'left', on=['Num_Acc'])
learning['manv'] = learning['manv'].fillna(0)

# Nombre de véhicules par catégorie
vehicules['bicyclette'] = ((vehicules.catv == 1) | (vehicules.catv == 2)).astype(int)
vehicules['VL'] = ((vehicules.catv == 7) | (vehicules.catv == 8) | (vehicules.catv == 9)).astype(int)
vehicules['VSP'] = ((vehicules.catv == 3)).astype(int)
vehicules['Moto'] = ((vehicules.catv == 4) | (vehicules.catv == 30) | (vehicules.catv == 32) | (vehicules.catv == 34)).astype(int)
vehicules['Motocyclette'] = ((vehicules.catv == 5) | (vehicules.catv == 6) | (vehicules.catv == 31) | (vehicules.catv == 33)).astype(int)
vehicules['Quadricycle'] = ((vehicules.catv == 35) | (vehicules.catv == 36)).astype(int)
vehicules['VU'] = ((vehicules.catv == 10) | (vehicules.catv == 11) | (vehicules.catv == 12)).astype(int)
vehicules['PL'] = ((vehicules.catv == 13) | (vehicules.catv == 14) | (vehicules.catv == 15)).astype(int)
vehicules['TracteurR'] = ((vehicules.catv == 16)).astype(int)
vehicules['TracteurRM'] = ((vehicules.catv == 17)).astype(int)
vehicules['TEC'] = ((vehicules.catv == 18)).astype(int)
vehicules['Tramway'] = ((vehicules.catv == 19) | (vehicules.catv == 40)).astype(int)
vehicules['special'] = ((vehicules.catv == 20)).astype(int)
vehicules['TracteurA'] = ((vehicules.catv == 21)).astype(int)
vehicules['Autobus'] = ((vehicules.catv == 37) | (vehicules.catv == 38)).astype(int)
vehicules['Train'] = ((vehicules.catv == 39)).astype(int)
vehicules['Autre'] = ((vehicules.catv == 99)).astype(int)

a = DataFrame({"Bicyclette": vehicules.groupby(['Num_Acc'])['bicyclette'].sum(), 
               "VL": vehicules.groupby(['Num_Acc'])['VL'].sum(),
               "VSP": vehicules.groupby(['Num_Acc'])['VSP'].sum(), 
               "Moto": vehicules.groupby(['Num_Acc'])['Moto'].sum(),
               "Motocyclette": vehicules.groupby(['Num_Acc'])['Motocyclette'].sum(), 
               "Quadricycle": vehicules.groupby(['Num_Acc'])['Quadricycle'].sum(),
               "VU": vehicules.groupby(['Num_Acc'])['VU'].sum(), 
               "PL": vehicules.groupby(['Num_Acc'])['PL'].sum(),
               "TracteurR": vehicules.groupby(['Num_Acc'])['TracteurR'].sum(), 
               "TracteurRM": vehicules.groupby(['Num_Acc'])['TracteurRM'].sum(),
               "TEC": vehicules.groupby(['Num_Acc'])['TEC'].sum(), 
               "Tramway": vehicules.groupby(['Num_Acc'])['Tramway'].sum(),
               "special": vehicules.groupby(['Num_Acc'])['special'].sum(),
               "TracteurA": vehicules.groupby(['Num_Acc'])['TracteurA'].sum(), 
               "Autobus": vehicules.groupby(['Num_Acc'])['Autobus'].sum(),
               "Train": vehicules.groupby(['Num_Acc'])['Train'].sum(), 
               "Autre": vehicules.groupby(['Num_Acc'])['Autre'].sum()})
a.reset_index(level=0, inplace=True)
learning= pd.merge(learning, a, how= 'left', on=['Num_Acc'])

# Drop Na
learning = learning.dropna() 

# Gravité : accident mortel ou pas
z = DataFrame(usagers[(usagers.grav == 2)].groupby('Num_Acc')['grav'].count())
z.reset_index(level=0, inplace=True)
learning= pd.merge(learning, z, how= 'left', on=['Num_Acc'])
learning['grav'] = learning['grav'].fillna(0)

######################
#####################
#####################
learning = learning[(learning.annee == 2010) | (learning.annee == 2011) ]
X = learning.drop(['grav', 'Num_Acc', 'jour', 'mois', 'annee'], axis=1)
#xtrain = X[ (X.annee == 2010) | (X.annee == 2011) ]
#xtrain = xtrain.drop(['annee'], axis=1)
#xtest = X[(X.annee == 2012)]
#xtest = xtest.drop(['annee'], axis=1)

Y = learning.grav
Y = (Y >= 1).astype(float)
#ytrain = Y[(X.annee == 2010) | (X.annee == 2011)]
#ytest = Y[(X.annee == 2012)]


# Cross validation : choix aléatoire des bases d'apprentissage et de validation
xtrain, xtest, ytrain, ytest = sklearn.cross_validation.train_test_split(X, Y, train_size=0.5)

#########################
# Linear Regression #####
#########################
linreg = LinearRegression()
linreg.fit(xtrain, ytrain)

#############################
# Logostic Regression #######
#############################
lreg = LogisticRegression(penalty='l2')
lreg = lreg.fit(xtrain, ytrain)

####################### 
# Gaussian NB #######
#######################
gnb =  GaussianNB()
gnb = gnb.fit(xtrain, ytrain)

####################### 
# Regression Rigide ###
#######################
ridge = GridSearchCV(Ridge(),
                     {'alpha': np.logspace(-10, 10, 10)})
ridge.fit(xtrain, ytrain)

#################################### 
# GradientBoostingClassifier #######
####################################
gbm = GradientBoostingClassifier(n_estimators=500)
gbm.fit(xtrain, ytrain)

####################### 
# LASSO #######
#######################
lasso = GridSearchCV(Lasso(),
                     {'alpha': np.logspace(-10, -8, 5)})
lasso.fit(xtrain, ytrain)

####################### 
# Decision Tree #######
#######################
tree = GridSearchCV(DecisionTreeClassifier(),
                    {'max_depth': np.arange(3, 10)})
tree.fit(xtrain, ytrain)

####################### 
# Random Forest #######
#######################
#rfc = GridSearchCV(RandomForestClassifier(n_estimators=200)),
 #                   {'n_estimators': [100, 200, 500]})

rfc = RandomForestClassifier(n_estimators=200)   
rfc = rfc.fit(xtrain, ytrain)                
###############################################################################
# Plot calibration plots
###############################################################################

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in[(linreg, 'Linear Regression'),
                (lreg, 'Logistic'),
                (gnb, 'Naive Bayes'),
                (ridge, 'Rigide'),
                (rfc, 'Random Forest'),
                (lasso, 'Lasso'),
                (gbm, 'GradientBoostingClassifier')]:
    if hasattr(clf, "predict_proba"):
       predicted = clf.predict_proba(xtest)[:, 1]
    else:
        predicted = clf.predict(xtest)
    
    fpr, tpr, _ = roc_curve(ytest, predicted)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label='%s : ROC curve (area = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

