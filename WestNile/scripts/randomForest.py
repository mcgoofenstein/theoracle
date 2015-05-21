__author__ = 'michael'

import pandas,numpy, sys
from sklearn import ensemble, preprocessing

def getMonth(date):
    return date.split("-")[1]

def getDay(date):
    return date.split("-")[2]

def getData(**kwargs):
    data = {}
    if kwargs.has_key("labels") and kwargs.has_key("train"):
        key = kwargs["labels"]
        data[key] = pandas.read_csv(kwargs["train"]).get(key)
        data["train"] = pandas.read_csv(kwargs["train"])
    for name, arg in [(name, arg) for name, arg in kwargs.iteritems() if name != "train" and name !="labels"]:
            data[name] = pandas.read_csv(arg)
    return data


sprayDataPath = "/home/michael/Kaggle/west_nile/"

#read the data
data = getData(train=sys.argv[1], test=sys.argv[2], sample=sys.argv[3], weather=sys.argv[4], labels="WnvPresent")
data["weather"] = data["weather"].drop("CodeSum", axis=1)

#parse the data
#labels = train.WnvPresent.values #obtain training labels

weather_st1 = data["weather"["Station"] == 1]
weather_st2 = weather[weather["Station"] == 2]
weather_st1 = weather_st1.drop("Station", axis=1)
weather_st2 = weather_st2.drop("Station", axis=1)
weather = weather_st1.merge(weather_st2, on="Date")

weather = weather.replace("M", -1).replace("-", -1).replace("T", -1).replace(" T", -1).replace("  T", -1) #replace missing values

train["Lat_int"] = train.Latitude.apply(int)
train["Long_int"] = train.Longitude.apply(int)
test["Lat_int"] = test.Latitude.apply(int)
test["Long_int"] = test.Longitude.apply(int)

train = train.drop(["Address", "AddressNumberAndStreet", "WnvPresent", "NumMosquitos"], axis=1)
test = test.drop(["Id", "Address", "AddressNumberAndStreet"], axis=1)

train = train.merge(weather, on="Date")
test = test.merge(weather, on="Date")
train = train.drop(["Date"], axis=1)
test = test.drop(["Date"], axis=1)

labeler = preprocessing.LabelEncoder()

labeler.fit(list(train["Species"].values) + list(test["Species"].values))
train["Species"] = labeler.transform(train["Species"].values)
test["Species"] = labeler.transform(test["Species"].values)

labeler.fit(list(train["Street"].values) + list(test["Street"].values))
train["Street"] = labeler.transform(train["Street"].values)
test["Street"] = labeler.transform(test["Street"].values)

labeler.fit(list(train["Trap"].values) + list(test["Trap"].values))
train["Trap"] = labeler.transform(train["Trap"].values)
test["Trap"] = labeler.transform(test["Trap"].values)

train = train.ix[:,(train != -1).any(axis=0)]
test = test.ix[:,(test != -1).any(axis=0)]

trapsAndTrees = ensemble.RandomForestClassifier(n_jobs = -1, n_estimators = 1000, min_samples_split=1)
trapsAndTrees.fit(train, labels)

predictions = trapsAndTrees.predict_proba(test)[:,1]
sample["WnvPresent"] = predictions
sample.to_csv("Abhihsek_random_forest.csv", index = False)
