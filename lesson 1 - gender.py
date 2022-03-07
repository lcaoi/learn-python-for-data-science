from sklearn import tree
from sklearn import ensemble
from sklearn import svm

# [height, weight, shoe_size]
X = [[181, 80, 40], [177, 78, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()  # clf = classifier
clf = clf.fit(X, Y)  # fit method trained our decision tree

prediction = clf.predict([[150, 40, 30]])   # output: female
prediction2 = clf.predict([[150, 90, 30]])  # output: male
prediction3 = clf.predict([[190, 90, 40]])  # output: male
prediction4 = clf.predict([[170, 60, 35]])  # output: female

print(prediction)
print(prediction2)
print(prediction3)
print(prediction4, '\n')


# RandomForestClassifier
clf2 = ensemble.RandomForestClassifier()
clf2 = clf2.fit(X, Y)

prediction5 = clf2.predict([[150, 40, 30]])   # output: female
prediction6 = clf2.predict([[150, 90, 30]])  # output: female
prediction7 = clf2.predict([[190, 90, 40]])  # output: male
prediction8 = clf2.predict([[170, 60, 35]])  # output: female

print(prediction5)
print(prediction6)
print(prediction7)
print(prediction8, '\n')


# DecisionTreeClassifier
clf3 = svm.SVC(probability=True)  # clf = classifier
clf3 = clf3.fit(X, Y)

prediction9 = clf3.predict([[150, 40, 30]])   # output: female
prediction10 = clf3.predict([[150, 90, 30]])  # output: male
prediction11 = clf3.predict([[190, 90, 40]])  # output: male
prediction12 = clf3.predict([[170, 60, 35]])  # output: male

print(prediction9)
print(prediction10)
print(prediction11)
print(prediction12, '\n')

# using [170, 60, 35] gives different result
probaDT = clf.predict_proba ([[170, 60, 35]])
probaRandomForest = clf2.predict_proba ([[170, 60, 35]])
probaSVC = clf3.predict_proba ([[170, 60, 35]])
print(probaDT)
print(probaRandomForest)
print(probaSVC)

print("\n")

# [150, 40, 30] gives same result
probaDT = clf.predict_proba ([[150, 40, 30]])
probaRandomForest = clf2.predict_proba ([[150, 40, 30]])
probaSVC = clf3.predict_proba ([[150, 40, 30]])
print(probaDT)
print(probaRandomForest)
print(probaSVC)

