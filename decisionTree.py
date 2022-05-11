import pandas as pd
import numpy as np

class Node:
  def __init__(self, feature_values, feature_name, impurity):
    self.true = True
    self.false = False
    self.feature_values = feature_values
    self.feature_name = feature_name
    self.impurity = impurity
  
  def get_feature_values(self):
    return self.feature_values
  
  def get_impurity(self):
    return self.impurity
  
  def get_true(self):
    return self.true
  
  def get_false(self):
    return self.false
  
  def get_feature_name(self):
    return self.feature_name
  
  def set_true(self, node):
    if node is None:
      node = True
    self.true = node
  
  def set_false(self, node):
    if node is None:
      node = False
    self.false = node
  
  def __str__(self):
    return f"{self.get_feature_name()}"
    
    

class DecisionTree:
  def __init__(self):
    self.tree = None

  def __weighted_values(self, v1, v2):
    if (v1 + v2) == 0:
      return 0, 0
    return v1/(v1 + v2), v2/(v1 + v2)

  def __gini(self, true_count, false_count):
    t1, t2 = self.__weighted_values(true_count, false_count)
    return 1.0 - t1*t1 - t2*t2
  
  def __get_node_impurity(self, mat):
    g_false = self.__gini(mat[0][1], mat[0][0])
    g_true = self.__gini(mat[1][1], mat[1][0])
    w_false, w_true = self.__weighted_values(mat[0][0]+mat[0][1], mat[1][0] + mat[1][0])
    return w_false * g_false + w_true * g_true

  def __get_power_set(self, lis):
    n = len(lis)
    ps = []
    for i in range(1, pow(2, n) - 1, 1):
      bi = (bin(i).replace('0b','')).rjust(n, "0")
      ret = []
      for j in range(len(bi)):
        if bi[j] == '1':
          ret.append(lis[j]) 
      ps.append(ret)
    return ps

  def __get_feature_atomic_impurity(self, X, y, feature):
    best_impurity = 100
    best_node = None
    unique_values = X[feature].unique()

    for uniques in self.__get_power_set(unique_values):
      mat = [
          # True False
          [0, 0], # True wrt feature
          [0, 0] # False ,,   ,,
      ]
      for i in range(X.shape[0]):
        r = int(X.iloc[i][feature] in uniques)
        c = int(y[i])
        mat[r][c] += 1
      impurity = self.__get_node_impurity(mat)
      if impurity < best_impurity:
        best_impurity = impurity
        best_node = Node(uniques, feature, impurity)
    return best_node, best_impurity


  def __get_best_node(self, X, y):
    features = list(X.columns)

    best_impurity = 100
    best_node = None

    for feature in features:
      node, impurity = self.__get_feature_atomic_impurity(X, y, feature)
      if impurity < best_impurity:
        best_impurity = impurity
        best_node = node
    return best_node, best_impurity

  def __get_child_data(self, X, y, parent_node):
    feature_name = parent_node.get_feature_name()
    feature_values = parent_node.get_feature_values()

    X_false = X.copy()
    X_true = X.copy()

    false_lis, true_lis = [], []

    for i in range(X.shape[0]):
      if X.iloc[i][feature_name] in feature_values:
        true_lis.append(i)
      else:
        false_lis.append(i)

    X_false = X_false.drop(labels = true_lis, axis = 0).drop(feature_name, axis = 1)
    X_true = X_true.drop(labels = false_lis, axis = 0).drop(feature_name, axis = 1)
    y_false = np.delete(y, true_lis, axis = 0)
    y_true = np.delete(y, false_lis, axis = 0)

    return (X_false.reset_index().drop("index", axis = 1), y_false, 
        X_true.reset_index().drop("index", axis=1), y_true)

  def __build_tree(self, X, y, parent_impurity = 100):
    if X.empty:
      return None
    feature_node, impurity = self.__get_best_node(X, y)
    if impurity >= parent_impurity:
      return None

    X_false, y_false, X_true, y_true = self.__get_child_data(X, y, feature_node)

    feature_node.set_true(self.__build_tree(X_true, y_true, impurity))
    feature_node.set_false(self.__build_tree(X_false, y_false, impurity))
    return feature_node
    
  def fit(self, X, y):
    self.tree = self.__build_tree(X, y)
  
  def __predict_single(self, x, parent):
    if type(parent) == bool:
      return parent
    
    feature_name = parent.get_feature_name()
    feature_values = parent.get_feature_values()

    val = x[feature_name]

    if val in feature_values:
      return self.__predict_single(x, parent.get_true())
    return self.__predict_single(x, parent.get_false())

  def predict(self, X):
    preds = []
    for i in range(X.shape[0]):
      pred = self.__predict_single(X.iloc[i], self.tree)
      preds.append(pred)
    return preds




df = pd.read_csv('dataset.csv').drop("ID", axis = 1)
X_train = df.copy().drop("Buys", axis = 1).drop([df.shape[0]-1], axis=0)
X_test = df.iloc[df.shape[0]-1:].drop("Buys", axis = 1)
y_train = np.array(df["Buys"].drop([df.shape[0]-1], axis = 0)) == "Yes"
print(X_train)
print(X_test)


dt = DecisionTree()

dt.fit(X_train, y_train)

print(dt.tree.get_feature_name())
print(dt.predict(X_test))
