from sklearn import tree

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive

X_train, y_train, X_test, y_test = titanic_survive()

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

explainer = ClassifierExplainer(
                model, X_test, y_test,
                cats=['Sex', 'Deck', 'Embarked'],
                labels=['Not survived', 'Survived'])

db = ExplainerDashboard(explainer, title="Titanic Explainer",
                    importances=False,
                    model_summary=True,
                    contributions=True,
                    whatif=False,
                    shap_dependence=True,
                    shap_interaction=True,
                    decision_trees=True)

# Machine Learning dashboard is useful to visualize the machine learning models. 
# I am using ExplainerDashboard to create dashboard for Sklearn model.
# We can take any custom dataset and create dashboard. However, I couldn't do that in the given time.
  
