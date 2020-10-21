import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelBinarizer

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix

import plotly.figure_factory as ff

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression


class CustomPipeline():
    MODELS ={'CART':{'params':
                  {'class_weight': 'balanced',
                   'criterion': 'entropy',
                    'max_depth': 7,
                    # 'max_features': 31,
                    'max_leaf_nodes': 21, 
                    'min_samples_leaf': 21, 
                    'min_samples_split': 19,
                  },
                  'model': DecisionTreeClassifier,
                 },
            'Random Forest':{'params':
                {'bootstrap': True,
                    'class_weight': 'balanced',
                    'criterion': 'gini',
                    'max_depth': 50,
                    # 'max_features': 5,
                    # 'max_leaf_nodes': 27,
                    # 'min_samples_leaf': 15,
                    'min_samples_split': 50,
                },
                'model': RandomForestClassifier,
                },
            'xgb':{'params':
                    {'alpha': 2,
                    'colsample_bytree': 0.63,
                    'gamma': 8,
                    'lambda': 6,
                    'learning_rate': 0.8,
                    'max_delta_step': 21,
                    'max_depth': 18,
                    'objective': 'binary:hinge',
                    'subsample': 0.85,
                    },
                    'model': XGBClassifier,
                },

            'SVM':{'params':
                        {'tol': 0.001,
                        'penalty': 'l2',
                        'loss': 'squared_hinge',
                        'intercept_scaling': 2,
                        'dual': False,
                        'C': 0.01,
                        'class_weight': 'balanced',
                        },
                        'model': LinearSVC,
                        },
            'KNN':{'params':
                    {
                        # 'leaf_size': 39,
                    'n_neighbors': 20,
                    'p': 1,
                    'weights': 'uniform',
                    # 'n_jobs':-1,
                    },
                    'model': KNeighborsClassifier,
                    }, 

            'Radius Neighbors':{'params':
                    {'leaf_size': 10,
                    'outlier_label': 'most_frequent',
                    'p': 9,
                    'radius': 17.0,
                    'weights': 'distance'
                    },
                    'model': RadiusNeighborsClassifier,
                    },
            'Logistic Regression':{'params':
                    {'tol': 0.001,
                        'solver': 'saga', 
                        'penalty': 'l2',
                        'max_iter': 110,
                        'l1_ratio': 0.09, 
                        'fit_intercept': True, 
                        'dual': False, 
                        'class_weight': 'balanced', 
                        'C': 1.75},
                    'model':LogisticRegression,
                    },
            'Naive Bayes':{'params': {'var_smoothing': 10**-14},
                    'model': GaussianNB},
            # 'cnb':{'params': {'norm': False, 
            #                     'alpha': 0.0},
            #         'model': ComplementNB}
        }
    
    # {
    #     'Random Forest': RandomForestClassifier,
    #     'SVM':,
    #     'KNN': KNeighborsClassifier,
    #     'Linear SVC': LinearSVC,
    #     'Logistic Regression': LogisticRegression, 
    #     'XGBoost': XGBClassifier,
    #     'Naive Bayes':GaussianNB,
    #     },

    VECTORIZERS = {'TF-IDF': TfidfVectorizer, 'BoW': CountVectorizer}

    def __init__(self, vectorizer, model):
        self.vectorizer = self.VECTORIZERS[vectorizer]
        self.model = self.MODELS[model]['model']
        self.params = self.MODELS[model]['params']




    def run(self, data):

        self.pipeline = Pipeline([
            ('lem', Lemma()),
            ('vectorizer', self.vectorizer()),
            ('scaler', CustomScaler()),
            # ('pca', PCA(n_components=10)),
            ('clf', self.model(**self.params))
        ])

        X_train, X_test, y_train_lbl, y_test_lbl = train_test_split(data.iloc[:, 0],  data.iloc[:, 1], test_size=0.20)

        # Binarize labels
        self.binarizer = LabelBinarizer().fit(y_train_lbl)
        y_train = self.binarizer.transform(y_train_lbl)
        y_test = self.binarizer.transform(y_test_lbl)

        # Fit pipeline
        self.pipeline.fit(X_train, y_train.ravel())

        y_preds = self.pipeline.predict(X_test)

        clf_report = pd.DataFrame(classification_report(y_test, y_preds, output_dict=True))
        clf_report.rename(columns={'0': 'Negative', '1': 'Positive'}, inplace=True)
        cm = confusion_matrix(y_test, y_preds) #plot_confusion_matrix(self.pipeline, X_test, y_test, display_labels= self.binarizer.classes_) #

        return clf_report, self.get_cm_plot(cm)

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[0]
        return f"this comment is {proba[0]:.2%} Negative and {proba[1]:.2%} Positive"

    def plot(self, data):
        fig = sns.scatterplot(x_pca[:,0], x_pca[:,1], df.sentiment)
        fig.set(ylim=(0, 10), xlim=(-2, 2))
        return fig
        
    def get_cm_plot(self, cm):
        z = cm

        x = list(self.binarizer.classes_)
        # y = list(self.binarizer.classes_)
        y = x[::]
        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]

        # set up figure 
        fig = ff.create_annotated_heatmap(z, x=x, y=y, 
                                        annotation_text=z_text,
                                        colorscale='Viridis' )

        # add title
        fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
        #                   xaxis = dict(title='Predictions'),
        #                   yaxis = dict(title='True')
                        )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.35,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        # fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig['data'][0]['showscale'] = True
        # fig.show()



        # labels= self.binarizer.classes_
        # # labels = ['business', 'health']
        # # cm = confusion_matrix(y_test, pred, labels)
        # fig = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
        # fig.set_title('Confusion matrix of the classifier')
        # # fig.colorbar(cax)
        # fig.set_xlabel('Predicted')
        # fig.set_ylabel('True')
        return fig





class Lemma():        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
      ## Tokenize

        tokenizer = RegexpTokenizer(r"(@\w+|\w+)")
        if type(X) == str:
            X = pd.Series(X)
        words = X.apply(tokenizer.tokenize)
        
      ## Remove stopwords
        sw = stopwords.words('english')
        words = words.apply(lambda x: [i for i in x if i not in sw])
        
      ## Lemmatize
        lemmatizer = WordNetLemmatizer()
        return words.apply(lambda s: ' '.join([lemmatizer.lemmatize(x) for x in s]))


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.toarray())
        return self

    def transform(self, X):
        return self.scaler.transform(X.toarray())