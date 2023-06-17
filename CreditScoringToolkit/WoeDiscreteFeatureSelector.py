import pandas as pd
from .WoeBaseFeatureSelector import WoeBaseFeatureSelector

class WoeDiscreteFeatureSelector(WoeBaseFeatureSelector):
    """
        Class for selecting discrete features based on their WoE transformation and 
        Information Value statistic. 
    """
    iv_report = None 
    def __init__(self):
        super().__init__()
    
    def fit(self,X:pd.DataFrame,y:pd.Series,iv_threshold:float=0.1)->None:
        """Learns best features given an IV threshold.

        Args:
            X (pd.DataFrame): Discrete predictors data
            y (pd.Series): Dichotomic response feature
            iv_threshold (float, optional):  IV value for a feature to be included in final selection. Defaults to 0.1.
        """
        disc_features = list(X.columns)
        aux = X.copy()
        aux['binary_target'] = y 
        iv = [(feature,self._information_value(aux[feature],aux['binary_target'])) for feature in disc_features]
        self.iv_report = pd.DataFrame(iv,columns=['feature','iv'])
        self.iv_report['selected'] = self.iv_report['iv']>=iv_threshold
        self.iv_report.sort_values('selected',ascending=False,inplace=True)
        iv = [(feature,value) for feature,value in iv if value>=iv_threshold]
        iv = pd.DataFrame(iv,columns=['feature','iv'])
        disc_features = list(iv['feature'])
        self.selected_features =  iv.set_index('feature')['iv'].to_dict()
        self.__is_fitted = True        
    
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        """Transforms data keeping only the selected features

        Args:
            X (pd.DataFrame): Discrete predictors data

        Raises:
            Exception: If fit method is not called first.
            Exception: If a fitted feature is not present in data.

        Returns:
            pd.DataFrame: Data containing best discrete features 
        """
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = [feature for feature in self.selected_features.keys()]
            non_present_features = [f for f in features if f not in X.columns]
            if len(non_present_features)>0:
                print(f'{",".join(non_present_features)} feature{"s" if len(non_present_features)>1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                aux = aux[features]
                return aux
    