from .model import model

import gshap
import numpy as np
import pandas as pd

class Explainer(gshap.KernelExplainer):
    explanation_functions = {}
    juv_offenses_cols = ['juv_fel_count', 'juv_misd_count', 'juv_other_count']
    column_mapping = dict(
        male='gender',
        age='age',
        juv_offenses='juvenile offenses',
        priors_count='prior convictions',
        felony='the charge',
        married='marital status'
    )
    
    def __init__(self, model, data, g=lambda x: x):
        super().__init__(model, data, g)
        self.columns = data.columns
        self.base_rate = model(data).mean()
        self.base_rate_explanation = '''
        <p>The average offender has a {} in 100 chance of committing another crime within 2 years.</p>
        '''.format(round(100*self.base_rate)).strip()
        
    def shap_values(self, X, nsamples=64):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        shap_values = super().gshap_values(X, nsamples=nsamples)
        df = pd.DataFrame(shap_values.T, columns=self.columns)
        df['juv_offenses'] = df[self.juv_offenses_cols].sum(axis=1)
        return df.drop(columns=self.juv_offenses_cols + ['black']) 
        
    def explain_observations(
            self, X, output=None, nsamples=64, threshold=.04
        ):
        def get_profile_df(X):
            if isinstance(X, pd.Series):
                X = X.to_frame().T
            df = X.copy()
            df['juv_offenses'] = df[self.juv_offenses_cols].sum(axis=1)
            return df.drop(columns=self.juv_offenses_cols + ['black'])   
        
        profile_df = get_profile_df(X)
        output = self.model(X) if output is None else output
        shap_df = self.shap_values(X, nsamples)
        return [
            self.explain_observation(
                profile_df.iloc[i], shap_df.iloc[i], output[i], threshold
            ) 
            for i in range(len(profile_df))
        ]
    
    def explain_observation(
            self, profile, shap_values, output, threshold=.04
        ):  
        def get_feature_explanations():
            if not any(shap_values):
                return '''
                <p>This offender is similar to the average offender.</p>
                '''
            return '\n'.join([
                self.explanation_functions[col](i, profile[col], shap_values, predictions[i])
                for i, col in enumerate(shap_values.index)
            ])
            
        def other_factors():
            return '''
            <p>After making small adjustments for other factors, my final estimate is that there is a <b>{} in 100</b> chance the offender will commit another crime within 2 years.</p>
            '''.format(round(100*output))

        shap_values = shap_values.sort_values(key=lambda x: -abs(x))
        predictions = np.cumsum(shap_values) + self.base_rate
        shap_values = shap_values[
            (abs(shap_values) > threshold) | abs(predictions - predictions[-1] > threshold)
        ]
        return self.base_rate_explanation + get_feature_explanations() + other_factors()
    
    @classmethod
    def variable_explanation(cls, key):
        def most_important_factor():
            return '''
            The most important factor when considering this specific offender is {}.
            '''.format(cls.column_mapping[key]).strip()
        
        def conjunctive_adverb(idx, shap_values):
            if idx == 0:
                return ''
            return (
                'Additionally,' if (shap_values[idx]*shap_values[idx-1]) > 0 
                else 'However,'
            )
        
        def effect(shap_value, prediction):
            return '''
            This <b>{}</b> the chances the offender will commit another crime to {} in 100.
            '''.format(
                'decreases' if shap_value < 0 else 'increases',
                min(round(100*prediction), 100)
            ).strip()
        
        def inner(func):
            def wrapper(idx, profile_value, shap_values, prediction):
                explanation = []
                if idx == 0:
                    explanation.append(most_important_factor())
                comparison = ' '.join([
                    conjunctive_adverb(idx, shap_values),
                    func(profile_value).strip()
                ]).strip()
                comparison = comparison[0].upper() + comparison[1:]
                explanation += [comparison, effect(shap_values[idx], prediction)]
                return '<p>{}</p>'.format(' '.join(explanation))
            
            cls.explanation_functions[key] = wrapper
            return wrapper
        
        return inner
    
@Explainer.variable_explanation('male')
def gender_explanation(male):
    return '''
    this offender is a <b>{}</b>, and {} are {} likely to commit future crimes.
    '''.format(
        'man' if male else 'woman',
        'men' if male else 'women',
        'more' if male else 'less'
    )

@Explainer.variable_explanation('age')
def age_explanation(age):
    comparison = get_comparison('age', age)
    return '''
    {} offenders are {} likely to commit future crimes. This offender is {} most offenders. 
    '''.format(
        'younger' if comparison < 2 else 'older',
        'more' if comparison < 2 else 'less',
        comparison_mapping['age'][comparison],
    )

@Explainer.variable_explanation('juv_offenses')
def juvenile_offenses(juv_offenses):
    comparison = get_comparison('juv_offenses', juv_offenses)
    return '''
    offenders with {} juvenile offenses are {} likely to commit future crimes. This offender has {} most offenders.
    '''.format(
        'fewer' if comparison < 2 else 'more',
        'less' if comparison < 2 else 'more',
        comparison_mapping['juv_offenses'][comparison]
    )
        
@Explainer.variable_explanation('priors_count')
def priors_explanation(priors_count):
    comparison = get_comparison('priors_count', priors_count)
    return '''
    offenders with {} prior convictions are {} likely to commit future crimes. This offender has {} most offenders.
    '''.format(
        'fewer' if comparison < 2 else 'more',
        'less' if comparison < 2 else 'more',
        comparison_mapping['priors_count'][comparison]
    )

@Explainer.variable_explanation('felony')
def felony(felony):
    return '''
    offenders whose most recent charge was a {} are {} likely to commit future crimes. This offender's most recent charge was a <b>{}</b>.
    '''.format(
        'felony' if felony else 'misdemeanor',
        'more' if felony else 'less',
        'felony' if felony else 'misdemeanor'
    )

@Explainer.variable_explanation('married')
def marital_status(married):
    return '''
    {} offenders are {} likely to commit future crimes. This offender is <b>{}</b>.
    '''.format(
        'married' if married else 'unmarried',
        'less' if married else 'more',
        'married' if married else 'unmarried'
    )

sum_df = pd.read_csv('survey/summary.csv', index_col=0)

def get_comparison(column, val):
    sum_column = sum_df[column]
    if val < sum_column['25%']:
        return 0
    if val < sum_column['50%']:
        return 1
    if val == sum_column['50%']:
        return 2
    if val < sum_column['75%']:
        return 3
    return 4

comparison_mapping = dict(
    priors_count=[
        '<b>fewer prior convictions</b> than',
        '<b>fewer prior convictions</b> than',
        'the same number of prior convictions as',
        '<b>slightly more prior convictions</b> than',
        '<b>many more prior convictions</b> than'
    ],
    age=[
        '<b>much younger</b> than',
        '<b>slightly younger</b> than',
        'the same age as',
        '<b>slightly older</b> than',
        '<b>much older</b> than'
    ],
    juv_offenses=[
        '<b>many fewer juvenile offenses</b> than',
        '<b>slightly fewer juvenile offenses</b> than',
        'the same number of juvenile offenses as',
        '<b>more juvenile offenses</b> than',
        '<b>more juvenile offenses</b> than'
    ]
)