import gshap
import pandas as pd
import xgboost as xgb
from flask_login import current_user
from gshap.datasets import load_recidivism
from hemlock import (
    Branch, Page, Embedded, Binary, Check, Label, Range, Input, 
    Compile as C, Validate as V, Submit as S, route
)
from hemlock.tools import (
    Assigner, comprehension_check, consent_page, completion_page, html_list
)
from hemlock_demographics import demographics

from random import choices

N_PRACTICE, N_FCAST = 1, 1

X = pd.read_csv('X_test.csv')
df = X.describe()
y = pd.read_csv('y_test.csv')
p_recid = y.mean()

avg_offender_label = (
    '<p>Here is some information about the criminal population in Broward County, Florida</p>'
    + html_list(
        '{} of every 100 offenders commit another crime within 2 years.'.format(round(100*y['two_year_recid'].mean())),
        'The average offender has {} prior convictions'.format(round(df.priors_count['mean'])),
        'The average offender is {} years old'.format(round(df.age['mean'])),
        '{} of every 100 offenders have committed a felony'.format(round(100*df.felony['mean'])),
        '{} of every 100 offenders are Black'.format(round(100*df.black['mean'])),
        '{} of every 100 offenders are male'.format(round(100*df.male['mean'])),
        'The average offender has {} juvenile felonies, {} juvenial misdemeanors, and {} other juvenile offenses'.format(
            round(df.juv_fel_count['mean']), 
            round(df.juv_misd_count['mean']), 
            round(df.juv_other_count['mean'])
        ),
        '{} of every 100 offenders are married'.format(round(100*df.married['mean'])),
        ordered=False
    )
)


class Model():
    def __init__(self, clf, X, t=0):
        self.clf = clf
        self.black_idx = list(X.columns).index('black')
        self.t = t
        
    def predict_proba(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        black = X[:,self.black_idx]
        output = self.clf.predict(xgb.DMatrix(X))
        return (
            (output < .5)*(
                (1-black)*output*((.5+self.t)/.5)
                + black*output*((.5-self.t)/.5)
            )
            + (output >= .5)*(
                (1-black)*(1-(1-output)*((1-.5-self.t)/(1-.5)))
                + black*(1-(1-output)*((1-.5+self.t)/(1-.5)))
            )
        )
        
    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        black = X[:,self.black_idx]
        output = self.clf.predict(xgb.DMatrix(X))
        output += (1-black)*self.t - black*self.t
        return output > .5


bst = xgb.Booster()
bst.load_model('recidivism.model')
model = Model(bst, X, t=.047)

assigner = Assigner({'Algorithm': (0,1)})

consent_label = '''
<p>Hello! We are researchers at the University of Pennsylvania and are interested how you look forward to predict the future. We will show profiles of criminal offenders and ask you to predict how likely they are to commit future crimes. Please read the information below and if you wish to participate, indicate your consent.</p>

<p><b>Because this is an experimental platform, you may encounter errors during this survey. If you experience an error, please email Dillon Bowen at dsbowen@wharton.upenn.edu. Copy this email address now in case you encounter an error during the survey.</b></p>

<p><b>Purpose.</b> The purpose of this study is to explore how people think about the future.</p> 

<p><b>Procedure.</b> You will be asked to complete a survey that will take approximately 20 minutes.</p> 

<p><b>Benefits & Compensation.</b> If you complete the survey, we will pay you $4. In addition, you will receive a bonus of up to $4 depending on the accuracy of your predictions.</p> 

<p><b>Risks.</b> There are no known risks or discomforts associated with participating in this study.</p> 

<p>Participation in this research is completely voluntary. You can decline to participate or withdraw at any point in this study without penalty though you will not be paid.</p> 

<p><b>Confidentiality.</b> Every effort will be made to protect your confidentiality. Your personal identifying information will not be connected to the answers that you put into this survey, so we will have no way of identifying you. We will retain anonymized data for up to 5 years after the results of the study are published, to comply with American Psychological Association data-retention rules.</p> 

<p><b>Questions</b> Please contact the experimenters if you have concerns or questions: dsbowen@wharton.upenn.edu. You may also contact the office of the University of Pennsylvania’s Committee for the Protection of Human Subjects, at 215.573.2540 or via email at irb@pobox.upenn.edu.</p>
'''

# @route('/survey')
def start():
    return Branch(
        consent_page(
            consent_label,
            '<p>Please enter your MTurk ID to consent.</p>'
        ),
        demographics(
            'gender', 'age_bins', 'race', 'education', require=True, page=True
        ),
        navigate=comprehension
    )

choice_txt = 'There is a {} in 100 chance the offender will commit another crime within 2 years'

@route('/survey')
def comprehension(origin=None):
    assigner.next()
    return Branch(
        # *comprehension_check(
        #     instructions=gen_instructions_pages(),
        #     checks=gen_checks_pages(),
        #     attempts=3
        # ),
        Page(
            Label('You passed the comprehension check')
        ),
        navigate=practice
    )

def gen_instructions_pages():
    pages = [
        Page(
            Label(
            '''<p>On the next pages, we will describe profiles of criminal
            offenders. Based on an offender's profile, you will predict
            how likely he or she is to commit another crime within 2
            years.</p>
            <p>For example, you might predict that an offender has a 50
            in 100 chance of committing another crime within 2 years.</p>
            <p>Because the profiles were collected several years ago, we
            know whether the offenders did or didn't commit another crime.
            This allows us to score your predictions based on how accurate
            they were.</p>
            <p>If you say there is a 100 in 100 chance an offender will
            commit another crime, and the offender <b>did</b> commit
            another crime, you will get a perfect score.
            Similarly, if you say there is a 0 in 100 chance an 
            offender will commit another crime, and the offender <b>did
            not</b> commit another crime, you will get a perfect 
            score.</p>
            <p>You will receive a larger bonus if your predictions are
            more accurate.</p>
            ''')
        )
    ]
    if current_user.meta['Algorithm']:
        pages.append(
            Page(
                Label(
                    '''<p>We will also show you the predictions of a 
                    mathematical model designed to predict criminal 
                    behavior.</p>
                    <p>Testing shows that the model is as accurate as the 
                    average person. Additionally, rigorous testing shows that,
                    unlike other models designed to predict criminal behavior,
                    this model does not discriminate against Black offenders.
                    </p>
                    '''
                )
            )
        )
    return pages

def gen_checks_pages():
    pages = [
        Page(
            Check(
                '''For an offender who <b>did</b> commit another crime 
                within 2 years, which forecast is the most accurate?''',
                [(choice_txt.format(x), x) for x in (25, 50, 75)],
                var='CompAccuracy0', data_rows=-1,
                validate=V.require(),
                submit=S.correct_choices(75)
            ),
            Check(
                '''For an offender who <b>did not</b> commit another crime
                within 2 years, which forecast is the most accurate?''',
                [(choice_txt.format(x), x) for x in (25, 50, 75)],
                var='CompAccuracy1', data_rows=-1,
                validate=V.require(),
                submit=S.correct_choices(25)
            ),
            Binary(
                'How will your bonus be determined?',
                [
                    'I will receive a larger bonus if my predictions are more accurate',
                    'My bonus will not be determined by the accuracy of my predictions'
                ],
                inline=False, var='CompBonus', data_rows=-1,
                validate=V.require(),
                submit=S.correct_choices(1)
            ),
            compile=C.clear_response()
        )
    ]
    if current_user.meta['Algorithm']:
        pages.append(
            Page(
                Check(
                    'The predictions of the model we will show you are',
                    [
                        ('Less accurate than the average person', 'worse'),
                        ('About as accurate as the average person', 'average'),
                        ('More accurate than the average person', 'better')
                    ],
                    var='CompModelAccurate', data_rows=-1,
                    submit=S.correct_choices('average')
                ),
                Binary( 
                    'True or false: the model is biased against Black offenders',
                    ['True', 'False'],
                    var='CompModelBiased', data_rows=-1,
                    submit=S.correct_choices(0)
                ),
                compile=C.clear_response()
            )
        )
    return pages

def practice(origin=None):
    idx = choices(list(range(len(X))), k=N_PRACTICE+N_FCAST)
    X_sample, y_sample = X.iloc[idx], y.iloc[idx]
    output = model.predict_proba(X_sample)
    current_user.embedded.append(
        Embedded('Practice', [1]*N_PRACTICE+[0]*N_FCAST)
    )
    current_user.embedded += [
        Embedded(col, list(X_sample[col])) for col in X_sample.columns
    ]
    current_user.embedded.append(Embedded('output', list(output)))
    current_user.embedded.append(
        Embedded('y', list(y_sample['two_year_recid']))
    )
    return Branch(
        Page(
            Label(
                '''You will now make {} practice predictions. You will receive
                feedback after each prediction, and these predictions will 
                <i>not</i> determine your bonus.'''.format(N_PRACTICE)
            )
        ),
        *gen_practice_pages(X_sample, y_sample, output),
        Page(
            Label(
                '''You will now make {} predictions. You will not receive 
                feedback, and these predictions <i>will</i> determine your
                bonus.'''.format(N_FCAST)
            )
        ),
        *gen_fcast_pages(X_sample, y_sample, output),
        completion_page()
    )

def gen_practice_pages(X, y, output):
    pages = []
    for i in range(N_PRACTICE):
        fcast_q = gen_fcast_q(X.iloc[i], output[i])
        pages += [
            Page(
                Label('Practice prediction {} of {}'.format(i+1, N_PRACTICE)),
                Label(avg_offender_label),
                fcast_q,
                timer='FcastTimer'
            ),
            Page(
                Label(compile=C.feedback(
                    int(y.iloc[i]), round(100*output[i]), fcast_q
                ))
            )
        ]
    return pages

def gen_fcast_pages(X, y, output):
    return [
        Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            Label(avg_offender_label),
            gen_fcast_q(X.iloc[i+N_PRACTICE], output[i+N_PRACTICE]),
            timer='FcastTimer'
        )
        for i in range(N_FCAST)
    ]

def gen_fcast_q(x, output):
    fcast_q = Range(
        '<p>Consider the following offender from Broward County</p>'
        + html_list(
            'Number of prior convictions: {}'.format(x.priors_count),
            'Age: {}'.format(x.age),
            'Charge: {}'.format('Felony' if x.felony else 'Misdemeanor'),
            'Race: {}'.format('Black' if x.black else 'White'),
            'Sex: {}'.format('Male' if x.male else 'Female'),
            'Juvenile felonies: {}'.format(x.juv_fel_count),
            'Juvenile misdemeanors: {}'.format(x.juv_misd_count),
            'Other juvenile offenses: {}'.format(x.juv_other_count),
            'Marital status: {}'.format(
                'Married' if x.married else 'Unmarried'
            ),
            ordered=False
        )
        + (
            '''<p>The model predicts there is a {} in 100 chance this 
            offender will commit another crime in the next two years.</p>
            '''.format(round(100*output)) if current_user.meta['Algorithm']
            else ''
        )
        + '''
        <p>Drag the slider to enter your prediction.</p>''',
        prepend='There is a ', 
        append=' in 100 chance this offender will commit another crime in the next two years.', 
        var='Fcast'
    )
    if current_user.meta['Algorithm']:
        fcast_q.default = round(100*output)
    return fcast_q

@C.register
def feedback(feedback_label, y, output, fcast_q):
    feedback_label.label = (
        '''<p>You predicted there was a {} in 100 chance the offender would
        commit another crime in the next two years.</p>'''.format(
            fcast_q.response
        )
        + (
            '''<p>The model predicted there was a {} in 100 chance.</p>'''\
                .format(output) if current_user.meta['Algorithm']
            else ''
        )
        + '<p>The offender <b>{}</b> commit another crime within two years.</p>'\
            .format('did' if y else 'did not')
    )