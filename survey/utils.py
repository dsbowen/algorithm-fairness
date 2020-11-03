from .model import model

import numpy as np
import pandas as pd
from flask_login import current_user
from gshap.datasets import load_recidivism
from hemlock import (
    Branch, Page, Binary, Embedded, Label, Range, 
    Debug as D, Validate as V, Submit as S
)
from hemlock.tools import consent_page, html_list
from hemlock_demographics import demographics
from sklearn.model_selection import train_test_split

import pickle
import random

consent_label = '''
<p>Hello! We are researchers at the University of Pennsylvania and are interested how you look forward to predict the future. We will show profiles of criminal offenders and ask you to predict how likely they are to commit future crimes. Please read the information below and if you wish to participate, indicate your consent.</p>

<p><b>Because this is an experimental platform, you may encounter errors during this survey. If you experience an error, please email Dillon Bowen at dsbowen@wharton.upenn.edu. Copy this email address now in case you encounter an error during the survey.</b></p>

<p><b>Purpose.</b> The purpose of this study is to explore how people think about the future.</p> 

<p><b>Procedure.</b> You will be asked to complete a survey that will take approximately 20 minutes.</p> 

<p><b>Benefits & Compensation.</b> If you complete the survey, we will pay you $3. In addition, you will receive a bonus of up to $3 ($1.50 on average) depending on the accuracy of your predictions.</p> 

<p><b>Risks.</b> There are no known risks or discomforts associated with participating in this study.</p> 

<p>Participation in this research is completely voluntary. You can decline to participate or withdraw at any point in this study without penalty though you will not be paid.</p> 

<p><b>Confidentiality.</b> Every effort will be made to protect your confidentiality. Your personal identifying information will not be connected to the answers that you put into this survey, so we will have no way of identifying you. We will retain anonymized data for up to 5 years after the results of the study are published, to comply with American Psychological Association data-retention rules.</p> 

<p><b>Questions</b> Please contact the experimenters if you have concerns or questions: dsbowen@wharton.upenn.edu. You may also contact the office of the University of Pennsylvania’s Committee for the Protection of Human Subjects, at 215.573.2540 or via email at irb@pobox.upenn.edu.</p>
'''

def gen_start_branch(navigate):
    return Branch(
        consent_page(
            consent_label,
            '<p>Please enter your MTurk ID to consent.</p>'
        ),
        demographics(
            'gender', 'age_bins', 'race', 'education', 'household_residents', 'children', 'live_with_parents', 'marital_status', 'employment', 'occupation', 'sector', 'primary_wage_earner', 'save_money', 'social_class', 'income_group',
            require=True, page=True
        ),
        navigate=navigate
    )

task_description = '''
<p>Please read these instructions carefully. We will test your understanding on the next pages.</p>

<p>In this study, we will describe profiles of criminal offenders. Based on an offender's profile, you will predict how likely he or she is to commit another crime within 2 years.</p>

<p>For example, you might predict that an offender has a 50 in 100 chance of committing another crime within 2 years.</p>

<p>Because the profiles were collected several years ago, we know whether the offenders did or didn't commit another crime. This allows us to score your predictions based on how accurate they were.</p>

<p>You will receive a larger bonus if your predictions are
more accurate.</p>
'''

task_check_txt = '''
<p>Imagine you've just read the profile of a criminal offender. Drag the slider to predict that there is a {} in 100 chance the offender will commit another crime within 2 years.</p>
'''

def gen_fcast_check_q():
    return Range(
        prepend='There is a ',
        append=' in 100 chance the offender will commit another crime within 2 years',
        var='FcastComprehension', data_rows=-1,
        compile=random_fcast_check
    )

def random_fcast_check(fcast_check_q):
    x = round(100*random.random())
    fcast_check_q.label = task_check_txt.format(x)
    fcast_check_q.submit = S.match(x)
    fcast_check_q.debug = D.drag_range(x, p_exec=.5)

def gen_bonus_check_q():
    return Binary(
        '<p>True or False: You will receive a larger bonus if your predictons are more accurate.</p>',
        ['True', 'False'],
        var='BonusComprehension', data_rows=-1,
        validate=V.require(),
        submit=S.correct_choices(1),
        debug=D.click_choices(1, p_exec=.5)
    )

# summary from training data
sum_df = pd.read_csv('survey/summary.csv', index_col=0)
# X and y from test data
# use same random seed for the train test split 
# to ensure the study uses test data
random.seed(0)
np.random.seed(0)
recidivism = load_recidivism()
X, y = recidivism.data, recidivism.target
X = X.drop(columns='high_supervision')
X_train, X_test, y_train, y_test = train_test_split(X, y)

def get_sample(n_practice, n_fcast):
    idx = random.choices(list(range(len(X_test))), k=n_practice+n_fcast)
    X_sample, y_sample = X_test.iloc[idx], y_test.iloc[idx]
    output = model.predict_proba(X_sample)
    current_user.embedded.append(
        Embedded('Practice', [1]*n_practice+[0]*n_fcast)
    )
    current_user.embedded += [
        Embedded(col, list(X_sample[col])) for col in X_sample.columns
    ]
    current_user.embedded.append(Embedded('output', list(output)))
    current_user.embedded.append(
        Embedded('y', list(y_sample))
    )
    return X_sample, y_sample, output

# describes the average offender from the training data
avg_offender_label = (
    '<p>Here is some information about the criminal population in Broward County, Florida</p>'
    + html_list(
        '{} of every 100 offenders commit another crime within 2 years.'.format(round(100*sum_df.two_year_recid['mean'])),
        'The average offender has {} prior convictions'.format(round(sum_df.priors_count['mean'])),
        'The average offender is {} years old'.format(round(sum_df.age['mean'])),
        '{} of every 100 offenders have committed a felony'.format(round(100*sum_df.felony['mean'])),
        '{} of every 100 offenders are Black'.format(round(100*sum_df.black['mean'])),
        '{} of every 100 offenders are male'.format(round(100*sum_df.male['mean'])),
        'The average offender has {} juvenile felonies, {} juvenial misdemeanors, and {} other juvenile offenses'.format(
            round(sum_df.juv_fel_count['mean']), 
            round(sum_df.juv_misd_count['mean']), 
            round(sum_df.juv_other_count['mean'])
        ),
        '{} of every 100 offenders are married'.format(round(100*sum_df.married['mean'])),
        ordered=False
    )
)

def gen_profile_label(x):
    return Label(
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
    )

def gen_fcast_question():
    return Range(
        '<p>Drag the slider to enter your prediction.</p>',
        prepend='There is a ', 
        append=' in 100 chance this offender will commit another crime within 2 years.',
        var='Fcast'
    )

def gen_practice_intro_page(n_practice):
    return Page(
        Label('''
        You will now make {} practice predictions. You will receive feedback after each prediction, and these predictions will <i>not</i> determine your bonus.'''.format(n_practice)
        )
    )

def gen_fcast_intro_page(n_fcast):
    return Page(
        Label('''
        You will now make {} predictions. You will not receive feedback, and these predictions <i>will</i> determine your bonus.
        '''.format(n_fcast))
    )