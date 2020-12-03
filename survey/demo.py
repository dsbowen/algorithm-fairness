from .utils import (
    X_test, y_test, model, gen_profile_table, explainer, 
    gen_model_prediction_label
)

import pandas as pd
from hemlock import (
    Branch, Page, Binary, Check, Input, Label, Compile as C, Validate as V, 
    route
)
from hemlock.tools import show_on_event

import random

@route('/survey')
def start():
    action_question = Check(
        '<p>What would you like to do?</p>',
        [
            ('See an example', 'example'),
            ('Assess an offender', 'assess')
        ],
        default='example'
    )
    profile_questions = [
        Input(
            '<p>How many prior convictions does this offender have?</p>',
            type='number', min=0
        ),
        Input(
            '<p>How old is this offender?</p>',
            type='number', min=0
        ),
        Binary("<p>Was this offender's most recent crime a felony?</p>"),
        Binary('<p>Is this offender Black or African American?</p>'),
        Binary('<p>This this offender male?</p>'),
        Input(
            '<p>How many juvenile felonies has this offender committed?</p>',
            type='number', min=0
        ),
        Input(
            '''
            <p>How many juvenile misdemeanors has this offender committed?</p>
            ''',
            type='number', min=0
        ),
        Input(
            '''
            <p>How many other juvenile offensees has this offender committed?
            </p>
            ''',
            type='number', min=0
        ),
        Binary('<p>Is this offender married?</p>')
    ]
    for question in profile_questions:
        show_on_event(question, action_question, 'assess')
        question.validate = V.require_if_displayed(action_question, 'assess')
    return Branch(
        Page(action_question, *profile_questions),
        Page(
            compile=C.gen_assessment(action_question, profile_questions),
            compile_worker=True,
            back=True, terminal=True
        )
    )

@V.register
def require_if_displayed(question, condition, value):
    if condition.response == value and question.response in (None, ''):
        return '<p>Please fill out this information.</p>'

@C.register
def gen_assessment(page, action_question, profile_questions):
    def get_example():
        random.seed()
        idx = random.choice(list(range(len(X_test))))
        x, y = X_test.iloc[idx], y_test.iloc[idx]
        output = model.predict_proba(x)[0]
        return x, output, y

    def get_df(profile_questions):
        x = pd.Series(
            [q.data for q in profile_questions],
            index=[
                'priors_count',
                'age',
                'felony',
                'black',
                'male',
                'juv_fel_count',
                'juv_misd_count',
                'juv_other_count',
                'married'
            ]
        )[X_test.columns]
        return x, model.predict_proba(x)[0], None

    x, output, y = (
        get_example() if action_question.response == 'example'
        else get_df(profile_questions)
    )
    page.questions = [
        Label(
            gen_profile_table(x, 'Offender profile')
        ),
        gen_model_prediction_label(
            round(100*output), explainer.explain_observations(x)[0]
        )
    ]
    if y is not None:
        page.questions.append(
            Label(
                '''
                The offender <b>{}</b> commit another crime within 2 years.
                '''.format('did' if y else 'did not')
            )
        )