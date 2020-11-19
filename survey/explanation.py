from .utils import (
    explainer, gen_start_branch, task_description, model_description, gen_fcast_check_q, 
    gen_bonus_check_q, gen_model_performance_check_q, gen_model_bias_check_q,
    get_sample, gen_practice_intro_page, 
    gen_fcast_intro_page, gen_profile_label, gen_fcast_question, gen_feedback_page,
    gen_most_important_feature_select
)

from flask_login import current_user
from hemlock import Branch, Page, Label, Compile as C, route
from hemlock.tools import Assigner, comprehension_check, completion_page

import random

N_PRACTICE, N_FCAST = 10, 1

assigner = Assigner({'Explanation': (1, 1)})

# @route('/survey')
def start():
    return gen_start_branch(comprehension)

# @route('/survey')
def comprehension(origin=None):
    return Branch(
        *comprehension_check(
            instructions=Page(
                Label(task_description),
                Label(model_description)
            ),
            checks=Page(
                gen_fcast_check_q(),
                gen_bonus_check_q(),
                gen_model_performance_check_q(),
                compile=C.clear_response()
            ),
            attempts=3
        ),
        Page(
            Label('You passed the comprehension check.')
        ),
        navigate=forecast
    )

@route('/survey')
def forecast(origin=None):
    assigner.next()
    X, y, output = get_sample(N_PRACTICE, N_FCAST)
    explanations = [''] * len(X)
    if current_user.meta['Explanation']:
        explanations = explainer.explain_observations(X, output)
    return Branch(
        gen_practice_intro_page(N_PRACTICE),
        *gen_practice_pages(X, y, output, explanations),
        gen_fcast_intro_page(N_FCAST),
        *gen_fcast_pages(X, y, output, explanations),
        completion_page()
    )

def gen_practice_pages(X, y, output, explanations):
    pages = []
    for i in range(N_PRACTICE):
        fcast_q = gen_fcast_question()
        pages += [
            Page(
                Label('Practice prediction {} of {}'.format(i+1, N_PRACTICE)),
                gen_profile_label(X.iloc[i]),
                gen_model_prediction_label(output[i], explanations[i]),
                gen_most_important_feature_select(),
                fcast_q,
                timer='FcastTimer'
            ),
            gen_feedback_page(i, y, output, fcast_q, disp_output=True)
        ]
    return pages

def gen_fcast_pages(X, y, output, explanations):
    return [
        Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            gen_profile_label(X.iloc[i+N_PRACTICE]),
            gen_model_prediction_label(
                output[i+N_PRACTICE], explanations[i+N_PRACTICE]
            ),
            gen_most_important_feature_select(),
            gen_fcast_question(),
            timer='FcastTimer'
        )
        for i in range(N_FCAST)
    ]

def gen_model_prediction_label(output, explanation=''):
    return Label(
        '''
        <p>The computer model predicts there is a <b>{} in 100</b> chance the offender
        will commit another crime within 2 years.</p>
        '''.format(round(100*output))
        + (
            '''
            <hr>
            <p><b>Here's what the computer model based its prediction 
            on.</b></p>
            ''' if explanation else ''
        )
        + explanation
    )