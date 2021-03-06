from .utils import (
    gen_start_branch, gen_comprehension_branch, get_sample, gen_practice_intro_page, 
    gen_fcast_intro_page, gen_profile_label, gen_fcast_question, 
    gen_most_important_feature_select
)

from hemlock import Branch, Page, Label, Compile as C, route
from hemlock.tools import comprehension_check, completion_page

import random

N_PRACTICE, N_FCAST = 5, 10

@route('/survey')
def start():
    return gen_start_branch(
        compensation='''
            If you complete the survey, we will pay you $2. Additionally, you will receive a bonus of up to $3 ($1.50 on average) depending on the accuracy of your predictions.
            ''',
        navigate=comprehension
    )

# @route('/survey')
def comprehension(origin=None):
    return gen_comprehension_branch(
        additional_instr='''
        <p>We will pay you a larger bonus if your predictions are more accurate.</p>
        ''',
        navigate=forecast
    )

# @route('/survey')
def forecast(origin=None):
    X, y, output = get_sample(N_PRACTICE, N_FCAST)
    return Branch(
        gen_practice_intro_page(N_PRACTICE),
        *gen_practice_pages(X, y, output),
        gen_fcast_intro_page(N_FCAST),
        *gen_fcast_pages(X, y, output),
        completion_page()
    )

def gen_practice_pages(X, y, output):
    pages = []
    for i in range(N_PRACTICE):
        fcast_q = gen_fcast_question()
        pages += [
            Page(
                Label('Practice prediction {} of {}'.format(i+1, N_PRACTICE)),
                gen_profile_label(X.iloc[i]),
                fcast_q,
                gen_most_important_feature_select(),
                timer='FcastTimer'
            ),
            Page(
                Label(compile=C.feedback(
                    int(y.iloc[i]), round(100*output[i]), fcast_q
                ))
            )
        ]
    return pages

@C.register
def feedback(feedback_label, y, output, fcast_q):
    feedback_label.label = '''
        <p>You predicted there was a {} in 100 chance the offender would
        commit another crime within 2 years.</p>
        <p>The offender <b>{}</b> commit another crime within 2 years.</p>
        '''.format(fcast_q.response, 'did' if y else 'did not')

def gen_fcast_pages(X, y, output):
    return [
        Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            gen_profile_label(X.iloc[i+N_PRACTICE]),
            gen_fcast_question(),
            gen_most_important_feature_select(),
            timer='FcastTimer'
        )
        for i in range(N_FCAST)
    ]