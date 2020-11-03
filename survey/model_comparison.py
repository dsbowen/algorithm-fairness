from .utils import (
    gen_start_branch, task_description, gen_fcast_check_q, 
    gen_bonus_check_q, get_sample, gen_practice_intro_page, 
    gen_fcast_intro_page, gen_profile_label, gen_fcast_question, 
    avg_offender_label
)

from hemlock import Branch, Page, Label, Compile as C, route
from hemlock.tools import comprehension_check, completion_page

import random

# test deploy
# commit and tag
# submit prereg
# deploy and launch
# set N_PRACTICE and N_FCAST

N_PRACTICE, N_FCAST = 1, 10

@route('/survey')
def start():
    return gen_start_branch(comprehension)

# @route('/survey')
def comprehension(origin=None):
    return Branch(
        *comprehension_check(
            instructions=Page(
                Label(task_description)
            ),
            checks=Page(
                gen_fcast_check_q(),
                gen_bonus_check_q()
            ),
            attempts=3
        ),
        Page(
            Label('You passed the comprehension check.')
        ),
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
                Label(avg_offender_label),
                gen_profile_label(X.iloc[i]),
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
            Label(avg_offender_label),
            gen_profile_label(X.iloc[i+N_PRACTICE]),
            gen_fcast_question(),
            timer='FcastTimer'
        )
        for i in range(N_FCAST)
    ]