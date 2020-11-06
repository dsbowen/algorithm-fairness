from .utils import (
    gen_start_branch, task_description, model_description, gen_fcast_check_q, 
    gen_bonus_check_q, gen_model_performance_check_q, gen_model_bias_check_q,
    get_sample, gen_practice_intro_page, 
    gen_fcast_intro_page, gen_profile_label, gen_fcast_question, gen_feedback_page,  
    avg_offender_label
)

from flask_login import current_user
from hemlock import Branch, Page, Label, Compile as C, route
from hemlock.tools import Assigner, comprehension_check, completion_page

import random

N_PRACTICE, N_FCAST = 1, 1

assigner = Assigner({'Algorithm': (0, 1)})

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
            checks=[
                Page(
                    gen_fcast_check_q(),
                    gen_bonus_check_q()
                ),
                Page(
                    gen_model_performance_check_q(),
                    gen_model_bias_check_q(),
                    compile=C.clear_response()
                )
            ],
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
        fcast_page = Page(
            Label('Practice prediction {} of {}'.format(i+1, N_PRACTICE)),
            Label(avg_offender_label),
            gen_profile_label(X.iloc[i]),
            fcast_q,
            timer='FcastTImer'
        )
        if current_user.meta['Algorithm']:
            fcast_page.questions.insert(
                -1, gen_model_prediction_label(i, output)
            )
        pages += [
            fcast_page,
            gen_feedback_page(i, y, output, fcast_q)
        ]
    return pages

def gen_model_prediction_label(i, output):
    return Label(
        '''
        The computer model predicts there is a {} in 100 chance the offender
        will commit another crime within 2 years.
        '''.format(round(100*output[i]))
    )

def gen_fcast_pages(X, y, output):
    def gen_fcast_page(i):
        page = Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            Label(avg_offender_label),
            gen_profile_label(X.iloc[i+N_PRACTICE]),
            gen_fcast_question(),
            timer='FcastTimer'
        )
        if current_user.meta['Algorithm']:
            page.questions.insert(
                -1, gen_model_prediction_label(i+N_PRACTICE, output)
            )
        return page

    return [gen_fcast_page(i) for i in range(N_FCAST)]