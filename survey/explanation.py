from .utils import (
    explainer, gen_start_branch, task_description, model_description, gen_fcast_check_q, 
    gen_bonus_check_q, gen_model_performance_check_q, gen_model_bias_check_q,
    get_sample, gen_practice_intro_page, gen_fcast_intro_page,
    gen_fcast_question, gen_profile_label, gen_feedback_page,
    gen_model_prediction_label
)

from flask_login import current_user
from hemlock import Branch, Page, Binary, Label, Compile as C, Validate as V, route
from hemlock.tools import Assigner, comprehension_check, completion_page, html_list

import random

N_PRACTICE, N_FCAST = 1, 1

assigner = Assigner({'Explanation': (1, 1), 'Adjust': (1, 1)})

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
        navigate=practice,
        navigate_worker=True
    )

@route('/survey')
def practice(origin=None):
    assigner.next()
    X, y, output = get_sample(N_PRACTICE, N_FCAST)
    explanations = [''] * len(X)
    if current_user.meta['Explanation']:
        explanations = explainer.explain_observations(X, output)
    g = current_user.g
    g['X'], g['y'], g['output'], g['explanations'] = (
        X, y, output, explanations
    )
    return Branch(
        gen_practice_intro_page(N_PRACTICE),
        *gen_practice_pages(X, y, output, explanations),
        navigate=forecast_intro
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
                fcast_q,
                timer='FcastTimer'
            ),
            gen_feedback_page(i, y, output, fcast_q, disp_output=True)
        ]
    return pages

def forecast_intro(origin=None):
    branch = Branch(
        gen_fcast_intro_page(N_FCAST),
        navigate=forecast
    )
    if current_user.meta['Adjust']:
        branch.pages.append(Page(
            Binary(
                '''
                <p>Before you make your predictions, please indicate whether to use your predictions or the computer model's predictions to determine your bonus.</p>
                '''
                + html_list(
                    '''
                    <b>If you choose to use your predictions, we will not show you the model's predictions.</b>
                    ''',
                    '''
                    <b>If you choose to use the model's predictions, you can adjust the model's predictions by up to 5 points.</b> For example, if the model predicts there is a 50 in 100 chance the offender will commit another crime within 2 years, your prediction must be between a 45 in 100 and a 55 in 100 chance.
                    '''
                )
                + '''
                <p>You will make predictions no matter which option you choose.</p>
                ''',
                [
                    'Use my estimates to determine my bonus.',
                    "Use the model's predictions to determine my bonus."
                ],
                inline=False, var='MyPredictions', data_rows=-1,
                validate=V.require(),
                submit=record_meta,
            ),
        ))
    return branch

def record_meta(adjust_question):
    current_user.meta['MyPredictions'] = adjust_question.data 

def forecast(origin=None):
    X, y, output, explanations = (
        current_user.g[key] for key in ('X', 'y', 'output', 'explanations')
    )
    return Branch(
        *gen_fcast_pages(X, y, output, explanations),
        completion_page()
    )

def gen_fcast_pages(X, y, output, explanations):
    def gen_fcast_page(i, x, output, explanation):
        page = Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            gen_profile_label(x),
            gen_model_prediction_label(output, explanation),
            gen_fcast_question(),
            timer='FcastTimer'
        )
        if current_user.meta['Adjust']:
            if current_user.meta['MyPredictions']:
                page.questions.pop(-2)
            else:
                output = round(100*output)
                fcast_q = page.questions[-1]
                fcast_q.min, fcast_q.max = output - 5, output + 5
        return page

    return [
        gen_fcast_page(
            i,
            X.iloc[i+N_PRACTICE], 
            output[i+N_PRACTICE], 
            explanations[i+N_PRACTICE]
        )
        for i in range(N_FCAST)
    ]