from .utils import (
    explainer, gen_start_branch, task_description, gen_fcast_check_q, 
    gen_bonus_check_q, gen_model_performance_check_q, gen_model_bias_check_q,
    get_sample, gen_practice_intro_page, gen_fcast_intro_page,
    gen_fcast_question, gen_profile_label, gen_feedback_page,
    gen_model_prediction_label
)

from flask_login import current_user
from hemlock import Branch, Page, Binary, Blank, Label, Compile as C, Validate as V, route
from hemlock.tools import Assigner, comprehension_check, completion_page, html_list, show_on_event

import random

ADJUST_AMOUNT = 5
N_SELF, N_TRIAL, N_FCAST = 1, 1, 1
N_PRACTICE = N_SELF + N_TRIAL

assigner = Assigner({'Explanation': (1, 1), 'Adjust': (1, 1)})

# @route('/survey')
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
                gen_bonus_check_q(),
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
        explanations = explainer.explain_observations(X, output/100)
    g = current_user.g
    g['X_self'], g['X_trial'], g['X_fcast'] = (
        X.iloc[:N_SELF], 
        X.iloc[N_SELF:N_PRACTICE], 
        X.iloc[N_PRACTICE:]
    )
    g['y_self'], g['y_trial'], g['y_fcast'] = (
        y.iloc[:N_SELF], 
        y.iloc[N_SELF:N_PRACTICE], 
        y.iloc[N_PRACTICE:]
    )
    g['output_self'], g['output_trial'], g['output_fcast'] = (
        output[:N_SELF], 
        output[N_SELF:N_PRACTICE], 
        output[N_PRACTICE:]
    )
    g['exp_self'], g['exp_trial'], g['exp_fcast'] = (
        explanations[:N_SELF], 
        explanations[N_SELF:N_PRACTICE],
        explanations[N_PRACTICE:]
    )
    return Branch(
        Page(
            Label(
                '''
                <p>You will now make {n_practice} practice predictions to familiarize yourself with the task. You will receive feedback after each prediction, and these predictions will <i>not</i> determine your bonus.</p>
                
                <p>You will make the first {n_self} practice predictions on your own. For the last {n_trial} practice predictions, we will show the computer model's prediction. Think of these last {n_trial} practice predictions as a 'free trial' to assess how helpful you find the computer model.</p>

                <p>After the practice predictions, you will make {n_fcast} 'real' predictions. You will not receive feedback, and these predictions <i>will</i> determine your bonus. Additionally, your free trial will expire, meaning you will have to decide how much of your bonus you're willing to pay to continue using the computer model.</p>
                '''.format(
                    n_practice=N_SELF + N_TRIAL,
                    n_self=N_SELF,
                    n_trial=N_TRIAL,
                    n_fcast=N_FCAST
                ) + (
                    '''
                    <p>Finally, when using the computer model, your prediction must be within {} points of the model's. For example, if the model predicts there is a 50 in 100 chance an offender will commit another crime within 2 years, your prediction must be between {} in 100 and {} in 100.</p>
                    '''.format(
                        ADJUST_AMOUNT, 50-ADJUST_AMOUNT, 50+ADJUST_AMOUNT
                    ) if current_user.meta['Adjust'] else ''
                )
            ),
            delay_forward=500
        ),
        *gen_practice_pages(
            g['X_self'], g['y_self'], g['output_self'], g['exp_self']
        ),
        Page(
            Label(
                '''
                Your 'free trial' with the computer model will begin on the next page.
                '''
            )
        ),
        *gen_practice_pages(
            g['X_trial'], g['y_trial'], g['output_trial'], g['exp_self'],
            trial=True
        ),
        navigate=forecast_intro
    )

def gen_practice_pages(X, y, output, explanations, trial=False):
    def gen_fcast_page(i, x, y, output, explanation):
        prediction_number = i+1+N_SELF if trial else i+1
        fcast_q = gen_fcast_question()
        fcast_page = Page(
            Label('Practice prediction {} of {}'.format(
                prediction_number, N_SELF+N_TRIAL
            )),
            gen_profile_label(x),
            fcast_q,
            timer='FcastTimer'
        )
        if trial:
            fcast_page.questions.insert(
                -1, gen_model_prediction_label(output, explanation)
            )
            if current_user.meta['Adjust']:
                fcast_q.min = output - ADJUST_AMOUNT
                fcast_q.max = output + ADJUST_AMOUNT
        return fcast_q, fcast_page

    pages = []
    for i in range(len(X)):
        fcast_q, fcast_page = gen_fcast_page(
            i, X.iloc[i], y.iloc[i], output[i], explanations[i] 
        )
        pages += [
            fcast_page, 
            gen_feedback_page(i, y, output, fcast_q, disp_output=trial)
        ]
    return pages

def forecast_intro(origin=None):
    convinced = Binary(
        '''
        <p><b>FAQ</b>. Should I bid less for the model than I'm actually willing to pay for it?</p>

        <p><b>Answer: NO!</b> We determine whether to accept your bid using an <i>incentive-compatible mechanism.</i> This is a fancy term economists use which means that you're best off when you bid exactly as much as the thing is worth to you. There's no way to game the system. Attempting to do so will make you worse off.</p>
        ''',
        [
            "I'm convinced. I'm going to bid exactly as much as the model is worth to me.",
            "I'm not convinced. Please explain more."
        ],
        var='Convinced', data_rows=-1, inline=False,
        validate=V.require()
    )
    explain = Binary(
        '''
        <p>We determine whether to accept your bid using the <i>Becker-DeGroot-Marschak mechanism</i> or <i>Vickrey auction</i>. Developed by a Nobel Prize-winning economist, it's mathematically proven to be impossible to game. It's not important to understand how it works. What <i>is</i> important is to understand that you're best off when you bid exactly as much as the model is worth to you.</p>

        <p>Still not convinced? <a href="https://en.wikipedia.org/wiki/Vickrey_auction#Proof_of_dominance_of_truthful_bidding" target="_blank">Read this</a> mathematical proof of why you should bid exactly as much as the model is worth to you.</p>
        ''',
        [
            "I'm convinced. I'm going to bid exactly as much as the model is worth to me.",
            "I'm still not convinced."
        ],
        var='Convinced1', data_rows=-1, inline=False,
    )
    show_on_event(explain, convinced, 0)
    bid = Blank(
        ('''
        <p>Recall that there is a 1 in 10 chance we will select you to receive a bonus. From previous studies, we estimate that <b>the average participant will earn a $1 larger bonus by continuing to use the model</b>.</p>

        <p>How much are you willing to bid (pay) to continue using the model?</p>

        <ul>
            <li><b>I am willing to pay up to ''', ''' cents to continue using the model</b></li>
            <li><b>I am unwilling to pay more than ''', ''' cents to continue using the model</b></li>
        </ul>
        '''),
        var='WTP', data_rows=-1, 
        blank_empty='_____', append='cents', type='number', min=0,
        validate=V.require()
    )
    return Branch(
        gen_fcast_intro_page(N_FCAST),
        Page(
            Label(
                '''
                <p>You will now make {} predictions. You will not receive feedback, and these predictions <i>will</i> determine your bonus.</p>

                <p>Additionally, your free trial using the computer model has expired. To continue using the model, you must place a bid for it.</p>

                <p>If we accept your bid, we'll reduce your bonus and you'll continue using the model. If we reject your bid, we won't reduce your bonus, but you won't get to use the model to help you make predictions. The larger your bid, the more likely we are to accept it.</p>
                '''.format(N_FCAST)
            ),
            convinced, 
            explain,
            bid,
            delay_forward=500
        ),
        Page(
            Label(compile=C.Vickrey(bid))
        ),
        navigate=forecast
    )

@C.register
def Vickrey(auction_label, bid_q):
    price = random.random()
    won_bid = price < float(bid_q.data) / 100.
    current_user.meta.update({'Price': price, 'WonBid': int(won_bid)})
    auction_label.label = (
        'Your bid to continue using the model was successful.'  if won_bid
        else 'Your bid to continue using the model was unsuccessful.'
    )

def forecast(origin=None):
    X, y, output, explanations = (
        current_user.g[key] for key in (
            'X_fcast', 'y_fcast', 'output_fcast', 'exp_fcast'
        )
    )
    return Branch(
        *gen_fcast_pages(X, y, output, explanations),
        completion_page()
    )

def gen_fcast_pages(X, y, output, explanations):
    def gen_fcast_page(i, x, output, explanation):
        fcast_q = gen_fcast_question()
        page = Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            gen_profile_label(x),
            fcast_q,
            timer='FcastTimer'
        )
        if current_user.meta['WonBid']:
            page.questions.insert(
                -1, gen_model_prediction_label(output, explanation),
            )
            if current_user.meta['Adjust']:
                fcast_q.min = output - ADJUST_AMOUNT
                fcast_q.max = output + ADJUST_AMOUNT
        return page

    return [
        gen_fcast_page(i, X.iloc[i], output[i], explanations[i])
        for i in range(N_FCAST)
    ]