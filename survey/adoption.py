from .utils import (
    explainer, gen_start_branch, gen_comprehension_branch, get_sample,
    split_iterables, gen_practice_intro_page, gen_fcast_question, 
    gen_profile_label, gen_model_prediction_label, 
    gen_most_important_feature_select, gen_feedback_page
)

import numpy as np
from flask_login import current_user
from hemlock import (
    Participant, Branch, Page, Binary, Blank, Input, Label, 
    Compile as C, Debug as D, Validate as V, Navigate as N, route
)
from hemlock.tools import Assigner, completion_page, titrate
from hemlock_berlin import berlin
from hemlock_crt import crt
from scipy.stats import expon

import random

N_SELF, N_TRIAL, N_FCAST = 1, 1, 1

assigner = Assigner({'Explanation': (1, 0)})

# @route('/survey')
def start():
    return gen_start_branch(
        compensation='''
            We will pay you $2 to complete this survey. Additionally, we will randomly select 1 in 10 participants to receive a bonus of up to $30 ($15 average). Your bonus will depend on the accuracy of your predictions.
            ''',
        navigate=comprehension
    )

@route('/survey')
def comprehension(origin=None):
    assigner.next()
    return gen_comprehension_branch(
        additional_instr='''
            <p>We will randomly select 1 in 10 participants to receive a bonus. If we select you, we will pay you a larger bonus if your predictions are more accurate.</p> 
            ''',
        navigate=practice,
        navigate_worker=True
    )

# @route('/survey')
def practice(origin=None):
    part = current_user or origin.part
    sample = get_sample(
        part,
        n_practice=N_SELF+N_TRIAL, 
        n_fcast=N_FCAST, 
        explanation=part.meta['Explanation']
    )
    X, y, output, explanations = split_iterables(
        sample, N_SELF, N_TRIAL, N_FCAST
    )
    part.g.update(dict(
        X=X[2], y=y[2], output=output[2], explanations=explanations[2]
    ))
    return Branch(
        gen_practice_intro_page(N_SELF, N_TRIAL, N_FCAST),
        *gen_practice_pages(X[0], y[0], output[0], explanations[0]),
        Page(
            Label(
                '''
                Your free trial with the computer model will begin on the next page.
                '''
            )
        ),
        *gen_practice_pages(
            X[1], y[1], output[1], explanations[1], trial=True
        ),
        navigate=auction
    )

def gen_practice_pages(X, y, output, explanations, trial=False):
    def gen_fcast_page(i, x, y, output, explanation):
        prediction_number = i+1+N_SELF if trial else i+1
        fcast_q = gen_fcast_question(output if trial else None)
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

# @route('/survey')
def auction(origin=None):
    return Branch(
        Page(
            Label(
                '''
                <p>You'll now make {} predictions. You won't receive feedback, and these predictions <i>will</i> determine your bonus.</p>

                <p>Additionally, your free trial with the computer model has expired. To continue receiving the model's advice, we'll pair you with another participant to bid for it.</p>

                <p>The auction works exactly like eBay. First, you'll tell us the highest price you're willing to pay for the computer model's advice. Then we'll bid in increments to keep you in the lead but only up to your limit.</p>
                
                <p>For example, if the other participant is willing to pay up to $1.00 and you're willing to pay up to $1.50, we'll bid $1.01 for you, and you'll win the auction. If the other participant is willing to pay up to $1.50 and you're willing to pay up to $1.00, we'll bid $1.01 for the other participant, and you'll lose the auction. We'll resolve ties randomly.</p>

                <p>If you win, you'll pay out of your bonus and continue receiving the model's advice, like you did in the final practice rounds. If you lose, you won't pay anything from your bonus but you'll have to make your predictions on your own, like you did in the first practice rounds.</p>

                We'll figure out how much you're willing to pay on the next pages.
                '''.format(N_FCAST)    
            ),
            Label(
                '''The forward button will appear in 30 seconds. Please take this time to read the instructions carefully.'''
            ),
            delay_forward=500
        ),
        titrate(
            gen_bid_q,
            expon(0, 2), 
            tol=.01,
            var='Bid',
            back=True
        ),
        Page(
            Label(compile=auction_result)
        ),
        navigate=forecast
    )

def gen_bid_q(other_bid):
    return Binary(
        '''
        <p>Would you outbid the other participant if he/she bid <b>${:.2f}</b> to purchase the model's advice?</p>
        '''.format(other_bid)
    )

def auction_result(result_label):
    def get_other_bid():
        parts = Participant.query.all()
        bids = np.array([
            part.g['Bid'] for part in parts 
            if (
                part != current_user
                and isinstance(part.g, dict) and 'Bid' in part.g
            )
        ])
        return np.quantile(bids, .5) if bids.size != 0 else 1

    bid = current_user.g['Bid']
    other_bid = get_other_bid()
    win = bid >= other_bid
    result_label.label = (
        '''
        <p>You were willing to pay up to ${:.2f}. The other participant was willing to pay up to ${:.2f}.</p>
        '''.format(bid, other_bid)
    ) + (
        '''
        This means you won the auction. <b>You'll pay ${:.2f} of your bonus to keep receiving the model's advice for the rest of your predictions.</b>
        '''.format(other_bid) if win
        else '''This means you lost the auction. <b>You'll make the rest of your predictions without the model's advice.</b>'''
    )
    current_user.meta.update({'OtherBid': other_bid, 'WonAuction': int(win)})

def forecast(origin):
    return Branch(
        *gen_fcast_pages(),
        completion_page()
    )

def gen_fcast_pages():
    def gen_fcast_page(i, x, y, output, explanation):
        page = Page(
            Label('Prediction {} of {}'.format(i+1, N_FCAST)),
            gen_profile_label(x),
            gen_fcast_question(),
            timer='FcastTimer'
        )
        if current_user.meta['WonAuction']:
            page.questions[-1].default = float(output)
            page.questions.insert(
                -1, gen_model_prediction_label(output, explanation)
            )
        return page

    X, y, output, explanations = (
        current_user.g[key] for key in ('X', 'y', 'output', 'explanations')
    )
    return [
        gen_fcast_page(i, X.iloc[i], y.iloc[i], output[i], explanations[i])
        for i in range(N_FCAST)
    ]