from .utils import (
    explainer, gen_start_branch, gen_comprehension_branch, get_sample,
    split_iterables, gen_practice_intro_page, gen_fcast_question, gen_profile_label, gen_model_prediction_label, gen_feedback_page
)

from flask_login import current_user
from hemlock import Participant, Branch, Page, Blank, Input, Label, Compile as C, Debug as D, Validate as V, Navigate as N, route
from hemlock.tools import Assigner, completion_page

import random

N_SELF, N_TRIAL, N_FCAST = 3, 3, 10

assigner = Assigner({'Explanation': (1, 0), 'Adopt': (1, 0)})

@route('/survey')
def start():
    return gen_start_branch(
        compensation='''
            We will pay you $2 to complete this survey. Additionally, we will randomly select 1 in 10 participants to receive a bonus of up to $30 ($15 average). Your bonus will depend on the accuracy of your predictions.
            ''',
        navigate=comprehension
    )

# @route('/survey')
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
        gen_practice_intro_page(N_SELF, N_TRIAL, N_FCAST, delay_forward=500),
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

def auction(origin):
    random_bid = '{:.2f}'.format(30*random.random())
    bid_q = Blank(
        ('''
        <p>From previous studies, we estimate that most participants' bonuses will be $0.10 to $1.70 larger if they have the model to assist them.</p>

        <p>How much are you willing to pay (bid) to continue using the model?</p>
        <ul>
            <li><b>I am willing to pay up to $''', ''' to continue using the model</b></li>
            <li><b>I am unwilling to pay more than $''', ''' to continue using the model</b></li>
            <li>This is roughly equivalent to, <b>I expect my bonus to be $''', ''' larger if I continue using the model</b></li>
        </ul>
        '''),
        blank_empty='_____', prepend='$', 
        type='number', min=0, max=30, step=.01,
        debug=D.send_keys(random_bid),
        validate=[V.require(), V.exact_decimals(2)]
    )
    return Branch(
        Page(
            Label(
                '''
                <p>You will now make {} predictions. You will not receive feedback, and these predictions <i>will</i> determine your bonus.</p>

                <p>Additionally, your free trial using the computer model has expired. To continue using the model, we will pair you with another participant to bid for it.</p>

                <p>We will enter both of your bids in a 'second-price auction'. If you outbid the other participant, you will get to keep using the model, but we will deduct the other participant's bid from your bonus. (We resolve ties randomly.) <b>The best strategy is to bid the exact amount you're willing to pay.</b> Trying to 'game the system' by bidding more or less than you're willing to pay will make you worse off.</p>

                <p><b>FAQ:</b> Why should I bid exactly what I'm willing to pay? Isn't it usually a good idea to underbid so I pay less?</p>

                <p><b>Answer: No!</b> You're thinking of a 'first-price' auction, not a second-price auction. The difference is that, if you win, you'll pay the <i>other</i> person's bid, not your own. Economists have <a href="https://en.wikipedia.org/wiki/Vickrey_auction#Proof_of_dominance_of_truthful_bidding" target="_blank">mathematically proven</a> that the best thing you can do in a second-price auction is bid exactly as much as you're willing to pay.</p>

                <p><b>Keep reading if you're not convinced. Otherwise, skip to the bottom of the page to enter your bid.</b></p>

                <p>Imagine you're bidding for a house you plan to flip for $100,000. (This means you're willing to pay $100,000 for it). You're wondering whether to bid $80,000 or $100,000. Consider these cases.</p>

                <ol>
                    <li>The other person bids $70,000. You get the house for $70,000 whether you bid $80,000 or $100,000.</li>
                    <li>The other person bids $110,000. You lose the auction whether you bid $80,000 or $100,000.</li>
                    <li>The other person bids $90,000. If you bid $80,000, you lose the auction. If you bid $100,000, you get the house for $90,000 and flip it for a $10,000 profit.</li>
                </ol>

                <p>In general, underbidding (e.g., bidding $80,000 when you're willing to pay $100,000) never makes you better off and sometimes makes you worse off. So you might as well bid the exact amount you're willing to pay.</p>

                <p>The same logic applies to overbidding. Imagine you bid $120,000. The other person bids $110,000. You get the house for $110,000 but flip it for $100,000 at a loss. In general, overbidding (e.g., bidding $120,000 when you're willing to pay at most $100,000) never makes you better off and sometimes makes you worse off. So you might as well bid the exact amount you're willing to pay.</p>
                '''.format(N_FCAST)    
            ),
            bid_q,
            Input(
                '<p>Please re-enter your bid.</p>',
                prepend='$', type='number', min=0, max=30, step=.01,
                debug=D.send_keys(random_bid),
                validate=V.match(
                    bid_q, 
                    error_msg='<p>Please make sure your bids match.</p>'
                )
            ),
            delay_forward=500
        ),
        Page(
            Label(compile=C(auction_result, bid_q))
        ),
        navigate=forecast
    )

def auction_result(result_label, bid_q):
    def get_other_bid():
        parts = Participant.query.limit(100).all()
        parts = [part for part in parts if part.meta.get('WTP') is not None]
        if not parts:
            return 0 if adopt else 10
        parts.sort(key=lambda part: part.meta['WTP'])
        return (
            parts[0].meta['WTP'] if adopt else parts[-1].meta['WTP']
        )

    adopt = current_user.meta['Adopt']
    bid = float(bid_q.data)
    other_bid = get_other_bid()
    win = bid >= other_bid if adopt else bid > other_bid
    result_label.label = '''
        You bid ${:.2f}. The other participant bid ${:.2f}.
        '''.format(bid, other_bid)
    result_label.label += (
        'You won the auction.' if win else 'You lost the auction.'
    )
    current_user.meta.update({
        'WTP': bid, 'OtherBid': other_bid, 'WonAuction': int(win)
    })

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