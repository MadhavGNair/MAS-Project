"""
This is the code that is part of Tutorial 1 for the ANL 2024 competition, see URL.

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

from negmas import (
    make_issue,
    SAOMechanism,
)

from negmas.preferences import LinearAdditiveUtilityFunction as UFun
from negmas.preferences import UtilityFunction #b
import matplotlib.pyplot as plt
import negmas.inout as io#
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
import pathlib
from group1 import Group1

def assignA():
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="Location", values=['Antalya', 'Barcelona', 'Milan']),
        make_issue(name="Duration", values=['1 week', '2 weeks']),
        make_issue(name='Hotel Quality', values=['Hostel', '3star hotel', '5 star hotel']),
    ]

    # create the mechanism
    session = SAOMechanism(issues=issues, n_steps=30)

    A_utility = UFun(
        {'Location': {'Antalya': 4, 'Barcelona': 10, 'Milan': 2},
        'Duration': {'1 week': 3, '2 weeks': 10},
        'Hotel Quality': {'Hostel': 10, '3star hotel': 2, '5 star hotel': 3}},
        weights={'Location': 0.5, 'Duration': 0.2, 'Hotel Quality': 0.3},
        issues=issues
    )

    B_utility = UFun(
        {'Location': {'Antalya': 3, 'Barcelona': 2, 'Milan': 10},
        'Duration': {'1 week': 4, '2 weeks': 10},
        'Hotel Quality': {'Hostel': 3, '3star hotel': 3, '5 star hotel': 10}},
        weights={'Location': 0.5, 'Duration': 0.4, 'Hotel Quality': 0.1},
        issues=issues
    )
    A_utility = A_utility.normalize()
    B_utility = B_utility.normalize()

    return session, A_utility, B_utility

def assignB():
     # Put the 'Party' folder in the same folder as this file.
     path_to_folder = pathlib.Path(__file__).parent / 'Party'
     domain = io.load_genius_domain_from_folder(path_to_folder)
     A_utility, _ = UtilityFunction.from_genius(file_name=path_to_folder / 'Party-prof1.xml', issues=domain.issues)
     B_utility, _ = UtilityFunction.from_genius(file_name=path_to_folder / 'Party-prof2.xml', issues=domain.issues)

     A_utility = A_utility.normalize()
     B_utility = B_utility.normalize()

     session = SAOMechanism(issues=domain.issues, n_steps=30)
     return session, A_utility, B_utility

def visualize(negotiation_setup):
    session, A_utility, B_utility = negotiation_setup
    # to be implemented further. Add rvs.
    # adding reserved values for both agents (1(b))

    # A_utility.reserved_value = 0.5
    # B_utility.reserved_value = 0.2

    # this code is already written for you.
    # It creates and adds two agents to the session. We create info about the two agents to share the opponent's utility.
    p_info_aboutB = create_info_about_opponentutility(B_utility)
    p_info_aboutA = create_info_about_opponentutility(A_utility)
    AgentA = Group1(name="A", private_info=p_info_aboutB)
    AgentB = Boulware(name="B", private_info=p_info_aboutA)
    
    # to be implemented further
    # plots the outcome space and Pareto efficient frontier (1(a/b))
    session.add(AgentA, ufun=A_utility)
    session.add(AgentB, ufun=B_utility)

    session.run()
    session.plot(ylimits=(0.0, 1.01), show_reserved=True)
    plt.show()


# This piece of code is added to make the opponent utility known to the negotiator, excluding the rv.
# The template agent AwesomeNegotiator assumes it can access the opponent's utility.
# When initializing the agent, you can use a parameter "private_info".
def create_info_about_opponentutility(opp_utility):
    return dict(
        opponent_ufun=UFun(values=opp_utility.values, weights=opp_utility.weights, bias=opp_utility._bias, reserved_value=0,
                           outcome_space=opp_utility.outcome_space))  # type: ignore


if __name__ == "__main__":
    negotiation_setup = assignA()
    # negotiation_setup = assignB()
    visualize(negotiation_setup)