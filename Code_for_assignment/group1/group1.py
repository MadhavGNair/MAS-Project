"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Group 1
*Authors* Alexo Castro Yáñez, Haico Wong, Madhav Girish Nair, Ralph Dreijer, and Tiago dos Santos Silva Peixoto Carriço

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import random

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from negmas.preferences import pareto_frontier, nash_points


class Group1(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0
    # pareto_outcomes has the form ((a,b),c) where a=our utility, b=opponent's utility, and c=index of bid in rational outcomes
    pareto_outcomes = list()
    pareto_utilities = list()
    pareto_indices = list()
    nash_outcomes = list()
    is_final_bid = bool
    acceptance_concession_phase = {1: (1, 0), 2: (1, 0), 3: (1, 0)}
    bidding_concession_phase    = {1: (1, 0), 2: (1, 0), 3: (1, 0)}
    pareto_utilities = list()
    pareto_indices = list()
    nash_outcomes = list()
    NR_DETECTING_CELLS = 20

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        # this stores all the possible outcomes (so filtering required)
        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        rational_outcomes_copy = self.rational_outcomes.copy()

        # from rational_outcomes, select pareto optimal outcomes using the multi-layer pareto strategy
        # the strategy is to set a threshold of pseudo-pareto outcomes. If the initial layer does not have
        # threshold amount of outcomes, removes that layer and calculate the next best pareto outcomes,
        # (hence pseudo), until the threshold is reached. Set threshold to 0 to get first frontier.
        # ISSUE: WHAT IF PARETO COUNT EXCEEDS THE TOTAL NUMBER OF BIDS?
        pareto_count = 5
        self.pareto_utilities = list(pareto_frontier([self.ufun, self.opponent_ufun], self.rational_outcomes)[0])
        self.pareto_indices = list(pareto_frontier([self.ufun, self.opponent_ufun], self.rational_outcomes)[1])
        # sort indices in descending order to avoid shrinking array issue
        self.pareto_indices = sorted(self.pareto_indices, reverse=True)
        while len(self.pareto_utilities) < pareto_count:
            # remove the pareto outcomes from rational outcomes
            for p_idx in self.pareto_indices:
                del self.rational_outcomes[p_idx]
            # recompute new pareto layer
            self.pareto_utilities.extend(list(pareto_frontier([self.ufun, self.opponent_ufun], self.rational_outcomes)[0]))
            self.pareto_indices.extend(list(pareto_frontier([self.ufun, self.opponent_ufun], self.rational_outcomes)[1]))

        # calculate and store the Nash points
        # ISSUE: WORKS FINE FOR ASSIGNMENT A, BUT NOT B, B SHOWS KALAI INSTEAD OF NASH POINT
        # POSSIBLE PROBLEM MIGHT BE RESERVATION VALUES NOT BEING TAKEN INTO ACCOUNT
        self.nash_outcomes = nash_points([self.ufun, self.opponent_ufun],
                                         pareto_frontier([self.ufun, self.opponent_ufun], rational_outcomes_copy)[
                                             0])[0][0]


        ### OPPONENT MODELLING INITIALIZATION ###

        # Detecting region: First bounds of opponent's reservation value          
        for utilities in sorted(self.pareto_utilities, key= lambda x: x[0]):
            if utilities[0]>self.ufun.reserved_value:
                self.opponent_rv_upper_bound = utilities[1]
                break 
        self.opponent_rv_lower_bound = sorted(self.pareto_utilities, key= lambda x: x[1])[0][1]

        # Dividing the detecting region into detecting cells
        detecting_region_lenght = self.opponent_rv_upper_bound - self.opponent_rv_lower_bound
        self.detecting_cells_bounds = list()
        for k in range(self.NR_DETECTING_CELLS+1):
            self.detecting_cells_bounds.append(self.opponent_rv_lower_bound+(k/self.NR_DETECTING_CELLS)*detecting_region_lenght)

        # Uniform distribution in the detecting cells
        self.detecting_cells_prob = [1/self.NR_DETECTING_CELLS] * self.NR_DETECTING_CELLS

        # First guess (if needed)
        self.partner_reserved_value = self.opponent_rv_upper_bound

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counteroffer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
            - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        # compute who gets the final bid in the first step (True = we end, False = opponent ends)
        if state.step == 0:
            self.is_final_bid = True
            if offer is None:
                # if we start and total steps is even, they end the bid
                if self.nmi.n_steps % 2 == 0:
                    self.is_final_bid = False
            else:
                # if opponent starts and total steps is odd, they end the bid
                if self.nmi.n_steps % 2 != 0:
                    self.is_final_bid = False

        # compute the current phase of negotiation
        current_phase = self.compute_phase(state)

        # update the opponent's reservation value
        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        # change the concession threshold to the curve function when decided
        concession_threshold = 0.6
        if self.acceptance_strategy(state, concession_threshold):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counteroffer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER,
                           self.bidding_strategy(state, concession_threshold, current_phase))

    def acceptance_strategy(self, state: SAOState, concession_threshold) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether to accept the offer.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            concession_threshold: the threshold above or equal to which our agent will accept the offer

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer
        offer_utility = float(self.ufun(offer))

        # define two strategies for when opponent has and does not have last bid
        if self.is_final_bid:
            # if offer is above or equal to Nash point, our reservation value and our concession threshold, accept
            if offer is not None and offer_utility >= self.nash_outcomes[0]\
                    and offer_utility >= self.ufun.reserved_value and offer_utility >= concession_threshold:
                return True
        else:
            # since we are at disadvantage, simply accept valid offers above reservation value and concession threshold
            if offer is not None and offer_utility >= self.ufun.reserved_value and offer_utility >= concession_threshold:
                return True
        return False

    def bidding_strategy(self, state: SAOState, concession_threshold, phase) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counteroffer.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            concession_threshold: the threshold above or equal to which our agent will accept the offer
            phase: integer indicating the current phase of negotiation (1 for first half, 2 for second half)

        Returns: The counteroffer as Outcome.
        """

        # if we have final bid, randomly propose bids from pareto_outcomes, above Nash point, reservation values, and
        # concession curve?
        if self.is_final_bid:
            possible_bids = [bids[1] for bids in self.pareto_outcomes if bids[0][0] >= self.nash_outcomes[0] and
                             bids[0][0] > self.ufun.reserved_value and bids[0][1] > self.partner_reserved_value and
                             bids[0][0] > concession_threshold]
        else:
            # if opponent has last bid, WHAT TO DO?
            possible_bids = [bids[1] for bids in self.pareto_outcomes if bids[0][0] >= self.nash_outcomes[0] and
                             bids[0][0] > self.ufun.reserved_value and bids[0][1] > self.partner_reserved_value and
                             bids[0][0] > concession_threshold]
        # ISSUE: FOR SOME REASON POSSIBLE BIDS CAN BE EMPTY, SO ADDED A CHECKER, POSSIBLY MODIFY THIS
        if len(possible_bids) > 0:
            return random.choice(self.rational_outcomes)
        return self.rational_outcomes[random.choice(possible_bids)]

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).

        Returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if self.opponent_ufun(offer) < self.partner_reserved_value:
            self.partner_reserved_value = float(self.opponent_ufun(offer)) / 2

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        rational_outcomes = self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]
    
    def acceptance_curve(self, state: SAOState, current_phase):
        """
        This function determines the acceptance curve point at the current time step.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
        """
        m = self.ufun.reserved_value
        if current_phase == 1:
            x = state.step
            T = self.nmi.n_steps
            M = 1
            beta = 0.5
        else:
            x = state.step - self.acceptance_concession_phase[current_phase - 1][1]
            T = self.nmi.n_steps - self.acceptance_concession_phase[current_phase - 1][1]
            M = self.acceptance_concession_phase[current_phase - 1][0]

            if current_phase == 2:
                beta = 1
            else:
                beta = 1.5

        self.acceptance_concession_phase[current_phase] = (M - ((M - m) * (x / T)**(1 / beta)), state.step)

        return self.acceptance_concession_phase[current_phase][0]
    
    def bidding_curve(self, state: SAOState, final_bid, current_phase):
        """
        This function determines the bidding curve point at the current time step.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
        """
        if final_bid:
            m = self.ufun.reserved_value
        else:
            # The concession threshold aims for the maximum reservation value between the two agents
            # This allows us to "follow" the opponent's strategy, but only in the case that 
            # (our prediction of) their reservation value is higher than ours
            m = max(self.ufun.reserved_value, self.partner_reserved_value)
        
        if current_phase == 1:
            x = state.step
            T = self.nmi.n_steps
            M = 1
            beta = 0.5
        else:
            x = state.step - self.bidding_concession_phase[current_phase - 1][1]
            T = self.nmi.n_steps - self.bidding_concession_phase[current_phase - 1][1]
            M = self.bidding_concession_phase[current_phase - 1][0]

            if current_phase == 2:
                beta = 1
            else:
                beta = 1.5

        self.bidding_concession_phase[current_phase] = (M - ((M - m) * (x / T)**(1 / beta)), state.step)

        return self.bidding_concession_phase[current_phase][0]
    
    def compute_phase(self, state: SAOState) -> int:
        """
        Function to compute the current phase of negotiation. First half is Phase 1, the next 1/4th is Phase 2, and
        the remaining steps is Phase 3.
        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).

        Returns:
            The phase as an Integer.
        """
        if state.step <= (self.nmi.n_steps // 2):
            return 1
        elif state.step <= ((3 * self.nmi.n_steps) // 4):
            return 2
        else:
            return 3


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Group1, small=True)
