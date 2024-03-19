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


class Group1(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0

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

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        self.partner_reserved_value = self.ufun.reserved_value

        # Initialize the utilities

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

        # Compute who gets the final bid
        is_final_bid = True
        if offer is None:
            # if we start and total steps is even, they end the bid
            if self.nmi.n_steps % 2 == 0:
                is_final_bid = False
        else:
            # if opponent starts and total steps is odd, they end the bid
            if self.nmi.n_steps % 2 != 0:
                is_final_bid = False

        # Compute the current phase (first or second half of negotiation)
        current_phase = 1 if state.step <= int(self.nmi.n_steps / 2) else 2

        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        # change the concession threshold to the curve function when decided
        concession_threshold = 0.6
        if self.acceptance_strategy(
            state, concession_threshold, is_final_bid, current_phase
        ):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counteroffer in the bidding_strategy
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            self.bidding_strategy(
                state, concession_threshold, is_final_bid, current_phase
            ),
        )

    def acceptance_strategy(
        self, state: SAOState, concession_threshold, final_bid, phase
    ) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether to accept the offer.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            concession_threshold: the threshold above or equal to which our agent will accept the offer
            final_bid: boolean indicating whether our agent has the final bid or not (True if we do, else False)
            phase: integer indicating the current phase of negotiation (1 for first half, 2 for second half)

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer

        print("Current offer = ", offer, ", value = ", self.ufun(offer))

        # if the offer is valid, not worse than our reservation value, and larger than or equal to
        # our concession threshold, accept bid, else reject
        if (
            offer is not None
            and float(self.ufun(offer)) > self.ufun.reserved_value
            and float(self.ufun(offer) >= concession_threshold)
        ):
            return True
        return False

    def bidding_strategy(
        self, state: SAOState, concession_threshold, final_bid, phase
    ) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counteroffer.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            concession_threshold: the threshold above or equal to which our agent will accept the offer
            final_bid: boolean indicating whether our agent has the final bid or not (True if we do, else False)
            phase: integer indicating the current phase of negotiation (1 for first half, 2 for second half)

        Returns: The counteroffer as Outcome.
        """

        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        # BASIC IDEA:
        # IF WE DO NOT HAVE LAST BID:
        # - If in first phase, filter out offers above estimated opponents reservation value, Nash point, along the Pareto
        #   front and sort them in descending order. Keep bidding in order and recycle when end of list is reached. Also,
        #   store the bids offered by the opponent (above reservation) and sort them in descending order of our utility.
        # - If in second phase, recycle the list of stored bids from the opponent.
        # IF WE HAVE LAST BID:
        # Same strategy as first phase of ^.

        if final_bid or phase == 1:
            # We have the last bid or are in the first phase
            pareto_outcomes = self._pareto_outcomes(self.rational_outcomes)
            print(pareto_outcomes)
            pass
        else:
            # We do not have the last bid and are in the second phase
            pass

        # print([self.ufun(s) for s in self.rational_outcomes])
        return random.choice(self.rational_outcomes)

    def _pareto_outcomes(self, outcomes):
        res = []
        for o in outcomes:
            util = self.ufun(o)
            opponent_util = self.opponent_ufun(o)
            if all(
                util < self.ufun(p)
                for p in outcomes
                if self.opponent_ufun(p) < opponent_util
            ):
                res.append(o)
        return res

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
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


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Group1, small=True)
