"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Group 1
*Authors* Alexo Castro Yáñez, Haico Wong, Madhav Girish Nair, Ralph Dreijer, and Tiago dos Santos Silva Peixoto Carriço

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import math
import random

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from negmas.preferences import pareto_frontier, nash_points
import numpy as np
np.seterr(all='raise')

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
    opponent_ends = bool
    acceptance_concession_phase = {1: (1, 0), 2: (1, 0), 3: (1, 0)}
    bidding_concession_phase    = {1: (1, 0), 2: (1, 0), 3: (1, 0)}
    NR_DETECTING_CELLS = 20
    NR_STEPS_BEFORE_OPP_MODEL = 10
    opponent_bid_history = list()
    opponent_utility_history = list()
    opponent_rv_upper_bound = float()
    opponent_rv_lower_bound = float()
    detecting_cells_bounds = list()
    detecting_cells_prob = np.array

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

        # sort pareto_utilities and pareto_indices in descending order
        combined_pareto = list(zip(self.pareto_utilities, self.pareto_indices))
        self.pareto_outcomes = sorted(combined_pareto, key=lambda x: x[0][0], reverse=True)

        # calculate and store the Nash points
        # ISSUE: WORKS FINE FOR ASSIGNMENT A, BUT NOT B, B SHOWS KALAI INSTEAD OF NASH POINT
        # POSSIBLE PROBLEM MIGHT BE RESERVATION VALUES NOT BEING TAKEN INTO ACCOUNT
        self.nash_outcomes = nash_points([self.ufun, self.opponent_ufun],
                                         pareto_frontier([self.ufun, self.opponent_ufun], rational_outcomes_copy)[
                                             0])[0][0]

        ### OPPONENT MODELLING INITIALIZATION ###

        # Detecting region: First bounds of opponent's reservation value          
        for utilities in sorted(self.pareto_utilities, key= lambda x: x[0]):
            if utilities[0] > self.ufun.reserved_value:
                self.opponent_rv_upper_bound = utilities[1]
                break 
        self.opponent_rv_lower_bound = sorted(self.pareto_utilities, key= lambda x: x[1])[0][1]

        # Uniform distribution in the detecting cells
        #self.detecting_cells_prob = [1/self.NR_DETECTING_CELLS] * self.NR_DETECTING_CELLS
        self.detecting_cells_prob = np.full(self.NR_DETECTING_CELLS, fill_value=1/self.NR_DETECTING_CELLS)

        self.partner_reserved_value = self.opponent_rv_upper_bound # First guess (if needed)


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

        # Compute who gets the final bid (only on first step)
        if state.step == 0:
            if offer is None:
                self.opponent_ends = True
            else:
                self.opponent_ends = False

        # Compute the current phase (first or second half of negotiation)
        current_phase = self.compute_phase(state)

        # History of opponent bids and utilities
        if offer is not None:
            self.opponent_bid_history.append(offer)
            self.opponent_utility_history.append(self.opponent_ufun(offer))
            #print(self.opponent_utility_history)

        # Update reservation value (only after the opponent has given us 2 proposals)
        if len(self.opponent_utility_history) >= self.NR_STEPS_BEFORE_OPP_MODEL:
            self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        # change the concession threshold to the curve function when decided
        concession_threshold = self.acceptance_curve(state, current_phase)
        if self.acceptance_strategy(state, concession_threshold):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        
        concession_threshold = self.bidding_curve(state, current_phase)

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
        if not self.opponent_ends:
            # if offer is above or equal to Nash point, our reservation value and our concession threshold, accept
            if offer is not None and offer_utility >= self.nash_outcomes[0]\
                    and offer_utility >= self.ufun.reserved_value and offer_utility >= concession_threshold:
                return True
        else:
            # since we are at disadvantage, simply accept valid offers above reservation value and concession threshold
            if offer is not None and offer_utility >= self.ufun.reserved_value and offer_utility >= concession_threshold:
                return True
        return False

    def bidding_strategy(self, state: SAOState, concession_threshold) -> Outcome | None:
        """
        This function implements how (counter-)offers are made.
        The basic idea is to filter bids that are on the Pareto frontier, that gives us better utility that Nash point,
        that gives utility above both our and the opponent's reservation value, and above the concession threshold.
        However, there is a 5% chance of randomly choosing a bid from all rational outcomes to throw the opponent's
        estimate off.
        One caveat is that, if we have the final bid, in the last few steps (last 5%), we will bid bids that are slightly
        above the opponent's reservation estimate, but gives us the highest utility.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            concession_threshold: the threshold above or equal to which our agent will accept the offer

        Returns: The counteroffer as Outcome.
        """

        # define epsilon to be the chance of random bid being offered
        epsilon = 0.05
        # this threshold defines the final number of bids where stubborn strategy is implemented
        final_bid_threshold = int(0.05 * self.nmi.n_steps) if self.nmi.n_steps > 100 else 5
        # compute all possible bids given the criteria for 'good' bids
        possible_bids = [bids for bids in self.pareto_outcomes if bids[0][0] >= self.nash_outcomes[0] and
                         bids[0][0] > self.ufun.reserved_value and bids[0][1] > self.partner_reserved_value and
                         bids[0][0] > concession_threshold]

        # in the rare case that there are no bids that satisfy the above conditions, bid the best bid for us
        if len(possible_bids) == 0:
            return self.rational_outcomes[max(self.pareto_outcomes, key=lambda x:x[0][0])[1]]
        # if we have final bid, and the final steps are reached, bid the best offers
        elif self.opponent_ends == False and state.step >= final_bid_threshold:
            best_offers = [offer for offer in self.rational_outcomes if offer[0][1] > self.partner_reserved_value]
            return self.rational_outcomes[min(best_offers, key=lambda x: x[0][0])[1]]
        # if in any other scenario, bid the lowest bid for opponent
        # IF PERFORMANCE IS BAD, MAYBE KEEP TRACK OF OFFERED BIDS AND DONT OFFER THEM AGAIN
        elif epsilon <= random.random():
            return self.rational_outcomes[min(self.pareto_outcomes, key=lambda x: x[0][1])[1]]
        return self.rational_outcomes[random.choice(possible_bids)[1]]

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).

        Returns: None.
        """
        assert self.ufun and self.opponent_ufun

        # Initialization part 2
        if len(self.opponent_utility_history) == self.NR_STEPS_BEFORE_OPP_MODEL:
            # Update upper bound with first bid information
            if self.opponent_rv_upper_bound >= self.opponent_utility_history[0]:
                self.opponent_rv_upper_bound = self.opponent_utility_history[0]-self.opponent_utility_history[0]/self.NR_DETECTING_CELLS
            self.partner_reserved_value = self.opponent_rv_upper_bound # First guess (if needed)

            # Dividing the detecting region into detecting cells
            detecting_region_length = self.opponent_rv_upper_bound - self.opponent_rv_lower_bound
            for k in range(self.NR_DETECTING_CELLS+1):
                self.detecting_cells_bounds.append(self.opponent_rv_lower_bound+(k/self.NR_DETECTING_CELLS)*detecting_region_length)

        def polynomial_concession(t, reservation_value, beta):
            return self.opponent_utility_history[0] + (reservation_value - self.opponent_utility_history[0]) * pow(t/self.nmi.n_steps, beta) 

        def fitted_beta(reservation_value):
            p_i = np.log([(self.opponent_utility_history[0]-self.opponent_utility_history[t]) for t in range(1, len(self.opponent_utility_history))]) -np.log([(self.opponent_utility_history[0] - reservation_value) for t in range(1, len(self.opponent_utility_history))])
            #p_i = np.log([(self.opponent_utility_history[0] - self.opponent_utility_history[t])/(self.opponent_utility_history[0] - reservation_value) for t in range(1, len(self.opponent_utility_history))])
            t_i = np.log([t/self.nmi.n_steps for t in range(1, len(self.opponent_utility_history))])
            beta = np.dot(p_i, t_i)/np.dot(t_i, t_i)
            if beta == float("inf"):
                return 5
            else:
                return beta

        def fitted_alpha(reservation_value):
            vp_max = max(self.opponent_utility_history)  # maximum utility achieved by the opponent
            vp_t = self.opponent_utility_history[-1]  # current utility, opponent
            vp_tminus = self.opponent_utility_history[-2]  # previous utility, opponent

            #alpha = math.log( vp_max - vp_t/vp_max - vp_tminus, base= state.step/(state.step - 1) )
            alpha =  (math.log(vp_max - vp_t) - math.log(vp_max - vp_tminus) ) / math.log( state.step / (state.step - 1)) # equation 8 Zhang et al. 
            if alpha !=0:
                return alpha
            else:
                raise ValueError("Null alpha value")


        def non_linear_correlation(reservation_value):
            beta = fitted_beta(reservation_value)
            fitted_offers = [polynomial_concession(t, reservation_value, beta) for t in range(1, len(self.opponent_utility_history))]
            normalized_fitted_offers = np.array(fitted_offers) - np.mean(fitted_offers)
            normalized_history_offers = np.array(self.opponent_utility_history[1:]) - np.mean(self.opponent_utility_history[1:])
            denominator = math.sqrt(np.dot(normalized_fitted_offers, normalized_fitted_offers) * np.dot(normalized_history_offers, normalized_history_offers))
            if denominator != 0:
                return np.dot(normalized_fitted_offers, normalized_history_offers) / denominator
            else:
                print(f"Beta: {beta}")
                print(f"RV: {reservation_value}")
                print(f"Denominator: {denominator}")
                print(f"Norm Offer fitted: {normalized_fitted_offers}")
                print(f"Norm Offer history: {normalized_history_offers}")
                raise ValueError("Null correlation denominator")

        # Take random reservation values in the detecting cells
        random_reservation_values = [random.uniform(self.detecting_cells_bounds[k], self.detecting_cells_bounds[k+1]) for k in range(self.NR_DETECTING_CELLS)]

        # Compute the fitted curves and the non-linear correlation with history utilities
        likelihoods = [non_linear_correlation(rv)**2 for rv in random_reservation_values]

        # Bayesian update of detection cells probabilities
        prior_prob = self.detecting_cells_prob
        posterior_prob = np.zeros(self.NR_DETECTING_CELLS)
        for k in range(self.NR_DETECTING_CELLS):
            posterior_prob[k] = prior_prob[k] * likelihoods[k] / np.dot(prior_prob, np.array(likelihoods))
        self.detecting_cells_prob = posterior_prob

        # Selecting a reservation value from detecting cells probability distribution: the upper bound of the cell with max prob.
        self.partner_reserved_value = self.detecting_cells_bounds[np.argmax(self.detecting_cells_prob) + 1]

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        """ rational_outcomes = self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ] """
    
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

        self.acceptance_concession_phase[current_phase] = (M - ((M - m) * pow(x/T, beta)), state.step)

        return self.acceptance_concession_phase[current_phase][0]

    def bidding_curve(self, state: SAOState, current_phase):
        """
        This function determines the bidding curve point at the current time step.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            current_phase: the current phase
        """
        if not self.opponent_ends:
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

        self.bidding_concession_phase[current_phase] = (M - ((M - m) * pow(x/T, beta)), state.step)

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
