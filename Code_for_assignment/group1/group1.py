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

import matplotlib
matplotlib.use('Agg')

class Group1(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes: list[Outcome]
    pareto_outcomes: list[tuple[tuple[float, float], int]]  #((a,b),c) where a=our utility, b=opponent's utility, and c=index of bid in rational outcomes
    pareto_utilities: list[tuple[float, float]]
    pareto_indices: list[int]
    nash_outcomes: list
    opponent_ends: bool
    acceptance_concession_phase : dict
    bidding_concession_phase : dict
    utility_history: list[float] 
    differences: list[list[float]] 
    phase_count: dict
    debug: bool = True

    verbosity_opponent_modelling: bool = False
    opponent_reserved_value: int
    opp_model_started: bool  
    nr_opponent_rv_updates: int 
    opponent_differences: list[list[float]] 
    opponent_bid_history: list[Outcome]
    opponent_utility_history: list[float]
    opponent_rv_upper_bound: float
    opponent_rv_lower_bound: float
    NR_DETECTING_CELLS: int = 20
    detecting_cells_bounds: list[float]
    detecting_cells_prob: np.array
    
    def reset_variables(self):
        self.rational_outcomes = list()
        self.pareto_outcomes = list()
        self.pareto_utilities = list()
        self.pareto_indices = list()
        self.nash_outcomes = list()
        self.acceptance_concession_phase = {1: (1, 0), 2: (1, 0), 3: (1, 0)}
        self.bidding_concession_phase    = {1: (1, 0), 2: (1, 0), 3: (1, 0)}
        self.utility_history = list()
        self.differences = list()
        self.phase_count = {'First': 0, 'Second': 0, 'Third': 0}

        self.opponent_reserved_value = 0
        self.opp_model_started = False
        self.nr_opponent_rv_updates = 0
        self.opponent_differences = list()
        self.opponent_bid_history = list()
        self.opponent_utility_history = list()
        self.opponent_rv_upper_bound = 1
        self.opponent_rv_lower_bound = 0
        self.detecting_cells_bounds = list()
        
    def get_pareto_outcomes(self):
        # from rational_outcomes, select pareto optimal outcomes using the multi-layer pareto strategy
        # the strategy is to set a threshold of pseudo-pareto outcomes. If the initial layer does not have
        # threshold amount of outcomes, removes that layer and calculate the next best pareto outcomes,
        # (hence pseudo), until the threshold is reached. Set threshold to 0 to get first frontier.
        pareto_count = 0
        self.pareto_utilities, self.pareto_indices = map(list, pareto_frontier([self.ufun, self.opponent_ufun],
                                                                               outcomes=self.rational_outcomes))
        # sort indices in descending order to avoid shrinking array issue
        # self.pareto_indices = sorted(self.pareto_indices, reverse=True)

        rational_copy = self.rational_outcomes.copy()

        while len(self.pareto_utilities) < pareto_count:
            # remove the pareto outcomes from rational outcomes
            rational_copy = [outcome for idx, outcome in enumerate(rational_copy) if idx not in self.pareto_indices]
            # recompute new pareto layer
            self.pareto_utilities.extend(list(pareto_frontier([self.ufun, self.opponent_ufun], rational_copy)[0]))
            self.pareto_indices.extend(list(pareto_frontier([self.ufun, self.opponent_ufun], rational_copy)[1]))

        # sort pareto_utilities and pareto_indices in descending order
        combined_pareto = list(zip(self.pareto_utilities, self.pareto_indices))
        self.pareto_outcomes = sorted(combined_pareto, key=lambda x: x[0][0], reverse=True)


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

        ###  INITIALIZE NEGOTIATION ENVIRONMENT ###
        self.reset_variables()

        # Store all possible outcomes with higher utility than our reservation value
        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Initialize the list of pareto outcomes
        self.get_pareto_outcomes()

        # Calculate and store the Nash points (we are selecting the first one)
        self.nash_outcomes = nash_points([self.ufun, self.opponent_ufun],
                                         pareto_frontier([self.ufun, self.opponent_ufun], self.rational_outcomes.copy())[0])[0][0]

        ### OPPONENT MODELLING INITIALIZATION ###

        # Detecting region: First bounds of opponent's reservation value
        for utilities in sorted(self.pareto_utilities, key= lambda x: x[0]):
            if utilities[0] > self.ufun.reserved_value:
                self.opponent_rv_upper_bound = utilities[1]
                break
        self.opponent_rv_lower_bound = sorted(self.pareto_utilities, key= lambda x: x[1])[0][1]

        # Uniform distribution in the detecting cells
        self.detecting_cells_prob = np.full(self.NR_DETECTING_CELLS, fill_value=1/self.NR_DETECTING_CELLS)
        
        # First guess (if needed)
        self.opponent_reserved_value = self.opponent_rv_upper_bound 


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

        # Save opponent's history of bids and utilities. Update utilities differences.
        if offer is not None:
            self.opponent_bid_history.append(offer)
            if len(self.opponent_utility_history) > 0 and self.opponent_ufun(offer) > self.opponent_utility_history[0]:
                self.opponent_utility_history.append(self.opponent_utility_history[-1])
            else:
                self.opponent_utility_history.append(self.opponent_ufun(offer))
        if len(self.opponent_utility_history) > 1:
            self.update_differences(differences=self.opponent_differences, utility_history=self.opponent_utility_history)

        # Compute time-dependency criterion
        if len(self.opponent_differences) > 1:
            time_criterion = self.compute_time_criterion(differences=self.opponent_differences)
            if self.verbosity_opponent_modelling:
                print(f"(opp ends={self.opponent_ends}) Step {state.step} (Rel t = {state.relative_time}):  Behaviour criterion = {self.compute_behaviour_criterion()}, Time criterion = {time_criterion}")
            # Start the opponent modelling when the time-dependency criterion is satisfied and opponent has proposed at least 3 different bids.
            if not self.opp_model_started:
                if time_criterion > 0.5 and len(np.unique(self.opponent_utility_history)) > 2:
                    self.opp_model_first_step = state.step
                    self.opp_model_started = True
                    if self.verbosity_opponent_modelling:
                        print(f"(Opponent ends={self.opponent_ends}) Opponent modelling started in step {state.step} (Rel time = {state.relative_time}) with time criterion {time_criterion}")
        if self.opp_model_started:
            self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy. Change the concession threshold to the curve function when decided
        concession_threshold = self.acceptance_curve(state, current_phase)
        if self.acceptance_strategy(state, concession_threshold):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        concession_threshold = self.bidding_curve(state, current_phase)

        # If it's not acceptable, determine the counteroffer in the bidding_strategy and return it. Update self utility differences.
        bid = self.bidding_strategy(state, concession_threshold)
        self.utility_history.append(self.ufun(bid))
        if len(self.utility_history) > 1:
            self.update_differences(differences=self.differences, utility_history=self.utility_history)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

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
        One caveat is that, if we have the final bid, in the last few steps (last 5%), we will bid bids that are slightly
        above the opponent's reservation estimate, but gives us the highest utility.
        Once a bid has been made, it is removed from the list of bids so that it is not offered again. However, once all
        possible bids have been made, and no agreement has been reached, the pareto frontier is recomputed and bids are
        recycled.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
            concession_threshold: the threshold above or equal to which our agent will accept the offer

        Returns: The counteroffer as Outcome.
        """
        # Initialization of bids:
        # check if pareto outcomes is empty, if so re-initialize the list
        if len(self.pareto_outcomes) == 0:
            self.get_pareto_outcomes()
        # compute all possible bids given the criteria for 'good' bids
        possible_bids = [bids for bids in self.pareto_outcomes if bids[0][0] >= self.nash_outcomes[0] and
                         bids[0][0] > self.ufun.reserved_value and bids[0][1] > self.opponent_reserved_value and
                         bids[0][0] > concession_threshold]

        # Bidding process:
        # this threshold defines the final number of bids where stubborn strategy is implemented
        final_bid_threshold = int(0.10 * self.nmi.n_steps) if self.nmi.n_steps > 100 else 10
        # in the case that there are no bids that satisfy the above conditions, bid the best bid for us
        if len(possible_bids) == 0:
            bid_idx = self.pareto_outcomes[0][1]
            return self.rational_outcomes[bid_idx]
        # if we have final bid, and the final steps are reached, bid the best offers, but do not recycle
        elif self.opponent_ends == False and state.step >= final_bid_threshold:
            best_offers = [offer for offer in possible_bids if offer[0][1] > self.opponent_reserved_value]
            bid_idx = max(best_offers, key=lambda x: x[0][0])[1]
            return self.rational_outcomes[bid_idx]
        # if in any other scenario, bid the best bids in decreasing order for us
        else:
            bid_idx = max(possible_bids, key=lambda x: x[0][0])[1]
            self.pareto_outcomes = [bid for bid in self.pareto_outcomes if bid[1] != bid_idx]
            return self.rational_outcomes[bid_idx]

    def update_differences(self, differences: list[list[float]], utility_history: list[float]) -> None:
        if self.debug: assert len(utility_history) > 1
        for k in range(len(utility_history)-1):
            if k == len(utility_history)-2: # Once per update (at the last step) we create the list with the new order of differences
                if k==0:
                    differences.append([utility_history[1]-utility_history[0]])
                else:
                    differences.append([differences[k-1][-1]-differences[k-1][-2]])
            else:
                if k==0:
                    differences[k].append(utility_history[-1]-utility_history[-2])
                else:
                    differences[k].append(differences[k-1][-1]-differences[k-1][-2])

    def sign_subcriterion(self, positive_differences, negative_differences) -> float:
        pos_sum = np.sum(positive_differences)
        neg_sum = np.sum(negative_differences)
        if len(positive_differences)>0 and len(negative_differences)>0:
            return abs((pos_sum/len(positive_differences))+(neg_sum/len(negative_differences))) / max([abs(pos_sum/len(positive_differences)), abs(neg_sum/len(negative_differences))])
        elif len(positive_differences) == 0 and len(negative_differences)==0:
            return 0
        else:
            return 1
        
    def compute_time_criterion(self, differences: list[list[float]]) -> float:
        if self.debug: assert len(differences) > 1
        D = np.empty(len(differences))
        sum_differences = list()
        for k in range(len(differences)):
            positive_differences = np.array(differences[k])[np.greater(differences[k], np.zeros(len(differences[k])))]
            negative_differences = np.array(differences[k])[np.less(differences[k], np.zeros(len(differences[k])))]
            sum_differences.append({"pos": np.sum(positive_differences), "neg": np.sum(negative_differences)})
            D[k] = self.sign_subcriterion(positive_differences, negative_differences)
        max_sum_differences = [max([sum_differences[k]["pos"], sum_differences[k]["neg"]]) for k in range(len(differences))]
        w = np.array([max_sum_differences[k]/np.sum(max_sum_differences) if np.sum(max_sum_differences)>0 else 0 for k in range(len(differences))])
        if self.debug: assert len(D) == len(D[np.invert(np.isnan(D))])
        return np.dot(w[np.invert(np.isnan(D))], D[np.invert(np.isnan(D))])
    
    def compute_behaviour_criterion(self) -> float:
        m = min([len(self.differences[0]), len(self.opponent_differences[0])])
        if self.debug: assert len(self.opponent_differences[0]) == len(self.differences[0]) or len(self.opponent_differences[0]) == len(self.differences[0])+1
        r_i =  [self.opponent_differences[0][-m:][i]/self.differences[0][-m:][i] if not math.isclose(self.differences[0][-m:][i], 0) else 0 for i in range(m)]
        w_i = [ 2*i/m*(m+1) for i in range(m)]
        r = np.dot(r_i, w_i)
        def h(r) -> float:
            if r/2 <=0:
                return 0
            elif r/2 >=1:
                return 1
            else:
                return r/2
        D_1 = h(r)
        r_i_pos_differences = np.array(r_i)[np.greater(r_i, np.zeros(len(r_i)))]
        r_i_neg_differences = np.array(r_i)[np.less(r_i, np.zeros(len(r_i)))]
        D_2 = self.sign_subcriterion(positive_differences=r_i_pos_differences, negative_differences=r_i_neg_differences)
        return (D_1+D_2)/2

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        This function updates the opponent reserved value when it is assumed to be following a time concession strategy.
        In order to do that, the following steps are done:
            1. Create a detecting region at the known deadline with possible reservation values. 
               Initilize a prior distribution for the detecting region.
            2. Fit different polinomial concession curves using the different reservation values and 
               a beta value computed by regression with the opponent bid history.
            3. Compute the correlatation between the curves and the bid history.
            4. Use those correlations as the likelihoods of each reservation value (i.e. each detecting cell) 
               and perform the bayesian update of the detecting region distribution.
            5. Update the reservation value from the distribution by using the cell with max probability.
        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).
        Returns: None.
        """
        # Create the detecting region
        if state.step == self.opp_model_first_step:
            # Update upper bound with first bid utility
            if self.opponent_rv_upper_bound >= self.opponent_utility_history[0]:
                self.opponent_rv_upper_bound = self.opponent_utility_history[0]-self.opponent_utility_history[0]/self.NR_DETECTING_CELLS
            self.opponent_reserved_value = self.opponent_rv_upper_bound # First guess (if needed)

            # Dividing the detecting region into detecting cells
            detecting_region_length = self.opponent_rv_upper_bound - self.opponent_rv_lower_bound
            for k in range(self.NR_DETECTING_CELLS+1):
                self.detecting_cells_bounds.append(self.opponent_rv_lower_bound+(k/self.NR_DETECTING_CELLS)*detecting_region_length)

        def polynomial_concession(t, reservation_value, beta):            
            return self.opponent_utility_history[0] + (reservation_value - self.opponent_utility_history[0]) * pow(t/self.nmi.n_steps, beta)

        def fitted_beta(reservation_value):
            p_i = np.log([(self.opponent_utility_history[0] - self.opponent_utility_history[t])/(self.opponent_utility_history[0] - reservation_value) 
                          if abs(self.opponent_utility_history[0] - self.opponent_utility_history[t])>1e-3 else 1e-3 for t in range(1, len(self.opponent_utility_history))])
            t_i = np.log([t/self.nmi.n_steps for t in range(1, len(self.opponent_utility_history))])
            beta = np.dot(p_i, t_i)/np.dot(t_i, t_i)
            if beta == float("inf"):
                if self.debug:
                    raise ValueError("Fitted beta computed with nonlinear regression is inf")
                else:
                    return 50
            else:
                return beta

        def fitted_alpha(reservation_value):
            max_utility = max(self.opponent_utility_history)  # maximum utility achieved by the opponent
            current_utility = self.opponent_utility_history[-1]  # current utility, opponent
            current_step = state.step
            # Take the last utility different from the current and the corresponding time step
            for k in range(2, len(self.opponent_utility_history)):
                previous_utility = self.opponent_utility_history[-k]
                if abs(current_utility-previous_utility)>1e-10:
                    previous_step = current_step - k+1
                    break
            alpha =  math.log((max_utility - current_utility)/(max_utility - previous_utility)) / math.log( current_step / previous_step) # equation 8 Zhang et al.
            if alpha !=0:
                if alpha>100:
                    if self.verbosity_opponent_modelling: print("Alpha adjusted to 100")
                    return 100
                else:
                    return alpha
            else:
                if self.verbosity_opponent_modelling: 
                    print(f"Null alpha in step {state.step} because the log argument is {(max_utility - current_utility)/(max_utility - previous_utility)}")
                return 0.001

        def non_linear_correlation(reservation_value: float) -> float:
            """
            Compute the non linear correlation between the fitted concesion and the bid history

            Args:
                reservation_value: The assumed reservation value of the fitted concession curve.

            Returns: The correlation between curve and history
            """
            beta = fitted_beta(reservation_value)
            fitted_offers = [polynomial_concession(t, reservation_value, beta) for t in range(1, len(self.opponent_utility_history))]
            normalized_fitted_offers = np.array(fitted_offers) - np.mean(fitted_offers)
            normalized_history_offers = np.array(self.opponent_utility_history[1:]) - np.mean(self.opponent_utility_history[1:])
            denominator = math.sqrt(np.dot(normalized_fitted_offers, normalized_fitted_offers) * np.dot(normalized_history_offers, normalized_history_offers))
            if denominator != 0:
                return np.dot(normalized_fitted_offers, normalized_history_offers) / denominator
            elif self.debug:
                if self.verbosity_opponent_modelling:
                    print(f"Null denominator in the correlation of the curve with beta={beta} and RV={reservation_value}")
                    print(f"Mean Norm Offer fitted: {np.mean(normalized_fitted_offers)}")
                    print(f"Mean Norm Offer history: {np.mean(normalized_history_offers)}")
                raise ValueError("Null correlation denominator")
            else:
                return 0

        # Take random reservation values in the detecting cells
        random_reservation_values = [random.uniform(self.detecting_cells_bounds[k], self.detecting_cells_bounds[k+1]) for k in range(self.NR_DETECTING_CELLS)]

        # Compute the fitted curves and the non-linear correlation with history utilities
        likelihoods = [non_linear_correlation(rv) for rv in random_reservation_values]

        # Bayesian update of detection cells probabilities
        prior_prob = self.detecting_cells_prob
        posterior_prob = np.zeros(self.NR_DETECTING_CELLS)
        for k in range(self.NR_DETECTING_CELLS):
            posterior_prob[k] = prior_prob[k] * likelihoods[k] / np.dot(prior_prob, np.array(likelihoods))
        self.detecting_cells_prob = posterior_prob

        # Selecting a reservation value from detecting cells probability distribution: the upper bound of the cell with max prob.
        self.opponent_reserved_value = self.detecting_cells_bounds[np.argmax(self.detecting_cells_prob) + 1]
        self.nr_opponent_rv_updates += 1

        def compute_expected_reserved_value():
            return np.dot(self.detecting_cells_prob, [(self.detecting_cells_bounds[k+1]-self.detecting_cells_bounds[k])/2 for k in range(len(self.detecting_cells_prob))])

        if self.verbosity_opponent_modelling and state.step % 10 == 0:
            print(f"(Opponent ends={self.opponent_ends}) Partner max predicted reserved value at step {state.step} (rel time={state.relative_time}) = {self.opponent_reserved_value}")
            print(f"(Opponent ends={self.opponent_ends}) Partner avg predicted reserved value at step {state.step} (rel time={state.relative_time}) = {compute_expected_reserved_value()}")
            print(f"Total nr of updates = {self.nr_opponent_rv_updates}")

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
            current_phase: the current phase of negotiation
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
            m = max(self.ufun.reserved_value, self.opponent_reserved_value)

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

    run_a_tournament(Group1, small=True, debug=False)
