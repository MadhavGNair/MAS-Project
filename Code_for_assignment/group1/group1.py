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
    utility_history_bidded_by_self: list[float] 
    differences_bidded_by_self: list[list[float]] 
    utility_history_bidded_by_opp: list[float] 
    differences_bidded_by_opp: list[list[float]] 
    debug: bool = False

    deactivate_opponent_modelling: bool = False
    verbosity_opponent_modelling: int = 0
    opponent_reserved_value: int
    opp_model_started: bool  
    nr_opponent_rv_updates: int 
    opponent_bid_history: list[Outcome]
    opponent_utility_history_bidded_by_opp: list[float]
    opponent_differences_bidded_by_opp: list[list[float]] 
    opponent_rv_upper_bound: float
    opponent_rv_lower_bound: float
    NR_DETECTING_CELLS: int = 10
    detecting_cells_bounds: list[float]
    detecting_cells_prob: np.array
    
    def reset_variables(self):
        self.concession_threshold = 1
        self.phase = 1
        self.rational_outcomes = list()
        self.pareto_outcomes = list()
        self.pareto_utilities = list()
        self.pareto_indices = list()
        self.nash_outcomes = list()
        self.utility_history_bidded_by_self = list()
        self.differences_bidded_by_self = list()
        self.utility_history_bidded_by_opp = list()
        self.differences_bidded_by_opp = list()

        self.opponent_reserved_value = 0
        self.opp_model_started = False
        self.nr_opponent_rv_updates = 0
        self.opponent_differences_bidded_by_opp = list()
        self.opponent_bid_history = list()
        self.opponent_utility_history_bidded_by_opp = list()
        self.opponent_rv_upper_bound = 1
        self.opponent_rv_lower_bound = 0
        self.detecting_cells_bounds = list()

        self.nr_steps_last_phase = max([3, math.ceil(0.1 * (self.nmi.n_steps - 1))]) if self.nmi.n_steps < 500 else 50
        self.phases = {1:{'T': self.nmi.n_steps-1, 'M': 1, 't_offset': 0}, 
                       2:{'T': self.nmi.n_steps-1, 'M': 1, 't_offset': 0}, 
                       3: {'T': self.nmi.n_steps-1, 'M': 1, 't_offset': 0}}
        
    def get_pareto_outcomes(self):
        # from rational_outcomes, select pareto optimal outcomes using the multi-layer pareto strategy
        # the strategy is to set a threshold of pseudo-pareto outcomes. If the initial layer does not have
        # threshold amount of outcomes, removes that layer and calculate the next best pareto outcomes,
        # (hence pseudo), until the threshold is reached. Set threshold to 0 to get first frontier.
        pareto_count = len(self.rational_outcomes) * 0.25
        self.pareto_utilities, self.pareto_indices = [], []

        rational_copy = self.rational_outcomes.copy()

        while len(self.pareto_utilities) < pareto_count or pareto_count == 0 :
            # recompute new pareto layer
            utilities, indices = map(list, pareto_frontier([self.ufun, self.opponent_ufun], rational_copy))

            if len(utilities) == 0:
                break

            for idx in range(len(indices)):
                outcome = rational_copy[indices[idx]]
                index_in_rational_outcomes = self.rational_outcomes.index(outcome)

                self.pareto_utilities.append(utilities[idx])
                self.pareto_indices.append(index_in_rational_outcomes)

            rational_copy = [outcome for idx, outcome in enumerate(rational_copy) if idx not in indices]

            if len(rational_copy) == 0:
                break

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

        # Initialize the list of pareto outcomes.
        self.get_pareto_outcomes()

        # Statarting bid as the pareto bid with highest utility for us.
        self.starting_bid_concession = self.pareto_outcomes[0][0][0]

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
        self.opponent_reserved_value = (self.opponent_rv_upper_bound - self.opponent_rv_lower_bound)/2

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
        self.compute_phase(state)

        # Save opponent's history of bids and utilities. Update utilities differences.
        if offer is not None:
            self.opponent_bid_history.append(offer)
            if len(self.opponent_utility_history_bidded_by_opp) > 0 and self.opponent_ufun(offer) > self.opponent_utility_history_bidded_by_opp[0]:
                self.opponent_utility_history_bidded_by_opp.append(self.opponent_utility_history_bidded_by_opp[-1])
            else:
                self.opponent_utility_history_bidded_by_opp.append(self.opponent_ufun(offer))
            self.utility_history_bidded_by_opp.append(self.ufun(offer))

        if not self.deactivate_opponent_modelling:
            if len(self.opponent_utility_history_bidded_by_opp) > 1:
                self.update_differences(differences=self.opponent_differences_bidded_by_opp, 
                                        utility_history=self.opponent_utility_history_bidded_by_opp)
                self.update_differences(differences=self.differences_bidded_by_opp, 
                                        utility_history=self.utility_history_bidded_by_opp,
                                        max_order=2)
            # Compute time-dependency criterion
            if len(self.opponent_differences_bidded_by_opp) > 1:
                time_criterion = self.compute_time_criterion(differences=self.opponent_differences_bidded_by_opp)
                if self.verbosity_opponent_modelling > 3:
                    print(f"(opp ends={self.opponent_ends}) Step {state.step} (Rel t = {state.relative_time}): \
                        Behaviour criterion = {self.compute_behaviour_criterion()}, Time criterion = {time_criterion}")
                    
                # Start the opponent modelling when the time-dependency criterion is satisfied and opponent has proposed at least 3 different bids.
                if not self.opp_model_started:
                    if time_criterion > 0.5 and len(np.unique(self.opponent_utility_history_bidded_by_opp)) > 2:
                        self.opp_model_first_step = state.step
                        self.opp_model_started = True
                        if self.verbosity_opponent_modelling > 0:
                            print(f"(Opponent ends={self.opponent_ends}) Opponent modelling started in step {state.step} \
                                    (Rel time = {state.relative_time}) with time criterion {time_criterion}")
            if self.opp_model_started:
                self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy. Change the concession threshold to the curve function when decided
        self.compute_concession(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counteroffer in the bidding_strategy and return it. Update self utility differences.
        bid = self.bidding_strategy(state)
        self.utility_history_bidded_by_self.append(self.ufun(bid))
        if len(self.utility_history_bidded_by_self) > 1:
            self.update_differences(differences=self.differences_bidded_by_self,
                                    utility_history=self.utility_history_bidded_by_self,
                                    max_order=2)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    def get_normalized_advantage(self, utility: float) -> float:
        return (utility - self.ufun.reserved_value)/(self.pareto_outcomes[0][0][0] - self.ufun.reserved_value)

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
        if state.step < ((self.nmi.n_steps - 1) * 0.8):
            self.phase = 1
        elif state.step < self.nmi.n_steps - 1 - self.nr_steps_last_phase:
            self.phase = 2
        else:
            self.phase = 3

    def compute_concession(self, state: SAOState):
        """
        This function determines the acceptance curve point at the current time step.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                and other information about the negotiation (e.g. current step, relative time, etc.).
        """
        m = self.ufun.reserved_value
        t = state.step
        total_steps = self.nmi.n_steps - 1

        if self.phase == 1:
            beta = 6
        elif self.phase == 2 and self.opponent_ends:
            beta = 3
        elif self.phase == 2 or not self.opponent_ends:
            beta = 4.5
        elif self.phase == 3 and self.opponent_ends:
            beta = 0.5
        else:
            beta = 1

        T = self.phases[self.phase]['T']
        M = self.phases[self.phase]['M']
        t_offset = self.phases[self.phase]['t_offset']

        self.concession_threshold = M - ((M - m) * pow((t - t_offset)/T, beta))

        if self.phase < 3:
            self.phases[self.phase + 1]['T'] = total_steps - state.step
            self.phases[self.phase + 1]['M'] = self.concession_threshold
            self.phases[self.phase + 1]['t_offset'] = state.step

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether to accept the offer.

        Args:
            state (SAOState): the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc.).

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer
        offer_utility = float(self.ufun(offer))

        # define two strategies for when opponent has and does not have last bid
        if offer is None:
            return False
        
        # Accept big changes in normalized advantages in our favour
        acceptable_advantage_difference = [.3, .2] # We need a bigger change to be acceptable when we end
        if len(self.utility_history_bidded_by_opp) >1:
            if self.get_normalized_advantage(self.utility_history_bidded_by_opp[-1]) - self.get_normalized_advantage(self.utility_history_bidded_by_opp[-2]) > acceptable_advantage_difference[int(self.opponent_ends)] \
            and self.get_normalized_advantage(self.utility_history_bidded_by_opp[-1]) > 0:
                return True

        if offer_utility >= self.concession_threshold:
            return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
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

        Returns: The counteroffer as Outcome.
        """
        concession_bids = []
        if self.phase == 3 and not self.opponent_ends:
            concession_bids = [bids for bids in self.pareto_outcomes if bids[0][0] > self.concession_threshold and bids[0][1] > (1.25 * self.opponent_reserved_value)]

        if len(concession_bids) == 0:
            # compute all possible bids given the criteria for 'good' bids
            concession_bids = [bids for bids in self.pareto_outcomes if bids[0][0] > self.concession_threshold]

        if len(concession_bids) == 0:
            bid_idx = max(range(len(self.rational_outcomes)), key=lambda x: self.ufun(self.rational_outcomes[x]))
        else:
            bid_idx = max(concession_bids, key=lambda x: x[0][1])[1]
            
        return self.rational_outcomes[bid_idx]

    def update_differences(self, 
                           differences: list[list[float]], 
                           utility_history: list[float], 
                           max_order: int | None = None
                           ) -> None:
        """
        Update the differences until order max_order from the given utility history.
        """
        if self.debug: assert len(utility_history) > 1
        
        m = max_order if max_order is not None and len(utility_history)-1 > max_order else len(utility_history)-1
        for k in range(m):
            # Once per update (at the last step and if max order was not achieved) we create the list with the new order differences.
            if k == len(utility_history)-2: 
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
        m = min([len(self.differences_bidded_by_self[0]), len(self.opponent_differences_bidded_by_opp[0])])
        if self.debug: assert len(self.opponent_differences_bidded_by_opp[0]) == len(self.differences_bidded_by_self[0]) or len(self.opponent_differences_bidded_by_opp[0]) == len(self.differences_bidded_by_self[0])+1
        r_i =  [self.opponent_differences_bidded_by_opp[0][-m:][i]/self.differences_bidded_by_self[0][-m:][i] if not math.isclose(self.differences_bidded_by_self[0][-m:][i], 0) else 0 for i in range(m)]
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
            if self.opponent_rv_upper_bound >= self.opponent_utility_history_bidded_by_opp[0]:
                self.opponent_rv_upper_bound = self.opponent_utility_history_bidded_by_opp[0]-self.opponent_utility_history_bidded_by_opp[0]/self.NR_DETECTING_CELLS
            self.opponent_reserved_value = self.opponent_rv_upper_bound # First guess (if needed)

            # Dividing the detecting region into detecting cells
            detecting_region_length = self.opponent_rv_upper_bound - self.opponent_rv_lower_bound
            for k in range(self.NR_DETECTING_CELLS+1):
                self.detecting_cells_bounds.append(self.opponent_rv_lower_bound+(k/self.NR_DETECTING_CELLS)*detecting_region_length)

        def polynomial_concession(t, reservation_value, beta):            
            return self.opponent_utility_history_bidded_by_opp[0] + (reservation_value - self.opponent_utility_history_bidded_by_opp[0]) * pow(t/self.nmi.n_steps, beta)

        def fitted_beta(reservation_value):
            p_i = np.log([(self.opponent_utility_history_bidded_by_opp[0] - self.opponent_utility_history_bidded_by_opp[t])/(self.opponent_utility_history_bidded_by_opp[0] - reservation_value) 
                          if abs(self.opponent_utility_history_bidded_by_opp[0] - self.opponent_utility_history_bidded_by_opp[t])>1e-3 else 1e-3 for t in range(1, len(self.opponent_utility_history_bidded_by_opp))])
            t_i = np.log([t/self.nmi.n_steps for t in range(1, len(self.opponent_utility_history_bidded_by_opp))])
            beta = np.dot(p_i, t_i)/np.dot(t_i, t_i)
            if beta == float("inf"):
                if self.debug:
                    raise ValueError("Fitted beta computed with nonlinear regression is inf")
                else:
                    return 50
            else:
                return beta

        def fitted_alpha(reservation_value):
            max_utility = max(self.opponent_utility_history_bidded_by_opp)  # maximum utility achieved by the opponent
            current_utility = self.opponent_utility_history_bidded_by_opp[-1]  # current utility, opponent
            current_step = state.step
            # Take the last utility different from the current and the corresponding time step
            for k in range(2, len(self.opponent_utility_history_bidded_by_opp)):
                previous_utility = self.opponent_utility_history_bidded_by_opp[-k]
                if abs(current_utility-previous_utility)>1e-10:
                    previous_step = current_step - k+1
                    break
            alpha =  math.log((max_utility - current_utility)/(max_utility - previous_utility)) / math.log( current_step / previous_step) # equation 8 Zhang et al.
            if alpha !=0:
                if alpha>100:
                    if self.verbosity_opponent_modelling > 1: print("Alpha adjusted to 100")
                    return 100
                else:
                    return alpha
            else:
                if self.verbosity_opponent_modelling > 1: 
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
            fitted_offers = [polynomial_concession(t, reservation_value, beta) for t in range(1, len(self.opponent_utility_history_bidded_by_opp))]
            normalized_fitted_offers = np.array(fitted_offers) - np.mean(fitted_offers)
            normalized_history_offers = np.array(self.opponent_utility_history_bidded_by_opp[1:]) - np.mean(self.opponent_utility_history_bidded_by_opp[1:])
            denominator = math.sqrt(np.dot(normalized_fitted_offers, normalized_fitted_offers) * np.dot(normalized_history_offers, normalized_history_offers))
            if denominator != 0:
                return abs(np.dot(normalized_fitted_offers, normalized_history_offers) / denominator)
            elif self.debug:
                if self.verbosity_opponent_modelling > 1:
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

        if self.debug: assert np.all(np.greater_equal(self.detecting_cells_prob, np.zeros(len(self.detecting_cells_prob))))

        def compute_expected_reserved_value():
            return np.dot(self.detecting_cells_prob, [self.detecting_cells_bounds[k+1] for k in range(len(self.detecting_cells_prob))])

        # Selecting a reservation value from detecting cells probability distribution: the upper bound of the cell with max prob.
        #self.opponent_reserved_value = self.detecting_cells_bounds[np.argmax(self.detecting_cells_prob) + 1]
        self.opponent_reserved_value = compute_expected_reserved_value()
        self.nr_opponent_rv_updates += 1

        if self.verbosity_opponent_modelling > 2 and state.step % 10 == 0:
            print(f"(Opponent ends={self.opponent_ends}) (Rel time={state.relative_time}) \
                  Predicted RV with {self.nr_opponent_rv_updates} updates = {compute_expected_reserved_value()}")
            print(f"Total nr of updates = ")
        if self.verbosity_opponent_modelling > 1 and self.opponent_ends:
            print(f"(Opponent ends={self.opponent_ends}) Predicted reserved value at step {state.step} (rel time={state.relative_time}) = {self.opponent_reserved_value}")
        

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        """ rational_outcomes = self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ] """

# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Group1, small=False, debug=False)
