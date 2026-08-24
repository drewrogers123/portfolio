import math
import pandas as pd


def expected_score(r_i, r_j):
    diff = max(-1000, min(1000, r_j - r_i))
    return 1.0 / (1.0 + 10.0 ** (diff / 400.0))

def mov_multiplier(margin, rating_diff, mov_k, mov_divisor):
    if margin <= 0: return 1.0
    # fivethirtyeight standard: denominator = (0.001 * elo_diff) + 2.2
    denominator = (mov_divisor * rating_diff) + mov_k
    # safety check to prevent division by zero or negative values
    if denominator <= 0.0001: denominator = 0.0001 
    return math.log(margin + 1) * (mov_k / denominator)

def compute_team_elo(long_df, priors=None, **params):
    # parameter extraction (with defaults)
    base_k = params.get('base_k', 24.0)
    start_rating = params.get('start_rating', 1500.0)
    home_adv = params.get('home_adv', 65.0)
    k_threshold_1 = params.get('k_threshold_1', 8)
    k_mult_1 = params.get('k_mult_1', 2.0)
    k_threshold_2 = params.get('k_threshold_2', 15)
    k_mult_2 = params.get('k_mult_2', 1.5)
    mov_k = params.get('mov_k', 2.2)
    mov_divisor = params.get('mov_divisor', 0.001)
    regress_weight = params.get('regress_weight', 0.75)

    ratings = {}
    games_played = {}
    history_rows = []
    
    for seq, group in long_df.groupby("game_seq", sort=False):
        if len(group) != 2: continue
        
        away_data, home_data = group.iloc[0], group.iloc[1]
        t_a, t_h = away_data["Team"], home_data["Team"]
        
        for team in [t_a, t_h]:
            if team not in ratings:
                if priors and team in priors:
                    # Regression to the mean
                    ratings[team] = (priors[team] * regress_weight) + (start_rating * (1 - regress_weight))
                else:
                    ratings[team] = start_rating
                games_played[team] = 0
        
        r_a_base, r_h_base = ratings[t_a], ratings[t_h]
        hca_effect = 0 if home_data["is_neutral"] else home_adv
        
        # expected probs
        exp_a = expected_score(r_a_base, r_h_base + hca_effect)
        s_a, s_h = away_data["Score"], home_data["Score"]
        act_a = 1.0 if s_a > s_h else (0.0 if s_a < s_h else 0.5)
        
        margin = abs(s_a - s_h)
        # rating difference: winner - loser
        if s_a > s_h:
            r_diff = r_a_base - (r_h_base + hca_effect)
        else:
            r_diff = (r_h_base + hca_effect) - r_a_base
            
        multiplier = mov_multiplier(margin, r_diff, mov_k, mov_divisor)

        # k-factor logic
        def get_k(count):
            if count < k_threshold_1: return base_k * k_mult_1
            if count < k_threshold_2: return base_k * k_mult_2
            return base_k

        k_a, k_h = get_k(games_played[t_a]), get_k(games_played[t_h])
        
        # update ratings
        delta = multiplier * (act_a - exp_a)
        ratings[t_a] += k_a * delta
        ratings[t_h] -= k_h * delta 
        
        games_played[t_a] += 1
        games_played[t_h] += 1          
        
        history_rows.append({
            "game_seq": seq, "actual": act_a, "expected": exp_a, 
            "team": t_a, "opp": t_h, "date": away_data.get('date', None)
        })
            
    final_df = pd.DataFrame([{"team": t, "rating": r} for t, r in ratings.items()])
    return final_df.sort_values("rating", ascending=False), pd.DataFrame(history_rows)
