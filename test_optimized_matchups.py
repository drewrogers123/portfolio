import pandas as pd
import json

# 1. load your optimized parameters
with open("yourfile.json", "r") as f:
    best_params = json.load(f)

# 2. load the rankings
final_ratings = pd.read_csv("yourfile.csv")

hca = best_params.get('home_adv', 65.0)
print(f"=== MATCHUP PREDICTIONS (HCA: {hca:.1f}) ===\n")

# 3. store results for csv
matchup_results = []

# define matchups
matchups = [
    ("TCU", "Ohio St."),
    ("Idaho", "Houston")
]

# 4. process each matchup and collect results
for team_a, team_b in matchups:
    try:
        # extract ratings
        r_a = final_ratings.loc[final_ratings['team'] == team_a, 'rating'].values[0]
        r_b = final_ratings.loc[final_ratings['team'] == team_b, 'rating'].values[0]
        
        # calculate probabilities and spread
        hca_effect = best_params.get('home_adv', 65.0)
        prob_a = 1.0 / (1.0 + 10.0 ** ((r_b + hca_effect - r_a) / 400.0))
        prob_b = 1.0 - prob_a
        
        div = best_params.get('spread_divisor', 17.5)
        elo_diff = (r_a - (r_b + hca_effect))
        spread = elo_diff / div
        
        # determine favorite
        favorite = team_a if spread > 0 else team_b
        
        # store results
        matchup_results.append({
            'team_a': team_a,
            'team_b': team_b,
            'team_a_rating': r_a,
            'team_b_rating': r_b,
            'team_a_win_prob': prob_a,
            'team_b_win_prob': prob_b,
            'spread': spread,
            'favorite': favorite
        })
        
        # print individual matchup
        print(f"##############################")
        print(f"{team_a} @ {team_b}")
        print(f"##############################")
        print(f"Win Probability:  {team_a} {prob_a:.1%}")
        print(f"Projected Line:   {team_a} {spread:+.1f}")
        print(f"Projected Favorite: {favorite}")
        print(f"##############################\n")
        
    except IndexError:
        print(f"Error: Could not find ratings for {team_a} or {team_b}")

# 5. save to csv        
matchups_df = pd.DataFrame(matchup_results)
matchups_df.to_csv("results.csv", index=False)
print(f"Saved {len(matchup_results)} matchup predictions to results.csv")
