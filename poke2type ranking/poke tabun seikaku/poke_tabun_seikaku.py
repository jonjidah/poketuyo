import numpy as np
import itertools
import matplotlib.pyplot as plt

# Define all 18 Pokémon single types
single_types = [
    'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting',
    'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
    'Dragon', 'Dark', 'Steel', 'Fairy'
]

# Define the type effectiveness chart
# Rows: Attacking types
# Columns: Defending types
# Values: Effectiveness multiplier
type_chart = {
    'Normal':     {'Rock': 0.5, 'Ghost': 0, 'Steel': 0.5},
    'Fire':       {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2},
    'Water':      {'Fire': 2, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2, 'Rock': 2, 'Dragon': 0.5},
    'Electric':   {'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0, 'Flying': 2, 'Dragon': 0.5},
    'Grass':      {'Fire': 0.5, 'Water': 2, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2, 'Dragon': 0.5, 'Steel': 0.5},
    'Ice':        {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 0.5, 'Ground': 2, 'Flying': 2, 'Dragon': 2, 'Steel': 0.5},
    'Fighting':   {'Normal': 2, 'Ice': 2, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dark': 2, 'Steel': 2, 'Fairy': 0.5},
    'Poison':     {'Grass': 2, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0, 'Fairy': 2},
    'Ground':     {'Fire': 2, 'Electric': 2, 'Grass': 0.5, 'Poison': 2, 'Flying': 0, 'Bug': 0.5, 'Rock': 2, 'Steel': 2},
    'Flying':     {'Electric': 0.5, 'Grass': 2, 'Fighting': 2, 'Bug': 2, 'Rock': 0.5, 'Steel': 0.5},
    'Psychic':    {'Fighting': 2, 'Poison': 2, 'Psychic': 0.5, 'Dark': 0, 'Steel': 0.5},
    'Bug':        {'Fire': 0.5, 'Grass': 2, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2, 'Ghost': 0.5, 'Dark': 2, 'Steel': 0.5, 'Fairy': 0.5},
    'Rock':       {'Fire': 2, 'Ice': 2, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2, 'Bug': 2, 'Steel': 0.5},
    'Ghost':      {'Normal': 0, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5},
    'Dragon':     {'Dragon': 2, 'Steel': 0.5, 'Fairy': 0},
    'Dark':       {'Fighting': 0.5, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5, 'Fairy': 0.5},
    'Steel':      {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2, 'Rock': 2, 'Steel': 0.5, 'Fairy': 2},
    'Fairy':      {'Fire': 0.5, 'Fighting': 2, 'Poison': 0.5, 'Dragon': 2, 'Dark': 2, 'Steel': 0.5},
}

def get_effectiveness(attacking_type, defending_types):
    """
    Calculate the effectiveness multiplier of an attacking type against a defending type or dual types.
    For dual types, the multipliers are multiplied together.
    """
    multiplier = 1.0
    for dtype in defending_types:
        if dtype in type_chart.get(attacking_type, {}):
            multiplier *= type_chart[attacking_type][dtype]
        else:
            multiplier *= 1.0
    return multiplier

# Generate all dual types (unordered pairs)
dual_type_pairs = list(itertools.combinations(single_types, 2))

# Combine single and dual types
all_types = single_types + [f"{t1}/{t2}" for t1, t2 in dual_type_pairs]

# Total number of types
num_types = len(all_types)  # 18 + 153 = 171

# Create a mapping from type name to index
type_to_index = {type_name: idx for idx, type_name in enumerate(all_types)}
index_to_type = {idx: type_name for type_name, idx in type_to_index.items()}

# Initialize the type effectiveness matrix (171 x 171)
# Rows: Attacking types
# Columns: Defending types
# Values: Effectiveness multiplier

# Precompute effectiveness for all type pairs
# For dual-type attackers, use the maximum effectiveness of the two types
effectiveness_matrix = np.ones((num_types, num_types))

for atk_idx, atk_type in enumerate(all_types):
    # Split dual types
    atk_components = atk_type.split('/')
    for def_idx, def_type in enumerate(all_types):
        def_components = def_type.split('/')
        if len(atk_components) == 1:
            # Single-type attacker
            eff = get_effectiveness(atk_components[0], def_components)
        else:
            # Dual-type attacker: choose the maximum effectiveness between the two types
            eff1 = get_effectiveness(atk_components[0], def_components)
            eff2 = get_effectiveness(atk_components[1], def_components)
            eff = max(eff1, eff2)
        effectiveness_matrix[atk_idx, def_idx] = eff

# Initialize type distribution equally
type_distribution = np.ones(num_types) / num_types

# Simulation parameters
iterations = 10000
learning_rate = 0.1

# Defense weight factor (greater than 1 to emphasize defense)
defense_weight = 2.0

# Run the simulation
for _ in range(iterations):
    # Calculate attack effectiveness: each type's total effectiveness against the current distribution
    attack_effectiveness = effectiveness_matrix.dot(type_distribution)
    
    # Calculate defense effectiveness:
    # For defending types, calculate the total damage they receive from all attacking types
    damage_received = effectiveness_matrix.T.dot(type_distribution)
    # To avoid division by zero, add a small epsilon
    defense_effectiveness = 1 / (damage_received + 1e-6)
    
    # Apply defense weight
    defense_effectiveness *= defense_weight
    
    # Calculate new distribution based on attack and defense
    new_distribution = attack_effectiveness + defense_effectiveness
    
    # Normalize the distribution to sum to 1
    new_distribution /= new_distribution.sum()
    
    # Update the distribution with a learning rate for stability
    type_distribution = (1 - learning_rate) * type_distribution + learning_rate * new_distribution

# Create a list of tuples (type, score)
type_scores = list(zip(all_types, type_distribution))

# Sort the types based on their scores in descending order
type_scores.sort(key=lambda x: x[1], reverse=True)

# Assign ranks
ranked_types = [(rank + 1, type_name, score) for rank, (type_name, score) in enumerate(type_scores)]

# Print all type rankings
print("\nFinal Pokémon Type Rankings (Defense Weighted):\n")
print(f"{'Rank':<5} {'Type':<20} {'Score':<10}")
print("-" * 40)
for rank, type_name, score in ranked_types:
    print(f"{rank:<5} {type_name:<20} {score:<10.6f}")

# Extract top 20 and bottom 20 types
top_20 = ranked_types[:20]
bottom_20 = ranked_types[-20:]

# Function to add rank and score labels
def add_labels(ax, data, is_top=True):
    for i, (rank, type_name, score) in enumerate(data):
        label = f"{rank}. {score:.6f}"
        ax.text(score, i, label, va='center', fontsize=8)

# Plotting Top 20 Types
top_types = [item[1] for item in top_20]
top_scores = [item[2] for item in top_20]
y_pos = np.arange(len(top_types))

plt.figure(figsize=(12, 10))
bars = plt.barh(y_pos, top_scores, color='skyblue')
plt.yticks(y_pos, top_types)
plt.gca().invert_yaxis()  # Highest score at the top
plt.xlabel('Score')
plt.title('Top 20 Pokémon Types (Defense Weighted)')
add_labels(plt.gca(), top_20, is_top=True)
plt.tight_layout()
plt.show()

# Plotting Bottom 20 Types
bottom_types = [item[1] for item in bottom_20]
bottom_scores = [item[2] for item in bottom_20]
y_pos = np.arange(len(bottom_types))

plt.figure(figsize=(12, 10))
bars = plt.barh(y_pos, bottom_scores, color='salmon')
plt.yticks(y_pos, bottom_types)
plt.gca().invert_yaxis()  # Highest (in bottom 20) score at the top
plt.xlabel('Score')
plt.title('Bottom 20 Pokémon Types (Defense Weighted)')
add_labels(plt.gca(), bottom_20, is_top=False)
plt.tight_layout()
plt.show()
