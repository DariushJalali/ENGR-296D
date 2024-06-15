#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to load and process player data
def load_player_data(filename, player_name):
    df = pd.read_csv(filename)
    df_player = df[df['Observed Player Name'] == player_name]
    win_count = df_player[df_player['Observed Player W/L'] == 'W'].shape[0]
    loss_count = df_player[df_player['Observed Player W/L'] == 'L'].shape[0]
    return win_count, loss_count, df_player

# Function to plot pie chart
def plot_pie_chart(win_count, loss_count, player_name):
    labels = 'Wins', 'Losses'
    sizes = [win_count, loss_count]
    colors = ['gold', 'lightcoral']
    explode = (0.1, 0)  # explode 1st slice (Wins)
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Win/Loss Distribution for {player_name}')
    plt.legend(labels=[f'Wins: {win_count}, Losses: {loss_count}'], loc="best")
    plt.show()

# Load and process data for each player
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal'}
]

for player in players:
    win_count, loss_count, df_player = load_player_data(player['filename'], player['name'])
    plot_pie_chart(win_count, loss_count, player['name'])


# In[9]:


# Function to load and preprocess data for each player
def load_and_preprocess_data(filename, player_name):
    df = pd.read_csv(filename)
    # Replace non-standard hyphens with standard hyphens in the Date column
    df['Date'] = df['Date'].str.replace('‑', '-')
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', errors='coerce')
    # Filter for matches of the player
    df = df[df['Observed Player Name'] == player_name]
    # Set the date as the index for resampling
    df.set_index('Date', inplace=True)
    return df

# Calculate end-of-year ranking, average ranking for the last 15 years, and career duration
def calculate_ranking_stats(df, player_name, custom_end_year=None):
    # Exclude the specified year if provided
    if custom_end_year:
        df = df[df.index.year <= custom_end_year]
    # Resample to yearly frequency and take the average ranking for each year
    yearly_ranking = df['Rk'].resample('Y').mean()
    # Calculate the average ranking over the past 15 years
    last_15_years = yearly_ranking.tail(15)
    average_ranking_last_15_years = last_15_years.mean()
    # Find the first date when the player reached number 1
    first_number_1_date = df[df['Rk'] == 1].index.min()
    career_start_date = df.index.min()
    career_end_date = df.index.max()
    career_duration = (career_end_date - career_start_date).days / 365
    days_to_number_1 = (first_number_1_date - career_start_date).days if first_number_1_date else None
    return yearly_ranking, average_ranking_last_15_years, days_to_number_1, last_15_years, career_duration

# Players data
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic', 'custom_end_year': None},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer', 'custom_end_year': None},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal', 'custom_end_year': 2022}
]

# Process data for each player and calculate stats
rankings_data = []
for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    yearly_ranking, avg_ranking_15_years, days_to_number_1, last_15_years, career_duration = calculate_ranking_stats(df, player['name'], player['custom_end_year'])
    rankings_data.append({'name': player['name'], 'yearly_ranking': yearly_ranking, 'avg_ranking_15_years': avg_ranking_15_years, 'days_to_number_1': days_to_number_1, 'career_duration': career_duration})

# Print average ranking over the last 15 years for each player
print("Average Rankings Over the Last 15 Years:")
for data in rankings_data:
    print(f"{data['name']}: {data['avg_ranking_15_years']:.2f}")

# Determine the quickest player to reach number 1
rankings_data = [data for data in rankings_data if data['days_to_number_1'] is not None]
rankings_data.sort(key=lambda x: x['days_to_number_1'])

print("\nQuickest to Reach Number 1 from Career Start Date (in years):")
for data in rankings_data:
    years_to_number_1 = data['days_to_number_1'] / 365
    print(f"{data['name']}: {years_to_number_1:.2f} years")

# Print the duration of each player's career
print("\nDuration of Each Player's Career (in years):")
for data in rankings_data:
    print(f"{data['name']}: {data['career_duration']:.2f} years")


# In[10]:


# Function to calculate the longest win streak
def calculate_longest_win_streak(df):
    max_streak = 0
    current_streak = 0
    
    for index, row in df.iterrows():
        if row['Observed Player W/L'] == 'W':
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
            
    return max_streak

# Players data
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal'}
]

# Calculate the longest win streak for each player
longest_win_streaks = []

for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    longest_win_streak = calculate_longest_win_streak(df)
    longest_win_streaks.append({'name': player['name'], 'longest_win_streak': longest_win_streak})

# Print the longest win streak for each player
print("Longest Win Streaks:")
for player in longest_win_streaks:
    print(f"{player['name']}: {player['longest_win_streak']} consecutive wins")


# In[13]:


# Function to load and filter player data by surface, excluding 'Carpet'
def load_surface_data(filename):
    # Load the CSV file
    df = pd.read_csv(filename)
    
    # Filter for relevant columns and exclude 'Carpet' surface
    df = df[['Surface', 'Observed Player W/L']].dropna()
    df = df[df['Surface'] != 'Carpet']
    
    # Count wins and losses by surface
    surface_summary = df.groupby(['Surface', 'Observed Player W/L']).size().unstack(fill_value=0)
    
    return surface_summary

# Function to plot surface performance for a player
def plot_surface_performance(surface_summary, player_name):
    # Prepare data for plotting
    surfaces = surface_summary.index.tolist()
    wins = surface_summary.get('W', 0).tolist()
    losses = surface_summary.get('L', 0).tolist()
    
    # Create the bar chart for Wins and Losses
    fig, ax = plt.subplots(figsize=(10, 6))
    index = range(len(surfaces))
    bar_width = 0.35
    
    # Plot bars with green for wins and red for losses
    bars1 = ax.bar(index, wins, bar_width, label='Wins', color='green')
    bars2 = ax.bar([p + bar_width for p in index], losses, bar_width, label='Losses', color='red')
    
    # Add number annotations above each bar
    for i, (win, loss) in enumerate(zip(wins, losses)):
        ax.annotate(f'{win}', xy=(index[i], win), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        ax.annotate(f'{loss}', xy=(index[i] + bar_width, loss), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    ax.set_xlabel('Surface')
    ax.set_ylabel('Number of Matches')
    ax.set_title(f'{player_name} Performance by Surface')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(surfaces)
    
    # Create legend for win/loss
    color_legend = ax.legend([bars1, bars2], ['Wins', 'Losses'], loc='upper left', bbox_to_anchor=(1, 0.8))
    ax.add_artist(color_legend)
    
    plt.tight_layout()
    plt.show()

# Calculate overall win percentage across all surfaces for a player
def calculate_overall_win_percentage(surface_summary):
    wins = surface_summary.get('W', 0).sum()
    losses = surface_summary.get('L', 0).sum()
    total_matches = wins + losses
    win_percentage = (wins / total_matches) * 100 if total_matches > 0 else 0
    return win_percentage

# List of players with their filenames and colors
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic', 'color': 'blue'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer', 'color': 'green'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal', 'color': 'red'}
]

# Process and plot data for each player
overall_win_percentages = []

for player in players:
    surface_summary = load_surface_data(player['filename'])
    win_percentage = calculate_overall_win_percentage(surface_summary)
    overall_win_percentages.append({'name': player['name'], 'win_percentage': win_percentage, 'color': player['color']})
    plot_surface_performance(surface_summary, player['name'])

# Determine the player with the best overall win percentage
best_player = max(overall_win_percentages, key=lambda x: x['win_percentage'])

# Print summary of the best player
print(f"The player with the best overall win percentage across all surfaces is {best_player['name']} with a win percentage of {best_player['win_percentage']:.2f}%.")

# Create a summary plot for overall win percentages
fig, ax = plt.subplots(figsize=(10, 6))
names = [p['name'] for p in overall_win_percentages]
win_percentages = [p['win_percentage'] for p in overall_win_percentages]
colors = [p['color'] for p in overall_win_percentages]
bars = ax.bar(names, win_percentages, color=colors)

# Add number annotations above each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

ax.set_xlabel('Player')
ax.set_ylabel('Win Percentage')
ax.set_title('Overall Win Percentage Across All Surfaces')
ax.set_ylim(0, 100)  # Set y-axis limit from 0 to 100

plt.tight_layout()
plt.show()


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to load and filter player data for Grand Slam and Masters finals
def load_finals_data(filename, player_name):
 
    
    # Define Grand Slam and historical Masters tournaments
    grand_slams_and_masters = [
        'Australian Open', 'Roland Garros', 'Wimbledon', 'US Open',
        'Indian Wells Masters', 'Miami Masters', 'Monte Carlo Masters',
        'Madrid Masters', 'Rome Masters', 'Canada Masters', 'Cincinnati Masters',
        'Shanghai Masters', 'Paris Masters', 'Hamburg Masters', 'Madrid Open',
        'Germany Masters', 'Stuttgart Masters'
    ]
    
    # Filter matches where the round is 'F' and tournament is a Grand Slam or Masters
    df_finals = df[(df['Rd'] == 'F') & (df['Tournament'].isin(grand_slams_and_masters))]
    
    # Count wins and losses in finals
    win_count = df_finals[df_finals['Observed Player W/L'] == 'W'].shape[0]
    loss_count = df_finals[df_finals['Observed Player W/L'] == 'L'].shape[0]
    
    return win_count, loss_count

# Function to plot bar chart for finals record
def plot_finals_record(win_count, loss_count, player_name):
    # Calculate total matches and percentages
    total_matches = win_count + loss_count
    win_percentage = f"{(win_count / total_matches * 100):.1f}%" if total_matches > 0 else "0%"
    loss_percentage = f"{(loss_count / total_matches * 100):.1f}%" if total_matches > 0 else "0%"
    
    # Labels and values for the bar chart
    labels = ['Wins', 'Losses']
    values = [win_count, loss_count]
    legend_labels = [f'Wins: {win_count} ({win_percentage})', f'Losses: {loss_count} ({loss_percentage})']
    
    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=['green', 'red'])
    
    # Set legend to explain the colors
    ax.legend(bars, legend_labels)
    
    # Set labels and title
    plt.xlabel('Outcome')
    plt.ylabel('Number of Matches')
    plt.title(f'Win/Loss Distribution for {player_name} in Grand Slam and Masters Finals')
    
    # Show the plot
    plt.show()

# Load and process data for each player
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal'}
]

# Iterate through each player, process their data, and plot the results
for player in players:
    win_count, loss_count = load_finals_data(player['filename'], player['name'])
    plot_finals_record(win_count, loss_count, player['name'])


# In[15]:


# Function to load and filter player data against top 10 opponents
def load_top10_data(filename, player_name):
    df = pd.read_csv(filename)
    df_top10_matches = df[(df['vRk'] <= 10) & ((df['Observed Player Name'] == player_name) | (df['Opponent Player Name'] == player_name))]
    win_count = df_top10_matches[df_top10_matches['Observed Player W/L'] == 'W'].shape[0]
    loss_count = df_top10_matches[df_top10_matches['Observed Player W/L'] == 'L'].shape[0]
    return win_count, loss_count, df_top10_matches

# Function to plot pie chart
def plot_pie_chart_top10(win_count, loss_count, player_name):
    if win_count + loss_count > 0:
        win_percentage = (win_count / (win_count + loss_count)) * 100
        labels = 'Wins', 'Losses'
        sizes = [win_count, loss_count]
        colors = ['green', 'red']
        explode = (0.1, 0)  # explode 1st slice (Wins)

        plt.figure(figsize=(8, 6))
        wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Win/Loss Distribution for {player_name} Against Top 10 Ranked Opponents')
        
        # Custom legend
        plt.legend(wedges, [f'Wins: {win_count} ({win_percentage:.2f}%)', f'Losses: {loss_count} ({100 - win_percentage:.2f}%)'], 
                   title=f'{player_name} Performance', loc="best")
        
        plt.show()

        print(f"{player_name}'s win percentage against top 10 ranked opponents is {win_percentage:.2f}%")
    else:
        print(f"No matches found against top 10 ranked opponents for {player_name} or no wins/losses data available.")

# Load and process data for each player
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal'}
]

for player in players:
    win_count, loss_count, df_top10_matches = load_top10_data(player['filename'], player['name'])
    plot_pie_chart_top10(win_count, loss_count, player['name'])


# In[14]:


# Function to calculate head-to-head results by surface
def calculate_head_to_head_surface(df, opponent):
    surfaces = ['Hard', 'Clay', 'Grass']
    head_to_head_surface = {surface: {'wins': 0, 'losses': 0} for surface in surfaces}
    for surface in surfaces:
        matches_on_surface = df[(df['Opponent Player Name'] == opponent) & (df['Surface'] == surface)]
        head_to_head_surface[surface]['wins'] = matches_on_surface[matches_on_surface['Observed Player W/L'] == 'W'].shape[0]
        head_to_head_surface[surface]['losses'] = matches_on_surface[matches_on_surface['Observed Player W/L'] == 'L'].shape[0]
    return head_to_head_surface

# Players data
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic', 'opponents': ['Roger Federer', 'Rafael Nadal']},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer', 'opponents': ['Novak Djokovic', 'Rafael Nadal']},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal', 'opponents': ['Novak Djokovic', 'Roger Federer']}
]

# Process data for each player and calculate head-to-head results by surface
head_to_head_surface_data = {}
for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    head_to_head_surface = {}
    for opponent in player['opponents']:
        head_to_head_surface[opponent] = calculate_head_to_head_surface(df, opponent)
    head_to_head_surface_data[player['name']] = head_to_head_surface

# Function to plot pie chart for head-to-head results by surface
def plot_pie_chart_surface(player_name, opponent, head_to_head_surface):
    labels = []
    sizes = []
    colors = ['green', 'red']
    
    total_wins = sum(results['wins'] for results in head_to_head_surface.values())
    total_losses = sum(results['losses'] for results in head_to_head_surface.values())
    
    labels.append(f'Wins vs {opponent}: {total_wins}')
    labels.append(f'Losses vs {opponent}: {total_losses}')
    sizes.extend([total_wins, total_losses])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Head-to-Head Results for {player_name} vs {opponent}')
    plt.show()

# Plot pie charts for each head-to-head matchup by surface
matchups = [
    ('Novak Djokovic', 'Roger Federer'),
    ('Novak Djokovic', 'Rafael Nadal'),
    ('Rafael Nadal', 'Roger Federer')
]

for player, opponent in matchups:
    plot_pie_chart_surface(player, opponent, head_to_head_surface_data[player][opponent])

# Calculate overall win percentages
overall_win_percentages = {}
for player_name, head_to_head_surface in head_to_head_surface_data.items():
    total_wins = sum(sum(surface_data['wins'] for surface_data in head_to_head_surface[opponent].values()) for opponent in head_to_head_surface)
    total_matches = total_wins + sum(sum(surface_data['losses'] for surface_data in head_to_head_surface[opponent].values()) for opponent in head_to_head_surface)
    win_percentage = (total_wins / total_matches) * 100 if total_matches > 0 else 0
    overall_win_percentages[player_name] = win_percentage

# Determine the player with the highest overall win percentage
best_player = max(overall_win_percentages, key=overall_win_percentages.get)
print(f"The player with the highest overall win percentage against the other two is {best_player} with a win percentage of {overall_win_percentages[best_player]:.2f}%.")

# Print the overall win percentages for each player
print("\nOverall Win Percentages Against Each Other:")
for player, win_percentage in overall_win_percentages.items():
    print(f"{player}: {win_percentage:.2f}%")

# Print surface breakdown for each player
print("\nSurface Breakdown:")
for player_name, head_to_head_surface in head_to_head_surface_data.items():
    if player_name == 'Novak Djokovic':
        print(f"\n{player_name}:")
        for opponent in ['Roger Federer', 'Rafael Nadal']:
            surfaces = head_to_head_surface[opponent]
            print(f"  Against {opponent}:")
            for surface, results in surfaces.items():
                print(f"    {surface}: {results['wins']}W - {results['losses']}L")
    elif player_name == 'Rafael Nadal':
        print(f"\n{player_name}:")
        opponent = 'Roger Federer'
        surfaces = head_to_head_surface[opponent]
        print(f"  Against {opponent}:")
        for surface, results in surfaces.items():
            print(f"    {surface}: {results['wins']}W - {results['losses']}L")


# In[16]:


# Function to calculate resilience metrics
def calculate_resilience_metrics(df, player_name):
    # Function to determine if the player won after losing the first set
    def won_after_losing_first_set(score, result):
        sets = score.split(' ')
        if result == 'W' and len(sets) > 1 and not sets[0].startswith('6-'):
            return True
        return False

    # Function to determine if the player won after being two sets to none down in a Grand Slam
    def won_after_being_two_sets_down(score, result, tournament):
        sets = score.split(' ')
        if result == 'W' and tournament in ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open']:
            return len(sets) >= 4 and not sets[0].startswith('6-') and not sets[1].startswith('6-')
        return False

    # Function to determine if the match was longer than two hours
    def is_long_duration_match(time, result):
        try:
            duration = pd.to_timedelta(str(time) + ':00')
            return duration > pd.Timedelta(hours=2) and result == 'W'
        except ValueError:
            return False

    # Function to determine if the match was a five-set or three-set victory
    def is_five_set_or_three_set_victory(score, result):
        sets = score.split(' ')
        return result == 'W' and len(sets) in [3, 5]

    # Apply the functions to determine resilience metrics
    df['Won After Losing First Set'] = df.apply(lambda row: won_after_losing_first_set(row['Score'], row['Observed Player W/L']), axis=1)
    df['Won After Being Two Sets Down'] = df.apply(lambda row: won_after_being_two_sets_down(row['Score'], row['Observed Player W/L'], row['Tournament']), axis=1)
    df['Won Long Duration Match'] = df.apply(lambda row: is_long_duration_match(row['Time'], row['Observed Player W/L']), axis=1)
    df['Five Set or Three Set Victory'] = df.apply(lambda row: is_five_set_or_three_set_victory(row['Score'], row['Observed Player W/L']), axis=1)

    # Calculate the resilience metrics
    metrics = {
        'Matches Won After Losing First Set': df['Won After Losing First Set'].sum(),
        'Matches Won After Being Two Sets Down': df['Won After Being Two Sets Down'].sum(),
        'Matches Won Longer Than Two Hours': df['Won Long Duration Match'].sum(),
        'Five Set or Three Set Victories': df['Five Set or Three Set Victory'].sum()
    }

    return metrics

# Function to calculate the total score as a percentage based on resilience metrics
def calculate_total_score(metrics, max_scores):
    total_score = (
        (metrics['Matches Won After Losing First Set'] / max_scores['Matches Won After Losing First Set']) * 100 +
        (metrics['Matches Won After Being Two Sets Down'] / max_scores['Matches Won After Being Two Sets Down']) * 100 +
        (metrics['Matches Won Longer Than Two Hours'] / max_scores['Matches Won Longer Than Two Hours']) * 100 +
        (metrics['Five Set or Three Set Victories'] / max_scores['Five Set or Three Set Victories']) * 100
    ) / 4
    return total_score

# Players data
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal'}
]

# Calculate max scores for each metric
max_scores = {
    'Matches Won After Losing First Set': 0,
    'Matches Won After Being Two Sets Down': 0,
    'Matches Won Longer Than Two Hours': 0,
    'Five Set or Three Set Victories': 0
}

# Process data for each player to calculate max scores
for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    metrics = calculate_resilience_metrics(df, player['name'])
    for metric in max_scores:
        if metrics[metric] > max_scores[metric]:
            max_scores[metric] = metrics[metric]

# Process data for each player and calculate resilience metrics and scores
player_scores = []
for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    metrics = calculate_resilience_metrics(df, player['name'])
    total_score = calculate_total_score(metrics, max_scores)
    player_scores.append({'name': player['name'], 'metrics': metrics, 'score': total_score})

# Print the scores for each player
print("Resilience Scores for Each Player:")
for player_score in player_scores:
    print(f"\n{player_score['name']}:")
    for metric, value in player_score['metrics'].items():
        print(f"  {metric}: {value}")
    print(f"  Total Score: {player_score['score']:.2f}%")

# Determine the player with the highest total score
best_player = max(player_scores, key=lambda x: x['score'])

# Print summary of the best player
print(f"\nThe player with the highest resilience score is {best_player['name']} with a score of {best_player['score']:.2f}%.")


# In[17]:


# Function to determine if the player won the deciding set
def won_deciding_set(score, result):
    sets = score.split(' ')
    return (len(sets) == 3 or len(sets) == 5) and result == 'W'

# Function to determine if there was a tiebreak in the match
def tiebreaks_played(score):
    return any('7-' in s or '-7' in s for s in score.split(' '))

# Function to safely evaluate fractions
def safe_eval(fraction):
    try:
        return eval(fraction)
    except ZeroDivisionError:
        return float('nan')

# Function to calculate clutchness metrics
def calculate_clutchness_metrics(df):
    # Deciding set wins
    df['Won Deciding Set'] = df.apply(lambda row: won_deciding_set(row['Score'], row['Observed Player W/L']), axis=1)
    # Tiebreaks played
    df['Tiebreak Played'] = df['Score'].apply(tiebreaks_played)
    
    # Filter matches where the player won the deciding set and tiebreaks
    won_deciding_set_count = df['Won Deciding Set'].sum()
    tiebreak_played_count = df['Tiebreak Played'].sum()
    tiebreak_won_count = df[(df['Tiebreak Played'] == True) & (df['Observed Player W/L'] == 'W')].shape[0]

    # Calculate win rates
    total_matches = df.shape[0]
    deciding_set_win_rate = (won_deciding_set_count / total_matches) * 100
    tiebreak_win_rate = (tiebreak_won_count / tiebreak_played_count) * 100 if tiebreak_played_count > 0 else 0

    # Convert BPSvd to numeric values
    def convert_bpsvd(x):
        if isinstance(x, str) and '/' in x:
            return safe_eval(x)
        return float('nan')

    df['BPSvd'] = df['BPSvd'].apply(convert_bpsvd)
    bpsvd_ratio = df['BPSvd'].mean()
    
    return deciding_set_win_rate, tiebreak_win_rate, bpsvd_ratio

# Function to calculate total clutchness score
def calculate_total_clutchness_score(deciding_set_win_rate, tiebreak_win_rate, bpsvd_ratio, max_values):
    # Define weights for each metric
    weight_deciding_set = 0.4
    weight_tiebreak = 0.4
    weight_bpsvd = 0.2
    
    # Normalize the metrics, ensuring no division by zero
    deciding_set_win_rate_normalized = deciding_set_win_rate / max_values['deciding_set_win_rate'] if max_values['deciding_set_win_rate'] else 0
    tiebreak_win_rate_normalized = tiebreak_win_rate / max_values['tiebreak_win_rate'] if max_values['tiebreak_win_rate'] else 0
    bpsvd_ratio_normalized = bpsvd_ratio / max_values['bpsvd_ratio'] if max_values['bpsvd_ratio'] else 0
    
    # Calculate the total clutchness score
    total_score = (deciding_set_win_rate_normalized * weight_deciding_set) + (tiebreak_win_rate_normalized * weight_tiebreak) + (bpsvd_ratio_normalized * weight_bpsvd)
    return total_score

# Players data
players = [
    {'filename': 'NovakDjokovic.csv', 'name': 'Novak Djokovic'},
    {'filename': 'RogerFederer.csv', 'name': 'Roger Federer'},
    {'filename': 'RafaelNadal.csv', 'name': 'Rafael Nadal'}
]

# Calculate max values for normalization
max_values = {'deciding_set_win_rate': 0, 'tiebreak_win_rate': 0, 'bpsvd_ratio': 0}

for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    deciding_set_win_rate, tiebreak_win_rate, bpsvd_ratio = calculate_clutchness_metrics(df)
    if deciding_set_win_rate > max_values['deciding_set_win_rate']:
        max_values['deciding_set_win_rate'] = deciding_set_win_rate
    if tiebreak_win_rate > max_values['tiebreak_win_rate']:
        max_values['tiebreak_win_rate'] = tiebreak_win_rate
    if bpsvd_ratio > max_values['bpsvd_ratio']:
        max_values['bpsvd_ratio'] = bpsvd_ratio

# Process data for each player and calculate clutchness metrics and score
player_clutchness_scores = []
for player in players:
    df = load_and_preprocess_data(player['filename'], player['name'])
    deciding_set_win_rate, tiebreak_win_rate, bpsvd_ratio = calculate_clutchness_metrics(df)
    total_clutchness_score = calculate_total_clutchness_score(deciding_set_win_rate, tiebreak_win_rate, bpsvd_ratio, max_values)
    player_clutchness_scores.append({'name': player['name'], 'deciding_set_win_rate': deciding_set_win_rate, 'tiebreak_win_rate': tiebreak_win_rate, 'bpsvd_ratio': bpsvd_ratio, 'clutchness_score': total_clutchness_score})

# Determine the player with the highest clutchness score
best_player_clutchness = max(player_clutchness_scores, key=lambda x: x['clutchness_score'])

# Print clutchness scores and metrics for each player
print("Clutchness Scores and Metrics:")
for player in player_clutchness_scores:
    print(f"{player['name']}:")
    print(f"  Deciding Set Win Rate: {player['deciding_set_win_rate']:.2f}%")
    print(f"  Tiebreak Win Rate: {player['tiebreak_win_rate']:.2f}%")
    print(f"  BPSvd Ratio: {player['bpsvd_ratio']:.2f}")
    print(f"  Clutchness Score: {player['clutchness_score']:.2f}\n")

print(f"The player with the highest clutchness score is {best_player_clutchness['name']} with a score of {best_player_clutchness['clutchness_score']:.2f}.")


# In[ ]:




