import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris

# --- Configuration ---
# Using a subset of Iris (Setosa and Versicolor) which is exactly 100 rows
DATA_SIZE = 100
# Initial probability that the AI's suggestion is intended to be 'bad'
INITIAL_INTENDED_ERROR_RATE = 0.4
# Maximum number of suggestion turns
MAX_TURNS = 5

# --- Data Loading ---
@st.cache_data # Cache the data so it's only loaded once
def load_subset_data():
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    # Filter for species 0 (setosa) and 1 (versicolor)
    df_subset = df[df['target'].isin([0, 1])].reset_index(drop=True)
    # Map target back to original names for clarity if needed, or use numbers
    df_subset['species'] = df_subset['target'].map({0: 'setosa', 1: 'versicolor'})
    df_subset = df_subset.drop('target', axis=1) # Drop the numerical target column
    # Rename columns for better readability
    df_subset.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
    return df_subset

df = load_subset_data()
column_names = df.columns.tolist()

# --- AI Suggestion Logic ---
def generate_suggestion(current_error_rate):
    """Generates a visualization suggestion, potentially erroneous."""
    cols = random.sample(column_names, 2)
    col1, col2 = cols[0], cols[1]

    is_intended_error = random.random() < current_error_rate

    if is_intended_error:
        # Simulate an erroneous suggestion - e.g., irrelevant columns, wrong plot type
        plot_type = random.choice(['line', 'pie', 'area']) # Often inappropriate types
        # Keep the columns, even if they are inappropriate for the plot type or relationship
        was_good = False
        confidence = random.uniform(0.3, 0.7) # Lower confidence for intended errors
    else:
        # Simulate a reasonable suggestion based on data types and potential interest
        was_good = True
        confidence = random.uniform(0.75, 0.95) # Higher confidence for good suggestions

        # Determine a suitable plot type based on selected columns
        if df[col1].dtype in ['float64', 'int64'] and df[col2].dtype in ['float64', 'int64']:
             # Good suggestions for two numerical columns
             plot_type = random.choice(['scatter', 'regplot'])
        elif df[col1].dtype == 'object' or df[col2].dtype == 'object':
             # Good suggestions involving a categorical column
             cat_col = col1 if df[col1].dtype == 'object' else col2
             num_col = col2 if df[col1].dtype == 'object' else col1

             if df[col1].dtype == 'object' and df[col2].dtype == 'object':
                 # Two categorical - suggest countplot or heatmap
                 plot_type = random.choice(['countplot', 'heatmap'])
             elif num_col: # One categorical, one numerical
                 plot_type = random.choice(['boxplot', 'violinplot', 'swarmplot'])
             else: # Should not happen with 2 columns selected unless one is datetime or something
                 plot_type = 'scatter' # Default fallback

        else:
            # Fallback for unexpected types, suggest scatter or simple hist/bar
            plot_type = random.choice(['scatter', 'hist', 'bar'])


    # Adjust text if a plot type typically uses only one column (like hist or countplot)
    # This logic is simplified and might not catch all cases
    suggestion_text = f"Suggesting a {plot_type} plot for '{col1}' and '{col2}'."
    if plot_type in ['hist', 'countplot']:
        # If the chosen plot type is usually for one column, pick one of the two selected
        target_col = col1 # Arbitrarily pick the first one
        if df[col2].dtype == 'object' and plot_type == 'countplot': target_col = col2 # If countplot, prefer categorical
        elif df[col2].dtype in ['float64', 'int64'] and plot_type == 'hist': target_col = col2 # If hist, prefer numerical
        suggestion_text = f"Suggesting a {plot_type} plot for '{target_col}'."


    return suggestion_text, col1, col2, plot_type, was_good, confidence

# --- Trust Calculation and Adaptation ---
def update_trust(session_state):
    """Updates trust score based on interaction history."""
    cacg = session_state.get('cacg_count', 0)
    iab = session_state.get('iab_count', 0)
    crb = session_state.get('crb_count', 0)
    irg = session_state.get('irg_count', 0)

    total_actions = cacg + iab + crb + irg

    correct_actions = cacg + crb
    if total_actions == 0:
        new_trust_score = 0.5 # Start neutral before any interaction
    else:
        new_trust_score = correct_actions / total_actions

    session_state['trust_score'] = new_trust_score

    # Adaptive: Adjust the AI's NEXT intended error rate based on trust
    # Lower trust means the AI tries to be 'more careful' (lower error rate)
    # Higher trust means the AI might become slightly 'less careful'.
    # Let's make the error rate decrease linearly with trust, within bounds.
    # Clamped between 0.1 and 0.6
    adaptive_error_rate = INITIAL_INTENDED_ERROR_RATE * (1.5 - new_trust_score)
    session_state['current_intended_error_rate'] = max(0.1, min(0.6, adaptive_error_rate))


# --- Streamlit App ---
st.title("AI Trust Calibration Prototype (Visual Analytics)")

st.sidebar.header("Simulation Controls")
user_domain_knowledge = st.sidebar.slider(
    "Simulated Domain Knowledge",
    0, 100, 60, # Default to slightly above average
    help="Higher knowledge increases your chance of correctly identifying good/bad suggestions."
)

if st.sidebar.button("Reset Simulation"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Initialize session state
if 'suggestions' not in st.session_state: st.session_state['suggestions'] = []
if 'actions' not in st.session_state: st.session_state['actions'] = []
if 'cacg_count' not in st.session_state: st.session_state['cacg_count'] = 0
if 'iab_count' not in st.session_state: st.session_state['iab_count'] = 0
if 'crb_count' not in st.session_state: st.session_state['crb_count'] = 0
if 'irg_count' not in st.session_state: st.session_state['irg_count'] = 0
if 'trust_score' not in st.session_state: st.session_state['trust_score'] = 0.5
if 'current_intended_error_rate' not in st.session_state:
    st.session_state['current_intended_error_rate'] = INITIAL_INTENDED_ERROR_RATE
if 'trust_history' not in st.session_state: st.session_state['trust_history'] = []
if 'current_turn' not in st.session_state: st.session_state['current_turn'] = 1


st.sidebar.metric("Simulated Trust Score", f"{st.session_state['trust_score']:.2f}")
total_actions_taken = st.session_state['cacg_count'] + st.session_state['iab_count'] + st.session_state['crb_count'] + st.session_state['irg_count']
st.sidebar.write(f"({total_actions_taken} suggestions evaluated)")
if st.session_state['current_turn'] <= MAX_TURNS:
    st.sidebar.write(f"AI's next intended error rate: {st.session_state['current_intended_error_rate']:.2f}")

st.write(f"This prototype simulates how user trust in an AI visualization assistant might change based on observing its suggestions over time, influenced by the user's simulated domain knowledge. The simulation runs for {MAX_TURNS} turns.")
st.write(f"Using the Iris dataset ({DATA_SIZE} rows, Setosa and Versicolor species).")
st.write("The AI will suggest plots. Your domain knowledge affects your ability to evaluate these suggestions correctly.")

st.header("Dataset Preview (Subset of Iris)")
st.dataframe(df.head())

# --- Simulation Logic ---
if st.session_state['current_turn'] <= MAX_TURNS:
    st.header(f"AI Visualization Suggestion (Turn {st.session_state['current_turn']}/{MAX_TURNS})")

    # Generate a new suggestion only if the last one has been acted upon or it's the first turn
    # We need to generate a suggestion for the current turn if we haven't already
    if len(st.session_state['suggestions']) < st.session_state['current_turn']:
        suggestion_text, col1, col2, plot_type, was_good, confidence = generate_suggestion(
            st.session_state['current_intended_error_rate']
        )
        st.session_state['suggestions'].append({
            'text': suggestion_text,
            'col1': col1,
            'col2': col2,
            'plot_type': plot_type,
            'was_good': was_good,
            'confidence': confidence,
            'turn': st.session_state['current_turn']
        })

    current_suggestion = st.session_state['suggestions'][st.session_state['current_turn'] - 1]
    st.write(current_suggestion['text'])
    st.write(f"AI Confidence: {current_suggestion['confidence']:.2f}")

    # Calculate probability of user acting 'correctly' based on domain knowledge
    # Higher domain knowledge means higher probability of correctly identifying good/bad
    user_skill_prob = 0.4 + (user_domain_knowledge / 100) * 0.5 # Scales from 0.4 to 0.9

    col_accept, col_reject = st.columns(2)

    action_taken = None
    if col_accept.button("Accept Suggestion"):
        action_taken = 'accepted'
    elif col_reject.button("Reject Suggestion"):
        action_taken = 'rejected'

    if action_taken:
        was_good = current_suggestion['was_good']
        action_outcome = None # 'correct' or 'incorrect'
        feedback_message = ""

        # Simulate if the user's action was 'correct' based on the suggestion's 'was_good' status
        # and the user's simulated domain knowledge.
        user_successfully_evaluated = random.random() < user_skill_prob

        if action_taken == 'accepted':
            if was_good:
                # User accepted a good suggestion - this is correct
                st.session_state['cacg_count'] += 1
                action_outcome = 'correct'
                feedback_message = "👍 Correctly Accepted a good suggestion."
            else:
                # User accepted a bad suggestion - this is incorrect (automation bias)
                st.session_state['iab_count'] += 1
                action_outcome = 'incorrect'
                feedback_message = "🚨 Incorrectly Accepted a bad suggestion (automation bias)."

        elif action_taken == 'rejected':
            if not was_good:
                # User rejected a bad suggestion - this is correct
                st.session_state['crb_count'] += 1
                action_outcome = 'correct'
                feedback_message = "✅ Correctly Rejected a bad suggestion."
            else:
                 # User rejected a good suggestion - this is incorrect
                 st.session_state['irg_count'] += 1
                 action_outcome = 'incorrect'
                 feedback_message = "❌ Incorrectly Rejected a good suggestion."

        # Display feedback message
        st.write(feedback_message)

        # Store the outcome of the action
        st.session_state['actions'].append({
            'action': action_taken,
            'was_good_suggestion': was_good,
            'user_action_outcome': action_outcome,
            'turn': current_suggestion['turn']
        })

        # Update trust and adaptive parameters AFTER the action is recorded
        update_trust(st.session_state)
        st.session_state['trust_history'].append({'turn': st.session_state['current_turn'], 'trust': st.session_state['trust_score']})

        # Move to the next turn
        st.session_state['current_turn'] += 1

        # Rerun to update the display (hide buttons, show next suggestion or final result)
        st.rerun()

# --- End of Simulation ---
elif st.session_state['current_turn'] > MAX_TURNS:
    st.header("Simulation Complete")
    st.write(f"The simulation has finished after {MAX_TURNS} turns.")
    st.write(f"Your final simulated trust score is: **{st.session_state['trust_score']:.2f}**")
    st.write("Review the history and the trust calibration plot below.")


# Display the plot if the last action was 'accepted' and the user action was 'correct'
# Only show plot if the user correctly accepted a good suggestion on the *last completed turn*
if st.session_state['actions'] and st.session_state['actions'][-1]['action'] == 'accepted' and st.session_state['actions'][-1]['user_action_outcome'] == 'correct':
    st.header("Generated Plot (Last Accepted Good Suggestion)")
    # Find the suggestion that corresponds to the last action by turn number
    last_suggestion_turn = st.session_state['actions'][-1]['turn']
    last_suggestion = next(
        (s for s in st.session_state['suggestions'] if s['turn'] == last_suggestion_turn),
        None
    )

    if last_suggestion:
        col1 = last_suggestion['col1']
        col2 = last_suggestion['col2']
        plot_type = last_suggestion['plot_type']

        fig, ax = plt.subplots()

        try:
            # Ensure appropriate plot calls based on types and number of columns needed
            if plot_type in ['scatter', 'regplot']:
                 if df[col1].dtype in ['float64', 'int64'] and df[col2].dtype in ['float64', 'int64']:
                     if plot_type == 'scatter': sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
                     elif plot_type == 'regplot': sns.regplot(data=df, x=col1, y=col2, ax=ax)
                 else:
                     st.warning(f"Plot type '{plot_type}' not suitable for selected columns.")
                     fig.clear() # Clear the figure
            elif plot_type in ['boxplot', 'violinplot', 'swarmplot']:
                 # Need one numerical and one categorical
                 if (df[col1].dtype in ['float64', 'int64'] and df[col2].dtype == 'object') or \
                    (df[col2].dtype in ['float64', 'int64'] and df[col1].dtype == 'object'):
                     num_col = col1 if df[col1].dtype in ['float64', 'int64'] else col2
                     cat_col = col2 if df[col1].dtype in ['float64', 'int64'] else col1
                     if plot_type == 'boxplot': sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
                     elif plot_type == 'violinplot': sns.violinplot(data=df, x=cat_col, y=num_col, ax=ax)
                     elif plot_type == 'swarmplot': sns.swarmplot(data=df, x=cat_col, y=num_col, ax=ax)
                 else:
                      st.warning(f"Plot type '{plot_type}' requires one numerical and one categorical column.")
                      fig.clear()
            elif plot_type in ['hist', 'countplot']:
                 # Need one column
                 # Determine which of the two suggested columns is appropriate
                 target_col = None
                 if plot_type == 'hist' and df[col1].dtype in ['float64', 'int64']: target_col = col1
                 elif plot_type == 'hist' and df[col2].dtype in ['float64', 'int64']: target_col = col2
                 elif plot_type == 'countplot' and df[col1].dtype == 'object': target_col = col1
                 elif plot_type == 'countplot' and df[col2].dtype == 'object': target_col = col2

                 if target_col:
                     if plot_type == 'hist': sns.histplot(data=df, x=target_col, ax=ax)
                     elif plot_type == 'countplot': sns.countplot(data=df, x=target_col, ax=ax)
                 else:
                     st.warning(f"Plot type '{plot_type}' not suitable for selected columns '{col1}' and '{col2}'.")
                     fig.clear()

            elif plot_type == 'heatmap':
                 # Need two categorical or correlation matrix of numericals
                 if df[col1].dtype == 'object' and df[col2].dtype == 'object':
                      # Simple counts heatmap for two categoricals - needs reshaping
                      heatmap_data = df.groupby([col1, col2]).size().unstack(fill_value=0)
                      if not heatmap_data.empty:
                         sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax)
                      else:
                         st.warning(f"No data to create heatmap for '{col1}' and '{col2}'.")
                         fig.clear()
                 elif df[col1].dtype in ['float64', 'int64'] and df[col2].dtype in ['float64', 'int64']:
                      # Correlation heatmap (simplified)
                      corr_matrix = df[[col1, col2]].corr()
                      if not corr_matrix.empty:
                         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                      else:
                         st.warning(f"No numerical data to create correlation heatmap.")
                         fig.clear()
                 else:
                      st.warning(f"Plot type '{plot_type}' not suitable for selected columns.")
                      fig.clear()
            else:
                st.warning(f"Unsupported plot type '{plot_type}' for visualization in this prototype.")
                fig.clear() # Clear the figure if plot fails

            if fig.get_axes() and not fig.get_axes()[0].lines and not fig.get_axes()[0].collections:
                 # Check if plot is empty even if axes were created (e.g., warning was issued internally by seaborn)
                 st.warning(f"Could not generate plot for the selected columns/type.")
                 fig.clear() # Ensure figure is cleared if plotting failed silently
            elif fig.get_axes():
                 # Set titles/labels only if axes were created and potentially populated
                 ax.set_title(f"{plot_type.capitalize()} Plot")
                 # Attempt to set labels if they make sense for the plot type
                 if plot_type in ['scatter', 'regplot', 'boxplot', 'violinplot', 'swarmplot', 'heatmap']:
                      ax.set_xlabel(col1)
                      ax.set_ylabel(col2)
                 elif plot_type in ['hist', 'countplot']:
                      # For single-column plots, label the X axis with the column name
                      ax.set_xlabel(last_suggestion['text'].split("'")[1]) # Extract column name from suggestion text
                 st.pyplot(fig)
            else:
                 # If axes creation failed entirely
                 st.warning(f"Could not generate plot for the selected columns/type.")


        except Exception as e:
            st.error(f"An error occurred while generating the plot: {e}")
            fig.clear() # Ensure figure is cleared on error


# Display Trust Calibration over time
if st.session_state['trust_history']:
    st.header("Trust Calibration over Time")
    trust_history_df = pd.DataFrame(st.session_state['trust_history'])
    fig_trust, ax_trust = plt.subplots()
    sns.lineplot(data=trust_history_df, x='turn', y='trust', marker='o', ax=ax_trust)
    ax_trust.set_xlabel("Suggestion Turn")
    ax_trust.set_ylabel("Simulated Trust Score")
    ax_trust.set_title("Simulated Trust Calibration")
    ax_trust.set_ylim(0, 1) # Trust is between 0 and 1
    ax_trust.set_xticks(range(1, MAX_TURNS + 1)) # Set ticks for each turn
    st.pyplot(fig_trust)

# Optional: Display history of suggestions and actions
st.header("Interaction History")

history_data = []
# Combine suggestion and action history by turn
suggestions_dict = {s['turn']: s for s in st.session_state['suggestions']}
actions_dict = {a['turn']: a for a in st.session_state['actions']}

# Iterate through turns 1 to MAX_TURNS
for turn in range(1, MAX_TURNS + 1):
    suggestion = suggestions_dict.get(turn)
    action = actions_dict.get(turn)

    history_entry = {
        'Turn': turn,
        'Suggestion': suggestion['text'] if suggestion else 'N/A',
        'AI Confidence': f"{suggestion['confidence']:.2f}" if suggestion else 'N/A',
        'Intended Good?': suggestion['was_good'] if suggestion is not None else 'N/A', # Use is not None check
        'User Action': action['action'] if action else 'Pending',
        'User Action Outcome': action['user_action_outcome'] if action else 'Pending'
    }
    history_data.append(history_entry)

history_df_display = pd.DataFrame(history_data)
st.dataframe(history_df_display)

