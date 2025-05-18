import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris

DATA_SIZE_FOR_IRIS = 100
MY_ERROR_RATE_INITIAL = 0.4
NUMBER_OF_MAXIMUM_TURNS = 5


# Load the data from iris dataset
@st.cache_data # Cache the data so it's only loaded once
def get_my_iris_data_subset():
    iris = load_iris()
    my_dataframe = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                columns=iris['feature_names'] + ['target'])
    # Only keep setosa and versicolor
    my_filtered_dataframe = my_dataframe[my_dataframe['target'].isin([0, 1])].reset_index(drop=True)
    # Make species names instead of numbers
    my_filtered_dataframe['species'] = my_filtered_dataframe['target'].map({0: 'setosa', 1: 'versicolor'})
    my_filtered_dataframe = my_filtered_dataframe.drop('target', axis=1)
    my_filtered_dataframe.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)',
                                     'species']
    return my_filtered_dataframe


my_dataframe = get_my_iris_data_subset()
names_of_my_columns = my_dataframe.columns.tolist()


def make_ai_suggestion(error_rate_for_now):
    cols = random.sample(names_of_my_columns, 2)
    first_column, second_column = cols[0], cols[1]

    does_ai_make_mistake = random.random() < error_rate_for_now

    if does_ai_make_mistake:
        type_of_plot = random.choice(['line', 'pie', 'area'])
        is_this_good = False
        how_confident_ai_is = random.uniform(0.3, 0.7)
    else:
        is_this_good = True
        how_confident_ai_is = random.uniform(0.75, 0.95)

        if my_dataframe[first_column].dtype in ['float64', 'int64'] and my_dataframe[second_column].dtype in ['float64',
                                                                                                              'int64']:
            type_of_plot = random.choice(['scatter', 'regplot'])
        elif my_dataframe[first_column].dtype == 'object' or my_dataframe[second_column].dtype == 'object':
            category_column = first_column if my_dataframe[first_column].dtype == 'object' else second_column
            number_column = second_column if my_dataframe[first_column].dtype == 'object' else first_column

            if my_dataframe[first_column].dtype == 'object' and my_dataframe[second_column].dtype == 'object':
                type_of_plot = random.choice(['countplot', 'heatmap'])
            elif number_column:
                type_of_plot = random.choice(['boxplot', 'violinplot', 'swarmplot'])
            else:
                type_of_plot = 'scatter'
        else:
            type_of_plot = random.choice(['scatter', 'hist', 'bar'])

    text_of_suggestion = f"Suggesting a {type_of_plot} plot for '{first_column}' and '{second_column}'."
    if type_of_plot in ['hist', 'countplot']:
        column_to_use = first_column
        if my_dataframe[second_column].dtype == 'object' and type_of_plot == 'countplot':
            column_to_use = second_column
        elif my_dataframe[second_column].dtype in ['float64', 'int64'] and type_of_plot == 'hist':
            column_to_use = second_column
        text_of_suggestion = f"Suggesting a {type_of_plot} plot for '{column_to_use}'."

    return text_of_suggestion, first_column, second_column, type_of_plot, is_this_good, how_confident_ai_is


def calculate_new_trust_score(session_state_stuff):
    correctly_accept_good = session_state_stuff.get('cacg_count', 0)
    incorrect_accept_bad = session_state_stuff.get('iab_count', 0)
    correctly_reject_bad = session_state_stuff.get('crb_count', 0)
    incorrect_reject_good = session_state_stuff.get('irg_count', 0)

    all_actions_total = correctly_accept_good + incorrect_accept_bad + correctly_reject_bad + incorrect_reject_good

    correct_actions_count = correctly_accept_good + correctly_reject_bad
    if all_actions_total == 0:
        updated_trust = 0.5
    else:
        updated_trust = correct_actions_count / all_actions_total

    session_state_stuff['trust_score'] = updated_trust

    adaptive_error_probability = MY_ERROR_RATE_INITIAL * (1.5 - updated_trust)
    session_state_stuff['current_intended_error_rate'] = max(0.1, min(0.6, adaptive_error_probability))


# Start my app!!
st.title("AI Trust Calibration Prototype (Visual Analytics)")

st.sidebar.header("Simulation Controls")
how_much_user_knows = st.sidebar.slider(
    "Simulated Domain Knowledge",
    0, 100, 60,
    help="Higher knowledge increases your chance of correctly identifying good/bad suggestions."
)

if st.sidebar.button("Reset Simulation"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Setup all my variables
if 'suggestions' not in st.session_state: st.session_state['suggestions'] = []
if 'actions' not in st.session_state: st.session_state['actions'] = []
if 'cacg_count' not in st.session_state: st.session_state['cacg_count'] = 0
if 'iab_count' not in st.session_state: st.session_state['iab_count'] = 0
if 'crb_count' not in st.session_state: st.session_state['crb_count'] = 0
if 'irg_count' not in st.session_state: st.session_state['irg_count'] = 0
if 'trust_score' not in st.session_state: st.session_state['trust_score'] = 0.5
if 'current_intended_error_rate' not in st.session_state:
    st.session_state['current_intended_error_rate'] = MY_ERROR_RATE_INITIAL
if 'trust_history' not in st.session_state: st.session_state['trust_history'] = []
if 'current_turn' not in st.session_state: st.session_state['current_turn'] = 1

st.sidebar.metric("Simulated Trust Score", f"{st.session_state['trust_score']:.2f}")
all_actions_taken_so_far = st.session_state['cacg_count'] + st.session_state['iab_count'] + st.session_state[
    'crb_count'] + st.session_state['irg_count']
st.sidebar.write(f"({all_actions_taken_so_far} suggestions evaluated)")
if st.session_state['current_turn'] <= NUMBER_OF_MAXIMUM_TURNS:
    st.sidebar.write(f"AI's next intended error rate: {st.session_state['current_intended_error_rate']:.2f}")

st.write(
    f"This prototype simulates how user trust in an AI visualization assistant might change based on observing its suggestions over time, influenced by the user's simulated domain knowledge. The simulation runs for {NUMBER_OF_MAXIMUM_TURNS} turns.")
st.write(f"Using the Iris dataset ({DATA_SIZE_FOR_IRIS} rows, Setosa and Versicolor species).")
st.write(
    "The AI will suggest plots. Your domain knowledge affects your ability to evaluate these suggestions correctly.")

st.header("Dataset Preview (Subset of Iris)")
st.dataframe(my_dataframe.head())

# Main game part
if st.session_state['current_turn'] <= NUMBER_OF_MAXIMUM_TURNS:
    st.header(f"AI Visualization Suggestion (Turn {st.session_state['current_turn']}/{NUMBER_OF_MAXIMUM_TURNS})")

    if len(st.session_state['suggestions']) < st.session_state['current_turn']:
        text_of_suggestion, first_column, second_column, type_of_plot, is_this_good, how_confident_ai_is = make_ai_suggestion(
            st.session_state['current_intended_error_rate']
        )
        st.session_state['suggestions'].append({
            'text': text_of_suggestion,
            'col1': first_column,
            'col2': second_column,
            'plot_type': type_of_plot,
            'was_good': is_this_good,
            'confidence': how_confident_ai_is,
            'turn': st.session_state['current_turn']
        })

    current_suggestion_now = st.session_state['suggestions'][st.session_state['current_turn'] - 1]
    st.write(current_suggestion_now['text'])
    st.write(f"AI Confidence: {current_suggestion_now['confidence']:.2f}")

    probability_of_correct_action = 0.4 + (how_much_user_knows / 100) * 0.5

    column_for_accept, column_for_reject = st.columns(2)

    action_user_took = None
    if column_for_accept.button("Accept Suggestion"):
        action_user_took = 'accepted'
    elif column_for_reject.button("Reject Suggestion"):
        action_user_took = 'rejected'

    if action_user_took:
        suggestion_was_good = current_suggestion_now['was_good']
        what_happened = None
        message_to_show = ""

        user_knows_enough = random.random() < probability_of_correct_action

        if action_user_took == 'accepted':
            if suggestion_was_good:
                st.session_state['cacg_count'] += 1
                what_happened = 'correct'
                message_to_show = "👍 Correctly Accepted a good suggestion."
            else:
                st.session_state['iab_count'] += 1
                what_happened = 'incorrect'
                message_to_show = "🚨 Incorrectly Accepted a bad suggestion (automation bias)."

        elif action_user_took == 'rejected':
            if not suggestion_was_good:
                st.session_state['crb_count'] += 1
                what_happened = 'correct'
                message_to_show = "✅ Correctly Rejected a bad suggestion."
            else:
                st.session_state['irg_count'] += 1
                what_happened = 'incorrect'
                message_to_show = "❌ Incorrectly Rejected a good suggestion."

        st.write(message_to_show)

        st.session_state['actions'].append({
            'action': action_user_took,
            'was_good_suggestion': suggestion_was_good,
            'user_action_outcome': what_happened,
            'turn': current_suggestion_now['turn']
        })

        calculate_new_trust_score(st.session_state)
        st.session_state['trust_history'].append(
            {'turn': st.session_state['current_turn'], 'trust': st.session_state['trust_score']})

        st.session_state['current_turn'] += 1

        st.rerun()

elif st.session_state['current_turn'] > NUMBER_OF_MAXIMUM_TURNS:
    st.header("Simulation Complete")
    st.write(f"The simulation has finished after {NUMBER_OF_MAXIMUM_TURNS} turns.")
    st.write(f"Your final simulated trust score is: **{st.session_state['trust_score']:.2f}**")
    st.write("Review the history and the trust calibration plot below.")

# Show plot if user accepted a good one
if st.session_state['actions'] and st.session_state['actions'][-1]['action'] == 'accepted' and \
        st.session_state['actions'][-1]['user_action_outcome'] == 'correct':
    st.header("Generated Plot (Last Accepted Good Suggestion)")

    last_turn_number = st.session_state['actions'][-1]['turn']
    last_suggestion_details = None
    for s in st.session_state['suggestions']:
        if s['turn'] == last_turn_number:
            last_suggestion_details = s
            break

    if last_suggestion_details:
        first_column = last_suggestion_details['col1']
        second_column = last_suggestion_details['col2']
        type_of_plot = last_suggestion_details['plot_type']

        my_figure, my_axis = plt.subplots()

        try:
            if type_of_plot in ['scatter', 'regplot']:
                if my_dataframe[first_column].dtype in ['float64', 'int64'] and my_dataframe[second_column].dtype in [
                    'float64', 'int64']:
                    if type_of_plot == 'scatter':
                        sns.scatterplot(data=my_dataframe, x=first_column, y=second_column, ax=my_axis)
                    elif type_of_plot == 'regplot':
                        sns.regplot(data=my_dataframe, x=first_column, y=second_column, ax=my_axis)
                else:
                    st.warning(f"Plot type '{type_of_plot}' not suitable for selected columns.")
                    my_figure.clear()
            elif type_of_plot in ['boxplot', 'violinplot', 'swarmplot']:
                if (my_dataframe[first_column].dtype in ['float64', 'int64'] and my_dataframe[
                    second_column].dtype == 'object') or \
                        (my_dataframe[second_column].dtype in ['float64', 'int64'] and my_dataframe[
                            first_column].dtype == 'object'):
                    number_column = first_column if my_dataframe[first_column].dtype in ['float64',
                                                                                         'int64'] else second_column
                    category_column = second_column if my_dataframe[first_column].dtype in ['float64',
                                                                                            'int64'] else first_column
                    if type_of_plot == 'boxplot':
                        sns.boxplot(data=my_dataframe, x=category_column, y=number_column, ax=my_axis)
                    elif type_of_plot == 'violinplot':
                        sns.violinplot(data=my_dataframe, x=category_column, y=number_column, ax=my_axis)
                    elif type_of_plot == 'swarmplot':
                        sns.swarmplot(data=my_dataframe, x=category_column, y=number_column, ax=my_axis)
                else:
                    st.warning(f"Plot type '{type_of_plot}' requires one numerical and one categorical column.")
                    my_figure.clear()
            elif type_of_plot in ['hist', 'countplot']:
                column_to_use = None
                if type_of_plot == 'hist' and my_dataframe[first_column].dtype in ['float64', 'int64']:
                    column_to_use = first_column
                elif type_of_plot == 'hist' and my_dataframe[second_column].dtype in ['float64', 'int64']:
                    column_to_use = second_column
                elif type_of_plot == 'countplot' and my_dataframe[first_column].dtype == 'object':
                    column_to_use = first_column
                elif type_of_plot == 'countplot' and my_dataframe[second_column].dtype == 'object':
                    column_to_use = second_column

                if column_to_use:
                    if type_of_plot == 'hist':
                        sns.histplot(data=my_dataframe, x=column_to_use, ax=my_axis)
                    elif type_of_plot == 'countplot':
                        sns.countplot(data=my_dataframe, x=column_to_use, ax=my_axis)
                else:
                    st.warning(
                        f"Plot type '{type_of_plot}' not suitable for selected columns '{first_column}' and '{second_column}'.")
                    my_figure.clear()

            elif type_of_plot == 'heatmap':
                if my_dataframe[first_column].dtype == 'object' and my_dataframe[second_column].dtype == 'object':
                    my_heatmap_data = my_dataframe.groupby([first_column, second_column]).size().unstack(fill_value=0)
                    if not my_heatmap_data.empty:
                        sns.heatmap(my_heatmap_data, annot=True, fmt='d', cmap='Blues', ax=my_axis)
                    else:
                        st.warning(f"No data to create heatmap for '{first_column}' and '{second_column}'.")
                        my_figure.clear()
                elif my_dataframe[first_column].dtype in ['float64', 'int64'] and my_dataframe[second_column].dtype in [
                    'float64', 'int64']:
                    my_corr_matrix = my_dataframe[[first_column, second_column]].corr()
                    if not my_corr_matrix.empty:
                        sns.heatmap(my_corr_matrix, annot=True, cmap='coolwarm', ax=my_axis)
                    else:
                        st.warning(f"No numerical data to create correlation heatmap.")
                        my_figure.clear()
                else:
                    st.warning(f"Plot type '{type_of_plot}' not suitable for selected columns.")
                    my_figure.clear()
            else:
                st.warning(f"Unsupported plot type '{type_of_plot}' for visualization in this prototype.")
                my_figure.clear()

            if my_figure.get_axes() and not my_figure.get_axes()[0].lines and not my_figure.get_axes()[0].collections:
                st.warning(f"Could not generate plot for the selected columns/type.")
                my_figure.clear()
            elif my_figure.get_axes():
                my_axis.set_title(f"{type_of_plot.capitalize()} Plot")
                if type_of_plot in ['scatter', 'regplot', 'boxplot', 'violinplot', 'swarmplot', 'heatmap']:
                    my_axis.set_xlabel(first_column)
                    my_axis.set_ylabel(second_column)
                elif type_of_plot in ['hist', 'countplot']:
                    my_axis.set_xlabel(last_suggestion_details['text'].split("'")[1])
                st.pyplot(my_figure)
            else:
                st.warning(f"Could not generate plot for the selected columns/type.")


        except Exception as e:
            st.error(f"An error occurred while generating the plot: {e}")
            my_figure.clear()

# Show trust over time
if st.session_state['trust_history']:
    st.header("Trust Calibration over Time")
    my_trust_history_data = pd.DataFrame(st.session_state['trust_history'])
    my_trust_figure, my_trust_axis = plt.subplots()
    sns.lineplot(data=my_trust_history_data, x='turn', y='trust', marker='o', ax=my_trust_axis)
    my_trust_axis.set_xlabel("Suggestion Turn")
    my_trust_axis.set_ylabel("Simulated Trust Score")
    my_trust_axis.set_title("Simulated Trust Calibration")
    my_trust_axis.set_ylim(0, 1)
    my_trust_axis.set_xticks(range(1, NUMBER_OF_MAXIMUM_TURNS + 1))
    st.pyplot(my_trust_figure)

# Show history
st.header("Interaction History")

my_history_data = []
my_suggestions_dict = {}
for s in st.session_state['suggestions']:
    my_suggestions_dict[s['turn']] = s

my_actions_dict = {}
for a in st.session_state['actions']:
    my_actions_dict[a['turn']] = a

for turn_number in range(1, NUMBER_OF_MAXIMUM_TURNS + 1):
    suggestion_for_this_turn = my_suggestions_dict.get(turn_number)
    action_for_this_turn = my_actions_dict.get(turn_number)

    my_history_entry = {
        'Turn': turn_number,
        'Suggestion': suggestion_for_this_turn['text'] if suggestion_for_this_turn else 'N/A',
        'AI Confidence': f"{suggestion_for_this_turn['confidence']:.2f}" if suggestion_for_this_turn else 'N/A',
        'Intended Good?': suggestion_for_this_turn['was_good'] if suggestion_for_this_turn is not None else 'N/A',
        'User Action': action_for_this_turn['action'] if action_for_this_turn else 'Pending',
        'User Action Outcome': action_for_this_turn['user_action_outcome'] if action_for_this_turn else 'Pending'
    }
    my_history_data.append(my_history_entry)

my_history_display_data = pd.DataFrame(my_history_data)
st.dataframe(my_history_display_data)
