import numpy as np

def read_data(file_path):
    """ 
    This function reads data from the given file
    The data is seperated into chunks denoted by two new line characters ('\n\n')
    In each chunk is expected to have three lines: a header, a sequence, and a state sequence
    returns: two lists sequences and state_sequences
    """
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')
    sequences = []
    state_sequences = []
    for entry in data:
        parts = entry.split('\n')
        if len(parts) == 3:
            sequences.append(parts[1].strip())
            state_sequences.append(parts[2].strip().lstrip('#').strip())
    return sequences, state_sequences

def count_trans_emiss(sequences, state_sequences):
    """
    inputs: sequences and state_sequences from the read file
    this function initalizes dictonaries for counting transactions between states (transitions)
    and emissions of symbols from states (emissions)
    it iterates through each sequence and its state sequence, updating the ocunts of transistions
    and emissions
    returns: transitions and emissions (as dictonaries)
    """
    states = {'o', 'M', 'i'}
    transitions = {s: {s2: 0 for s2 in states} for s in states}
    emissions = {s: {} for s in states}
    
    for seq, states_seq in zip(sequences, state_sequences):
        prev_state = None
        for i, state in enumerate(states_seq):
            if state not in states:
                continue

            aa = seq[i]
            emissions[state][aa] = emissions[state].get(aa, 0) + 1
            
            if prev_state is not None:
                transitions[prev_state][state] += 1
            prev_state = state
    
    return transitions, emissions

def calc_prob(transitions, emissions):
    """
    inputs: transitions, emissions
    this function calculates the probabilities, using smoothing where needed
    returns: transition_probs, emission_probs (as dictonaries) 
    """
    smoothing_val = 1e-3
    all_states = ['o', 'M', 'i']
    all_symbols = set(sym for em in emissions.values() for sym in em)

    transition_probs = {s: {} for s in transitions}
    for s in all_states:
        total_transitions = sum(transitions.get(s, {}).values()) + smoothing_val * len(all_states)
        for s2 in all_states:
            transition_probs[s][s2] = (transitions.get(s, {}).get(s2, 0) + smoothing_val) / total_transitions

    emission_probs = {s: {} for s in emissions}
    for s in all_states:
        total_emissions = sum(emissions.get(s, {}).values()) + smoothing_val * len(all_symbols)
        for symbol in all_symbols:
            emission_probs[s][symbol] = (emissions.get(s, {}).get(symbol, 0) + smoothing_val) / total_emissions

    return transition_probs, emission_probs

def log_viterbi(seq, transition_probs, emission_probs):
    """
    inputs: seq (sequence), transition_probs, emission_prob
    this function implements the Viterbi algorithm with logarithm transformation
    it inializes matricies (V and path) for storing probabilities and paths through states
    it computes the probability at each step, then backtraces to find the most likely 
    sequence of states
    returns: most likely state sequence for the given sequence
    """
    state_map = {'o': 0, 'M': 1, 'i': 2}
    V = np.full((3, len(seq)), -1000)
    path = np.zeros((3, len(seq)), dtype=int)
    state_list = ['o', 'M', 'i']

    for s in state_list:
        V[state_map[s], 0] = np.log(emission_probs[s].get(seq[0], 1e-10))

    for t in range(1, len(seq)):
        for s in state_list:
            for ps in state_list:
                prob = V[state_map[ps], t-1] + np.log(transition_probs[ps].get(s, 1e-10)) + np.log(emission_probs[s].get(seq[t], 1e-10))
                if prob > V[state_map[s], t]:
                    V[state_map[s], t] = prob
                    path[state_map[s], t] = state_map[ps]

    best_last_state = np.argmax(V[:, -1])
    best_path = [best_last_state]
    for t in range(len(seq) - 1, 0, -1):
        best_path.insert(0, path[best_path[0], t])

    best_path_str = ''.join(state_list[i] for i in best_path)
    return best_path_str

def evaluate_performance(true_state_sequences, predicted_state_sequences):
    """
    inputs: true state sequences, predicted state sequences
    this function evaluates the performance of predicted state sequences compared to the true
    state sequences
    it calculates the following: recall, precision, F1-score, and accuracy
    for each label 'M', 'i', 'o'
    returns: prints the evaluation metrics for each label
    """
    labels = ['M', 'i', 'o']
    metrics = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for label in labels}

    for true_states, predicted_states in zip(true_state_sequences, predicted_state_sequences):
        for i in range(len(true_states)):
            for label in labels:
                if predicted_states[i] == label:
                    if true_states[i] == label:
                        metrics[label]['TP'] += 1
                    else:
                        metrics[label]['FP'] += 1
                else:
                    if true_states[i] == label:
                        metrics[label]['FN'] += 1
                    else:
                        metrics[label]['TN'] += 1

    for label in labels:
        TP = metrics[label]['TP']
        FP = metrics[label]['FP']
        TN = metrics[label]['TN']
        FN = metrics[label]['FN']
        
        R = TP / (TP + FN) if TP + FN != 0 else 0
        P = TP / (TP + FP) if TP + FP != 0 else 0
        F1 = 2 * (R * P) / (R + P) if R + P != 0 else 0
        ACCU = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
        
        print(f"Label {label}: Recall (R)={R:.4f}, Precision (P)={P:.4f}, F1-Score={F1:.4f}, Accuracy (ACCU)={ACCU:.4f}")


def main():
    train_sequences, train_state_sequences = read_data('hw2_train')
    transitions, emissions = count_trans_emiss(train_sequences, train_state_sequences)
    transition_probs, emission_probs = calc_prob(transitions, emissions)
    
    test_sequences, true_state_sequences = read_data('hw2_test')
    
    predicted_state_sequences = []
    for seq in test_sequences:
        predicted_states = log_viterbi(seq, transition_probs, emission_probs)
        predicted_state_sequences.append(predicted_states)
    
    evaluate_performance(true_state_sequences, predicted_state_sequences)

if __name__ == "__main__":
    main()
