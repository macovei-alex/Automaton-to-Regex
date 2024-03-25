import copy

from constants import *


class FiniteAutomaton:
    def __init__(self, states: list[str], alphabet: list[str], initial_state: str, final_states: list[str],
                 transitions: list[tuple[str, str, str]]) -> None:
        """Initializes the automaton. The parameters are deep copied to avoid any side effects.
        :param self: the automaton to initialize
        :param states: the states of the automaton
        :param alphabet: the alphabet of the automaton
        :param initial_state: the initial state of the automaton
        :param final_states: the final states of the automaton
        :param transitions: the transitions of the automaton
        :return: None"""

        self.states: list[str] = copy.deepcopy(states)
        self.alphabet: list[str] = sorted(copy.deepcopy(alphabet))
        self.initial_state: str = initial_state
        self.final_states: list[str] = copy.deepcopy(final_states)
        self.transitions: list[tuple[str, str, str]] = copy.deepcopy(transitions)

    def __str__(self) -> str:
        """Gets the string representation of the automaton.
        :param self: the automaton
        :return: the string representation of the automaton"""

        return (f"States: {self.states}\n"
                f"Alphabet: {self.alphabet}\n"
                f"Initial state: {self.initial_state}\n"
                f"Final states: {self.final_states}\n"
                f"Transitions: \n\t" +
                '\n\t'.join([f"({transition[0]}, {transition[1]}) -> {transition[2]}" for transition in
                             self.transitions]))

    def __bool__(self) -> bool:
        """Determines whether the automaton is valid.
        :param self: the automaton
        :return: True or False"""

        return self.is_valid()

    def __len__(self) -> int:
        """Gets the number of states in the automaton.
        :param self: the automaton
        :return: the number of states in the automaton"""

        return len(self.states)

    def __getitem__(self, item: int) -> str:
        """Gets the state at the specified index.
        :param self: the automaton
        :param item: the index of the state
        :return: the state at the specified index"""

        return self.states[item]

    def __eq__(self, other: 'FiniteAutomaton') -> bool:
        """Determines whether the 2 automata are equivalent.
        The algorithm used is the comparison method described in the book
        "Theory of Computer Science: Automata, Languages, and Computation" by Mishra and Chandrasekaran,
        3rd edition, 2008, page 158.
        :param self: the first automaton
        :param other: the second automaton
        :return: True or False"""

        m1: FiniteAutomaton = self.to_deterministic().minimize()
        m2: FiniteAutomaton = other.to_deterministic().minimize()
        # we will consider n = max(len(m1.states), len(m2.states))

        # O(n^2)
        m1.alphabet = list({symbol for _, symbol, _ in m1.transitions})
        m2.alphabet = list({symbol for _, symbol, _ in m2.transitions})
        if m1.alphabet != m2.alphabet:
            return False

        alphabet: list[str] = m1.alphabet
        # we will consider s = len(alphabet)

        # O(n)
        final_states_set_1: set[str] = set(m1.final_states)
        final_states_set_2: set[str] = set(m2.final_states)

        pair_t = tuple[str, str]

        # O(n^2)
        transitions_dict_1: dict[pair_t, str] = {(t[0], t[1]): t[2] for t in m1.transitions}
        transitions_dict_2: dict[pair_t, str] = {(t[0], t[1]): t[2] for t in m2.transitions}

        # O(n^2 * s)
        table: dict[pair_t, dict[str, pair_t]] = {
            (state1, state2): {symbol: ('', '') for symbol in alphabet}
            for state1 in m1.states
            for state2 in m2.states
            if (state1 in final_states_set_1) == (state2 in final_states_set_2)
        }

        # O(n^2 * s)
        for symbol in alphabet:
            for (state1, state2) in table.keys():

                # O(1)
                next_state1: str = transitions_dict_1.get(state1, symbol)
                next_state2: str = transitions_dict_2.get(state2, symbol)

                # O(1)
                if (next_state1 in final_states_set_1) != (next_state2 in final_states_set_2):
                    return False

                # the insertion is optional, it doesn't affect the correctness of the algorithm
                table[(state1, state2)][symbol] = (next_state1, next_state2)

        return True
        # total complexity: O(n^2 * s)

    @staticmethod
    def new() -> 'FiniteAutomaton':
        """Generates a new empty automaton.
        :return: the new automaton but empty"""

        return FiniteAutomaton([], [], '', [], [])

    def empty(self) -> 'FiniteAutomaton':
        """Clears the automaton.
        :param self: the original automaton
        :return: the automaton but empty"""

        self.states.clear()
        self.alphabet.clear()
        self.initial_state = ''
        self.final_states.clear()
        self.transitions.clear()
        return self

    @staticmethod
    def new_primitive(symbol: str) -> 'FiniteAutomaton':
        """Generates a new automaton that accepts a single symbol.
        :param symbol: the symbol that the automaton will accept
        :return: the new automaton that accepts the symbol"""

        if symbol == LAMBDA:
            return FiniteAutomaton(['q0'], [], 'q0', ['q0'], [])

        return FiniteAutomaton(['q0', 'q1'], [symbol], 'q0', ['q1'], [('q0', symbol, 'q1')])

    def sorted(self) -> 'FiniteAutomaton':
        """Sorts the states, the alphabet, the final states, and the transitions of the automaton.
        :param self: the original automaton
        :return: the automaton but sorted"""

        self.states[:] = sorted(self.states, key=lambda state: int(state[1:]))
        self.alphabet[:] = sorted(self.alphabet)
        self.final_states[:] = sorted(self.final_states, key=lambda state: int(state[1:]))
        self.transitions[:] = sorted(self.transitions, key=lambda t: (t[0][1:], t[1], t[2][1:]))

        return self

    @staticmethod
    def read_from_console(do_show_messages: bool = True) -> 'FiniteAutomaton':
        """Reads an automaton from the console input.
        :param do_show_messages: whether to show instruction messages or not (default True)
        :return: the automaton"""

        automaton = FiniteAutomaton.new()

        if do_show_messages:
            print('Number of states: ', end='')
        states_count = int(input())

        for i in range(states_count):
            if do_show_messages:
                print(f'State {i + 1}: ', end='')
            automaton.states.append(input())

        if do_show_messages:
            print('Number of symbols: ', end='')
        symbols_count = int(input())
        for i in range(symbols_count):
            if do_show_messages:
                print(f'Symbol {i + 1}: ', end='')
            automaton.alphabet.append(input())

        if do_show_messages:
            print('Initial state: ')
        automaton.initial_state = input()

        if do_show_messages:
            print('Number of final states: ', end='')
        final_states_count = int(input())
        for i in range(final_states_count):
            if do_show_messages:
                print(f'Final state {i + 1}: ', end='')
            automaton.final_states.append(input())

        if do_show_messages:
            print('Number of transitions: ', end='')
        transitions_count = int(input())
        for i in range(transitions_count):
            if do_show_messages:
                print(f'Transition {i + 1}: ', end='')
            first_state, symbol, result_state = input().split(' ')
            automaton.transitions.append((first_state, symbol, result_state))

        return automaton

    @staticmethod
    def read_from_file_path(file_name: str) -> 'FiniteAutomaton':
        """Reads an automaton from the file at the specified path.
        :param file_name: the path of the file
        :return: the automaton"""

        with open(file_name, 'r') as file:
            return FiniteAutomaton.read_from_file_obj(file)

    @staticmethod
    def read_from_file_obj(file) -> 'FiniteAutomaton':
        """Reads an automaton from an open file object.
        :param file: the file object
        :return: the automaton"""

        automaton: FiniteAutomaton = FiniteAutomaton.new()
        automaton.states = file.readline().strip().split(' ')
        automaton.alphabet = file.readline().strip().split(' ')
        automaton.initial_state = file.readline().strip()
        automaton.final_states = file.readline().strip().split(' ')
        while (line := file.readline().strip()) != '':
            line_split: list[str] = line.split(' ')
            automaton.transitions.append((line_split[0], line_split[1], line_split[2]))

        return automaton.sorted()

    def is_deterministic(self) -> bool:
        """Determines whether the automaton is deterministic.
        :param self: the automaton
        :return: True or False"""

        existing_transitions: set[tuple] = set()

        for transition in self.transitions:
            if transition[1] == LAMBDA:
                return False
            if (transition[0], transition[1]) in existing_transitions:
                return False
            existing_transitions.add((transition[0], transition[1]))

        return True

    def is_valid(self) -> bool:
        """Determines whether the automaton is valid.
        :param self: the automaton
        :return: True or False"""

        if len(self.states) == 0 or len(self.alphabet) == 0:
            return False

        if self.initial_state not in self.states:
            return False

        if any(final_state not in self.states for final_state in self.final_states):
            return False

        for transition in self.transitions:
            if len(transition) != 3:
                return False
            if transition[0] not in self.states:
                return False
            if transition[1] not in self.alphabet and transition[1] != LAMBDA:
                return False
            if transition[2] not in self.states:
                return False

        return True

    def accepts(self, word: str) -> bool:
        """Determines whether the automaton accepts a word.
        :param self: the automaton
        :param word: the word
        :return: True or False"""

        current_states: set[str] = self.lambda_closure(self.initial_state)

        for symbol in word:
            next_states: set[str] = {transition[2] for transition in self.transitions
                                     if transition[0] in current_states and transition[1] == symbol}
            current_states = self.lambda_closure_all(next_states)

        for state in current_states:
            if state in self.final_states:
                return True

        return False

    def lambda_closure(self, state: str) -> set[str]:
        """Calculates the lambda closure of a state.
        :param self: the automaton
        :param state: the state
        :return: the lambda closure of the state"""

        if state not in self.states:
            return set()

        closure: set[str] = {state}
        closure_len: int = 0

        while closure_len != len(closure):
            closure_len = len(closure)
            closure |= {transition[2] for transition in self.transitions
                        if transition[0] in closure and transition[1] == LAMBDA}

        return closure

    def lambda_closure_all(self, states: list[str] | set[str]) -> set[str]:
        """Calculates the lambda closure of a set of states.
        :param self: the automaton
        :param states: the set (or list) of states
        :return: the lambda closure of the set of states"""

        return set().union(*(self.lambda_closure(state) for state in states))

        # closure: set[str] = set()
        # for state in states:
        #     closure |= self.lambda_closure(state)
        # return closure

    @staticmethod
    def shift_state(state: str, count: int) -> str:
        """Shifts a state by a certain count. The state must follow the format '[a-zA-Z][0-9]+'.
        :param state: the original state
        :param count: the number of positions to shift the state by
        :return: the new state after shifting it by the count"""

        return state[0] + str(int(state[1:]) + count)

    def shift_states(self, count: int) -> 'FiniteAutomaton':
        """Modifies the original automaton so that all the states are shifted by a certain count.
        :param self: the original automaton
        :param count: the number of positions to shift the states by"""

        self.states[:] = [FiniteAutomaton.shift_state(state, count) for state in self.states]
        self.initial_state = FiniteAutomaton.shift_state(self.initial_state, count)
        self.final_states[:] = [FiniteAutomaton.shift_state(final_state, count)
                                for final_state in self.final_states]
        self.transitions[:] = [(FiniteAutomaton.shift_state(transition[0], count), transition[1],
                                FiniteAutomaton.shift_state(transition[2], count))
                               for transition in self.transitions]
        return self

    def defragmentation(self) -> 'FiniteAutomaton':
        """Modifies the original automaton so that the states all have the format 'q[0-9]+'.
        :param self: the original automaton
        :return: the modified automaton"""
        try:
            fragmented: bool = all(0 <= int(self[i][1:]) < len(self) for i in range(len(self)))
        except ValueError:
            fragmented = True

        if not fragmented:
            return self.sorted().remove_useless_states()

        states_dict: dict[str, str] = {self[i]: f'q{i}' for i in range(len(self))}

        self.states[:] = [states_dict[state] for state in self]
        self.final_states[:] = [states_dict[state] for state in self.final_states]
        self.initial_state = states_dict[self.initial_state]
        self.transitions[:] = [(states_dict[transition[0]], transition[1], states_dict[transition[2]])
                               for transition in self.transitions]
        return self.sorted()

    def to_deterministic(self) -> 'FiniteAutomaton':
        """Modifies the original automaton so that it accepts the same language as the original
        automaton, but is deterministic. The method used is called the 'powerset construction'.
        :param self: the original automaton
        :return: the modified automaton that accepts the same language as the original automaton,
        but is deterministic"""

        if self.is_deterministic():
            return self

        self.defragmentation()

        states_dict: dict[tuple, str] = {}
        next_dict: dict[tuple, str] = {tuple(self.lambda_closure(self.initial_state)): 'q0'}
        new_transitions: list[tuple[str, str, str]] = []

        while next_dict:
            states_dict.update(next_dict)
            next_dict: dict[tuple, str] = {}

            for components, new_name in states_dict.items():
                for symbol in self.alphabet:
                    new_state: set[str] = {transition[2] for old_state in components
                                           for transition in self.transitions
                                           if transition[0] == old_state and transition[1] == symbol}

                    if len(new_state) == 0:
                        continue

                    new_state_tuple: tuple = tuple(self.lambda_closure_all(new_state))

                    if new_state_tuple not in states_dict and new_state_tuple not in next_dict:
                        new_state_result: str = f'q{len(states_dict) + len(next_dict)}'
                        next_dict[new_state_tuple] = new_state_result
                        new_transitions.append((new_name, symbol, new_state_result))
                        continue

                    if new_name not in [transition[0] for transition in new_transitions if symbol == transition[1]]:
                        if new_state_tuple in next_dict:
                            new_transitions.append((new_name, symbol, next_dict[new_state_tuple]))
                        else:
                            new_transitions.append((new_name, symbol, states_dict[new_state_tuple]))

        new_final_states: list[str] = [
            new_name for components, new_name in states_dict.items()
            if any(component in self.final_states for component in components)]

        self.states[:] = [value for key, value in states_dict.items()]
        self.final_states = new_final_states
        self.transitions = new_transitions
        return self

    def alternate(self, other: 'FiniteAutomaton') -> 'FiniteAutomaton':
        """Generates a new automaton that accepts the union of the languages of the original 2 automata.
        :param self: the first automaton
        :param other: the second automaton
        :return: the new automaton that accepts the union of the languages of the original 2 automata"""

        ret: FiniteAutomaton = FiniteAutomaton.new()

        new_final_state: str = f'q{str(len(self.states) + len(other.states) + 1)}'

        self_shifted: FiniteAutomaton = self.shift_states(1)
        other_shifted: FiniteAutomaton = other.shift_states(len(self.states) + 1)

        ret.states = ['q0'] + self_shifted.states + other_shifted.states + [new_final_state]
        ret.alphabet = sorted(list(set(self_shifted.alphabet) | set(other_shifted.alphabet)))
        ret.initial_state = 'q0'
        ret.final_states = [new_final_state]
        ret.transitions = ([('q0', LAMBDA, self_shifted.initial_state)]
                           + [('q0', LAMBDA, other_shifted.initial_state)]
                           + self_shifted.transitions
                           + other_shifted.transitions
                           + [(state, LAMBDA, new_final_state) for state in self_shifted.final_states]
                           + [(state, LAMBDA, new_final_state) for state in other_shifted.final_states])
        return ret

    def concatenate(self, other: 'FiniteAutomaton') -> 'FiniteAutomaton':
        """Generates a new automaton that accepts the concatenation of the languages of the original 2 automata.
        :param self: the first automaton
        :param other: the second automaton
        :return: the new automaton that accepts the concatenation of the languages of the original 2 automata"""

        ret: FiniteAutomaton = FiniteAutomaton.new()
        other_shifted: FiniteAutomaton = other.shift_states(len(self.states) - 1)
        ret.initial_state = self.initial_state
        ret.final_states = copy.deepcopy(other_shifted.final_states)
        ret.states = self.states + other_shifted.states[1:]
        ret.alphabet = sorted(list(set(self.alphabet) | set(other_shifted.alphabet)))
        ret.transitions = self.transitions + other_shifted.transitions

        return ret

    def star(self) -> 'FiniteAutomaton':
        """Generates a new automaton that accepts the Kleene star of the language of the original automaton.
        :param self: the original automaton
        :return: the new automaton that accepts the Kleene star of the language of the original automaton"""

        ret: FiniteAutomaton = FiniteAutomaton.new()

        self_shifted: FiniteAutomaton = self.shift_states(1)
        new_final_state: str = f'q{len(self_shifted.states) + 1}'

        ret.states = ['q0'] + self_shifted.states + [new_final_state]
        ret.alphabet = sorted(copy.deepcopy(self_shifted.alphabet))
        ret.initial_state = 'q0'
        ret.final_states = [new_final_state]
        ret.transitions = (self_shifted.transitions
                           + [(ret.initial_state, LAMBDA, self_shifted.initial_state)]
                           + [(self_shifted.final_states[0], LAMBDA, ret.final_states[0])]
                           + [(ret.initial_state, LAMBDA, ret.final_states[0])]
                           + [(self_shifted.final_states[0], LAMBDA, self_shifted.initial_state)])
        return ret

    def remove_unreachable_states(self) -> 'FiniteAutomaton':
        """Modifies the original automaton so that the states that can't be reached are eliminated.
        :param self: the original automaton
        :return: the modified automaton"""

        reachable_states: set[str] = {self.initial_state}

        change: bool = True
        while change:
            change = False
            for transition in self.transitions:
                if transition[0] in reachable_states and transition[2] not in reachable_states:
                    reachable_states.add(transition[2])
                    change = True

        unreachable_states: set[str] = set(self.states) - reachable_states
        if not unreachable_states:
            return self

        self.states[:] = list(reachable_states)
        self.final_states[:] = list(set(self.final_states) - unreachable_states)
        self.transitions[:] = [transition for transition in self.transitions
                               if transition[0] in reachable_states]
        self.alphabet[:] = [symbol for symbol in self.alphabet
                            if any(transition[1] == symbol for transition in self.transitions)]
        return self.sorted()

    def remove_useless_states(self) -> 'FiniteAutomaton':
        """Modifies the original automaton so that the states that don't result in a final state and the states that
        can't be reached are both eliminated.
        :param self: the original automaton
        :return: the modified automaton"""

        self.remove_unreachable_states()
        productive_states: set[str] = {final_state for final_state in self.final_states}
        productive_len: int = 0

        while productive_len != len(productive_states):
            productive_len = len(productive_states)

            for transition in self.transitions:
                if transition[2] in productive_states:
                    productive_states.add(transition[0])

        if self.initial_state not in productive_states:
            self.empty()
            return self

        self.states[:] = list(productive_states)
        self.final_states[:] = list(set(self.final_states) & productive_states)
        self.transitions[:] = [transition for transition in self.transitions if transition[0] in
                               productive_states and transition[2] in productive_states]

        return self.sorted()

    def insert_sink_if_needed(self) -> 'FiniteAutomaton':
        """Inserts a new state that all the other states will go to
        with the symbols that they don't have transitions for if there are such states.
        :param self: the original automaton
        :return: the modified (or unmodified) automaton"""

        if not any(not any(transition[0] == state and transition[1] == symbol for transition in self.transitions)
                   for symbol in self.alphabet
                   for state in self.states):
            return self

        sink: str = f'q{len(self)}'
        self.states.append(sink)
        self.transitions += [(state, symbol, sink) for state in self for symbol in self.alphabet
                             if not any(transition[0] == state and transition[1] == symbol
                                        for transition in self.transitions)]

        return self.sorted()

    def minimize(self) -> 'FiniteAutomaton':
        """Minimizes the automaton.
        :param self: the original automaton
        :return: the automaton after minimization"""

        if not self.is_deterministic():
            self.to_deterministic().defragmentation()
        else:
            self.defragmentation()

        self.remove_useless_states().insert_sink_if_needed()

        # triangular under-diagonal matrix with False values
        matrix: list[list[int]] = [[0] * (i + 1) for i in range(len(self) - 1)]

        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                state1: str = self[i]
                state2: str = self[j]

                # mark all the different-classed state pairs (one final state and one not final state)
                if (state1 in self.final_states) ^ (state2 in self.final_states):
                    matrix[j - 1][i] = 1

        # initiate the dictionary with empty lists for each ordered pair of states.
        # we map each ordered pair of states with all their possible outcomes (pairs) when following the same symbol
        # in a transition
        pairs: dict[tuple[str, str], list[tuple[int, int]]] = {
            (self[i], self[j]): []
            for i in range(len(self))
            for j in range(i + 1, len(self))}

        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if self.transitions[i][0] == self.transitions[j][0]:  # same starting state
                    continue
                if self.transitions[i][1] != self.transitions[j][1]:  # different symbol
                    continue
                if self.transitions[i][2] == self.transitions[j][2]:  # same result state
                    continue

                # different result state classes (one final and one not final).
                # we don't care about these because they are already marked in the matrix by definition
                if ((self.transitions[i][0] in self.final_states)
                        ^ (self.transitions[j][0] in self.final_states)):
                    continue

                if (self.states.index(self.transitions[i][0])
                        < self.states.index(self.transitions[j][0])):
                    key: tuple[str, str] = (self.transitions[i][0], self.transitions[j][0])
                else:
                    key: tuple[str, str] = (self.transitions[j][0], self.transitions[i][0])

                value1: int = self.states.index(self.transitions[i][2])
                value2: int = self.states.index(self.transitions[j][2])

                if value1 > value2:
                    value1, value2 = value2, value1

                # append the new ordered pair of states to the dictionary's values
                if (value1, value2) not in pairs[key]:
                    pairs[key].append((value1, value2))

        changed: bool = True
        cycles_count: int = 1
        while changed:
            changed = False
            cycles_count += 1
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if matrix[i][j] != 0:
                        continue

                    for pair in pairs[(self.states[j], self.states[i + 1])]:
                        if matrix[pair[1] - 1][pair[0]]:
                            matrix[i][j] = cycles_count
                            changed = True

        # no changes to the matrix => no changes to the automaton => return the automaton from the start (not self)
        if cycles_count <= 2:
            return self

        # "connected components"
        connected_components: list[int] = [i for i in range(len(self))]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    for k in range(len(connected_components)):
                        if connected_components[k] == i + 1:
                            connected_components[k] = j

        # states defragmentation: mapping to consecutive negative numbers, and then back to natural consecutive numbers
        negative_dict: dict[int, int] = {}
        next_negative: int = -1
        for i in range(len(connected_components)):
            if connected_components[i] not in negative_dict:
                negative_dict[connected_components[i]] = next_negative
                next_negative -= 1
        for i in range(len(connected_components)):
            connected_components[i] = (-1 * negative_dict[connected_components[i]]) - 1  # - 1 in the end to get 0-based

        # remapping the states and transitions
        new_states_dict: dict[str, str] = {self[i]: f'q{connected_components[i]}' for i in range(len(self))}
        new_transitions: list[tuple[str, str, str]] = []
        for transition in self.transitions:
            new_transition: tuple[str, str, str] = (new_states_dict[transition[0]], transition[1], new_states_dict[
                transition[2]])
            if new_transition not in new_transitions:
                new_transitions.append(new_transition)

        self.states[:] = list(set(new_states_dict[state] for state in self.states))
        self.initial_state = new_states_dict[self.initial_state]
        self.final_states[:] = list(set(new_states_dict[state] for state in self.final_states))
        self.transitions = new_transitions

        return self.remove_useless_states().sorted()

    def normalize(self) -> 'FiniteAutomaton':
        """Adds a new initial state and a new final state, and connects them
        to the old initial state and final states with lambda transitions.
        :param self: the original automaton
        :return: the normalized automaton"""

        self.shift_states(1)
        new_final_state: str = f'q{len(self) + 1}'

        self.states[:] = ['q0'] + self.states + [new_final_state]
        self.transitions[:] = ([('q0', '', self.initial_state)]
                               + self.transitions
                               + [(state, '', new_final_state) for state in self.final_states])
        self.initial_state = 'q0'
        self.final_states[:] = [new_final_state]

        return self.sorted()

    def merge_parallel_transitions(self) -> 'FiniteAutomaton':
        """Merges parallel transitions with the same starting and ending states into 1 single transition.
        :param self: the original automaton
        :return: the modified automaton"""

        parallels: dict[tuple[str, str], list[str]] = {(t[0], t[2]): [] for t in self.transitions}

        for transition in self.transitions:
            parallels[(transition[0], transition[2])].append(transition[1])

        for states_pair, symbols_list in parallels.items():
            if len(symbols_list) > 1:
                parallels[states_pair] = [symbol if symbol != '' else LAMBDA for symbol in symbols_list]

        self.transitions[:] = [
            (state1, f"({'|'.join(symbols)})" if len(symbols) > 1 else symbols[0], state2)
            for (state1, state2), symbols in parallels.items() if len(symbols) > 0]

        return self.sorted()

    def to_regex(self) -> str:
        """Generates 1 of the regular expressions that represents the language of the automaton.
        :param self: the original automaton
        :return: the regular expression that represents the language of the automaton"""

        self.minimize().normalize()

        while len(self) > 2:
            self.sorted().merge_parallel_transitions()
            state: str = self[1]

            t: list[tuple[str, str, str]] = self.transitions

            in_transitions: list[int] = [i for i in range(len(t))
                                         if t[i][2] == state and t[i][0] != state]
            out_transitions: list[int] = [i for i in range(len(t))
                                          if t[i][0] == state and t[i][2] != state]
            star_loop: str = ''
            for i in range(len(self.transitions)):
                if self.transitions[i][0] == state and self.transitions[i][2] == state:
                    star_loop: str = f'{self.transitions[i][1]}*'
                    break

            cardinal_product: list[tuple[int, int]] = [(in_t, out_t)
                                                       for in_t in in_transitions
                                                       for out_t in out_transitions]

            for in_t, out_t in cardinal_product:
                in_state: str = t[in_t][0]
                out_state: str = t[out_t][2]

                new_expr: str = f'{t[in_t][1]}{star_loop}{t[out_t][1]}'
                if t[in_t][1] != '' and t[out_t][1] != '':
                    new_expr: str = f'({new_expr})'
                t.append((in_state, new_expr, out_state))

            self.transitions = [transition for transition in self.transitions
                                if transition[0] != state and transition[2] != state]
            self.states.pop(1)

        self.merge_parallel_transitions()
        return self.transitions[0][1]
