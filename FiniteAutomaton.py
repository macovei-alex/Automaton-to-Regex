import copy

from constants import *


class FiniteAutomaton:
    def __init__(self, states: list[str], alphabet: list[str], initial_state: str, final_states: list[str],
                 transitions: list[tuple[str, str, str]]) -> None:
        self.states: list[str] = copy.deepcopy(states)
        self.alphabet: list[str] = sorted(copy.deepcopy(alphabet))
        self.initial_state: str = initial_state
        self.final_states: list[str] = copy.deepcopy(final_states)
        self.transitions: list[tuple[str, str, str]] = copy.deepcopy(transitions)

    def __str__(self) -> str:
        return (f"States: {self.states}\n"
                f"Alphabet: {self.alphabet}\n"
                f"Initial state: {self.initial_state}\n"
                f"Final states: {self.final_states}\n"
                f"Transitions: \n\t" +
                '\n\t'.join([f"({transition[0]}, {transition[1]}) -> {transition[2]}" for transition in
                             self.transitions]))

    def __bool__(self) -> bool:
        return self.is_valid()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item: int):
        return self.states[item]

    def __eq__(self, other: 'FiniteAutomaton') -> bool:
        m1: FiniteAutomaton = self.to_deterministic().minimize()
        m2: FiniteAutomaton = other.to_deterministic().minimize()

        if m1.alphabet != m2.alphabet:
            return False

        alphabet: list[str] = m1.alphabet
        added_state_pairs: set[tuple[str, str]] = {(m1.initial_state, m2.initial_state)}
        state_pairs: set[tuple[str, str]] = set()

        changed: bool = True

        while changed:
            changed = False
            new_state_pairs: set[tuple[str, str]] = set()

            for pair in added_state_pairs:
                for symbol in alphabet:
                    next1: str = ''
                    next2: str = ''

                    for t1 in m1.transitions:
                        if t1[0] == pair[0] and t1[1] == symbol:
                            next1: str = t1[2]
                            break

                    for t2 in m2.transitions:
                        if t2[0] == pair[1] and t2[1] == symbol:
                            next2: str = t2[2]
                            break

                    if next1 == '' and next2 == '':
                        continue

                    if ((next1, next2) not in state_pairs
                            and (next1, next2) not in added_state_pairs
                            and (next1, next2) not in new_state_pairs):
                        if (next1 in m1.final_states) ^ (next2 in m2.final_states):
                            return False
                        new_state_pairs.add((next1, next2))
                        changed = True

            state_pairs |= added_state_pairs
            added_state_pairs = new_state_pairs

        return True

    @staticmethod
    def new() -> 'FiniteAutomaton':
        return FiniteAutomaton([], [], '', [], [])

    @staticmethod
    def new_primitive(symbol: str) -> 'FiniteAutomaton':
        if symbol == LAMBDA:
            return FiniteAutomaton(['q0'], [], 'q0', ['q0'], [])

        return FiniteAutomaton(['q0', 'q1'], [symbol], 'q0', ['q1'], [('q0', symbol, 'q1')])

    def sorted(self) -> 'FiniteAutomaton':
        self.states[:] = sorted(self.states, key=lambda state: int(state[1:]))
        self.alphabet[:] = sorted(self.alphabet)
        self.final_states[:] = sorted(self.final_states, key=lambda state: int(state[1:]))
        self.transitions[:] = sorted(self.transitions, key=lambda t: (t[0][1:], t[1], t[2][1:]))

        return self

    @staticmethod
    def read_from_console(do_show_messages: bool = True) -> 'FiniteAutomaton':
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
        with open(file_name, 'r') as file:
            return FiniteAutomaton.read_from_file_obj(file)

    @staticmethod
    def read_from_file_obj(file) -> 'FiniteAutomaton':
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
        existing_transitions: set[tuple] = set()

        for transition in self.transitions:
            if transition[1] == LAMBDA:
                return False
            if (transition[0], transition[1]) in existing_transitions:
                return False

            existing_transitions.add((transition[0], transition[1]))

        return True

    def is_valid(self) -> bool:
        if len(self.states) == 0 or len(self.alphabet) == 0:
            return True

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

    def accepts(self, word) -> bool:
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
        return set().union(*(self.lambda_closure(state) for state in states))

        # closure: set[str] = set()
        # for state in states:
        #     closure |= self.lambda_closure(state)
        # return closure

    @staticmethod
    def shift_state(state: str, count: int) -> str:
        return state[0] + str(int(state[1:]) + count)

    def shift_states(self, count: int) -> 'FiniteAutomaton':
        self.states[:] = [FiniteAutomaton.shift_state(state, count) for state in self.states]
        self.initial_state = FiniteAutomaton.shift_state(self.initial_state, count)
        self.final_states[:] = [FiniteAutomaton.shift_state(final_state, count)
                                for final_state in self.final_states]
        self.transitions[:] = [(FiniteAutomaton.shift_state(transition[0], count), transition[1],
                                FiniteAutomaton.shift_state(transition[2], count))
                               for transition in self.transitions]
        return self

    def defragmentation(self) -> 'FiniteAutomaton':
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
        ret: FiniteAutomaton = FiniteAutomaton.new()
        other_shifted: FiniteAutomaton = other.shift_states(len(self.states) - 1)
        ret.initial_state = self.initial_state
        ret.final_states = copy.deepcopy(other_shifted.final_states)
        ret.states = self.states + other_shifted.states[1:]
        ret.alphabet = sorted(list(set(self.alphabet) | set(other_shifted.alphabet)))
        ret.transitions = self.transitions + other_shifted.transitions

        return ret

    def star(self) -> 'FiniteAutomaton':
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
            return copy.deepcopy(self)

        self.states = list(reachable_states)
        self.final_states = list(set(self.final_states) - unreachable_states)
        self.transitions = [transition for transition in self.transitions if transition[0] in reachable_states]
        self.alphabet = [symbol for symbol in self.alphabet
                         if any(transition[1] == symbol for transition in self.transitions)]
        return self.sorted()

    def remove_useless_states(self) -> 'FiniteAutomaton':
        self.remove_unreachable_states()
        productive_states: set[str] = {final_state for final_state in self.final_states}
        productive_len: int = 0

        while productive_len != len(productive_states):
            productive_len = len(productive_states)

            for transition in self.transitions:
                if transition[2] in productive_states:
                    productive_states.add(transition[0])

        if self.initial_state not in productive_states:
            return FiniteAutomaton.new()

        self.states = list(productive_states)
        self.final_states = list(set(self.final_states) & productive_states)
        self.transitions = [transition for transition in self.transitions if transition[0] in
                            productive_states and transition[2] in productive_states]

        return self.sorted()

    def insert_sink_if_needed(self) -> 'FiniteAutomaton':
        if not any(not any(transition[0] == state and transition[1] == symbol for transition in self.transitions)
                   for symbol in self.alphabet
                   for state in self.states):
            return copy.deepcopy(self)

        automaton: FiniteAutomaton = copy.deepcopy(self)

        sink: str = f'q{len(automaton)}'
        automaton.states.append(sink)
        automaton.transitions += [(state, symbol, sink) for state in automaton for symbol in automaton.alphabet
                                  if not any(transition[0] == state and transition[1] == symbol
                                             for transition in automaton.transitions)]

        return automaton.sorted()

    def minimize(self) -> 'FiniteAutomaton':
        if not self.is_deterministic():
            automaton: FiniteAutomaton = self.to_deterministic().defragmentation()
        else:
            automaton: FiniteAutomaton = self.defragmentation()

        automaton = automaton.insert_sink_if_needed()

        # triangular under-diagonal matrix with False values
        matrix: list[list[int]] = [[0] * (i + 1) for i in range(len(automaton) - 1)]

        for i in range(len(automaton)):
            for j in range(i + 1, len(automaton)):
                state1: str = automaton[i]
                state2: str = automaton[j]

                # mark all the different-classed state pairs (one final state and one not final state)
                if (state1 in automaton.final_states) ^ (state2 in automaton.final_states):
                    matrix[j - 1][i] = 1

        # initiate the dictionary with empty lists for each ordered pair of states.
        # we map each ordered pair of states with all their possible outcomes (pairs) when following the same symbol
        # in a transition
        pairs: dict[tuple[str, str], list[tuple[int, int]]] = {
            (automaton[i], automaton[j]): []
            for i in range(len(automaton))
            for j in range(i + 1, len(automaton))}

        for i in range(len(automaton.transitions)):
            for j in range(len(automaton.transitions)):
                t1 = automaton.transitions[i]
                t2 = automaton.transitions[j]

                if automaton.transitions[i][0] == automaton.transitions[j][0]:  # same starting state
                    continue
                if automaton.transitions[i][1] != automaton.transitions[j][1]:  # different symbol
                    continue
                if automaton.transitions[i][2] == automaton.transitions[j][2]:  # same result state
                    continue

                # different result state classes (one final and one not final).
                # we don't care about these because they are already marked in the matrix by definition
                if ((automaton.transitions[i][0] in automaton.final_states)
                        ^ (automaton.transitions[j][0] in automaton.final_states)):
                    continue

                if (automaton.states.index(automaton.transitions[i][0])
                        < automaton.states.index(automaton.transitions[j][0])):
                    key: tuple[str, str] = (automaton.transitions[i][0], automaton.transitions[j][0])
                else:
                    key: tuple[str, str] = (automaton.transitions[j][0], automaton.transitions[i][0])

                value1: int = automaton.states.index(automaton.transitions[i][2])
                value2: int = automaton.states.index(automaton.transitions[j][2])

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

                    for pair in pairs[(automaton.states[j], automaton.states[i + 1])]:
                        if matrix[pair[1] - 1][pair[0]]:
                            matrix[i][j] = cycles_count
                            changed = True

        # no changes to the matrix => no changes to the automaton => return the automaton from the start (not self)
        if cycles_count <= 2:
            return automaton

        # "connected components"
        connected_components: list[int] = [i for i in range(len(automaton))]
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
        new_states_dict: dict[str, str] = {automaton[i]: f'q{connected_components[i]}' for i in range(len(automaton))}
        new_transitions: list[tuple[str, str, str]] = []
        for transition in automaton.transitions:
            new_transition: tuple[str, str, str] = (new_states_dict[transition[0]], transition[1], new_states_dict[
                transition[2]])
            if new_transition not in new_transitions:
                new_transitions.append(new_transition)

        # creating the new automaton with sorted states and transition
        return FiniteAutomaton(
            list(set(new_states_dict[state] for state in automaton.states)),
            automaton.alphabet,
            new_states_dict[automaton.initial_state],
            list(set(new_states_dict[state] for state in automaton.final_states)),
            new_transitions).remove_useless_states().sorted()

    def normalize(self) -> 'FiniteAutomaton':
        automaton: FiniteAutomaton = self.shift_states(1)
        new_final_state: str = f'q{len(automaton) + 1}'

        automaton.states = ['q0'] + automaton.states + [new_final_state]
        automaton.transitions = ([('q0', '', automaton.initial_state)]
                                 + automaton.transitions
                                 + [(state, '', new_final_state) for state in automaton.final_states])
        automaton.initial_state = 'q0'
        automaton.final_states = [new_final_state]

        return automaton.sorted()

    def merge_parallel_transitions(self) -> 'FiniteAutomaton':
        automaton: FiniteAutomaton = copy.deepcopy(self)

        parallels: dict[tuple[str, str], list[str]] = {(t[0], t[2]): [] for t in automaton.transitions}

        for transition in automaton.transitions:
            parallels[(transition[0], transition[2])].append(transition[1])

        for states_pair, symbols_list in parallels.items():
            if len(symbols_list) > 1:
                parallels[states_pair] = [symbol if symbol != '' else LAMBDA for symbol in symbols_list]

        automaton.transitions = [
            (state1, f"({'|'.join(symbols)})" if len(symbols) > 1 else symbols[0], state2)
            for (state1, state2), symbols in parallels.items() if len(symbols) > 0]

        return automaton.sorted()

    def to_regex(self) -> str:
        automaton: FiniteAutomaton = self.minimize().normalize()

        while len(automaton) > 2:
            automaton = automaton.sorted()
            automaton = automaton.merge_parallel_transitions()
            automaton = automaton.sorted()
            state: str = automaton[1]

            t: list[tuple[str, str, str]] = automaton.transitions

            in_transitions: list[int] = [i for i in range(len(t))
                                         if t[i][2] == state and t[i][0] != state]
            out_transitions: list[int] = [i for i in range(len(t))
                                          if t[i][0] == state and t[i][2] != state]
            star_loop: str = ''
            for i in range(len(automaton.transitions)):
                if automaton.transitions[i][0] == state and automaton.transitions[i][2] == state:
                    star_loop: str = f'{automaton.transitions[i][1]}*'
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

            automaton.transitions = [transition for transition in automaton.transitions
                                     if transition[0] != state and transition[2] != state]
            automaton.states.pop(1)

        automaton = automaton.merge_parallel_transitions()
        return automaton.transitions[0][1]
