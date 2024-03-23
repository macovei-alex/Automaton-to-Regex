from RegularExpression import RegularExpression as Regex
from FiniteAutomaton import FiniteAutomaton as FA
import unit_testing


def main():
    # regex: Regex = Regex.read('resources/regex.txt')
    # print(regex)
    #
    # automaton: FA = regex.to_lambda_automaton()
    # print(automaton)
    #
    # automaton = automaton.to_deterministic()
    # print(automaton)
    #
    # automaton = automaton.minimize()
    # print(automaton)
    #
    # regex: Regex = Regex(automaton.to_regex())
    # print(regex)
    #
    # regex = regex.remove_redundant_parentheses()
    # print(regex)

    unit_testing.run_all_tests()


if __name__ == '__main__':
    main()
