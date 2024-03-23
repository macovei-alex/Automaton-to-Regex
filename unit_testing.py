from FiniteAutomaton import FiniteAutomaton as FA
from RegularExpression import RegularExpression as Regex

regex_auto_tests_count: int = 5
auto_regex_tests_count: int = 6
dir_path: str = 'unit_tests'


def run_all_tests():
    run_transformation_tests()
    run_minimization_tests()
    run_to_regex_tests()


def run_transformation_tests():
    print('\nTransformation tests')

    for i in range(1, regex_auto_tests_count + 1):
        file_path: str = f'{dir_path}/regex-auto/test{i}.txt'

        regex: Regex = Regex.read(file_path)
        automaton: FA = regex.to_lambda_automaton().to_deterministic()

        test_passed: bool = True

        with open(file_path, 'r') as file:
            file.readline()
            line_counter: int = 2

            while line := file.readline().strip():
                split_line: list[str] = line.split()

                result: bool = automaton.accepts(split_line[0])

                if split_line[1] == 'True':
                    expected: bool = True

                elif split_line[1] == 'False':
                    expected: bool = False

                else:
                    print(f'Test {i}: FAILED on line {line_counter}: invalid expected value {split_line[1]}')
                    test_passed = False
                    break

                if result != expected:
                    print(f'Test {i}: FAILED on line {line_counter}: expected {expected}, got {result}')
                    test_passed = False

                line_counter += 1

        if test_passed:
            print(f'Test {i}: PASSED')


def run_minimization_tests():
    print("\nMinimization tests")

    for i in range(1, regex_auto_tests_count + 1):
        file_path: str = f'{dir_path}/regex-auto/test{i}.txt'

        regex: Regex = Regex.read(file_path)
        automaton: FA = regex.to_lambda_automaton().to_deterministic().minimize()

        test_passed: bool = True

        with open(file_path, 'r') as file:
            file.readline()
            line_counter: int = 2

            while line := file.readline().strip():
                split_line: list[str] = line.split()

                result: bool = automaton.accepts(split_line[0])

                if split_line[1] == 'True':
                    expected: bool = True

                elif split_line[1] == 'False':
                    expected: bool = False

                else:
                    print(f'Test {i}: FAILED on line {line_counter}: invalid expected value {split_line[1]}')
                    test_passed = False
                    break

                if result != expected:
                    print(f'Test {i}: FAILED on line {line_counter}: expected {expected}, got {result}')
                    test_passed = False

                line_counter += 1

        if test_passed:
            print(f'Test {i}: PASSED')


def run_to_regex_tests():
    print("\nTo regex tests")

    for i in range(1, auto_regex_tests_count + 1):
        file_path: str = f'{dir_path}/auto-regex/test{i}.txt'

        if open(file_path, 'r').readline().strip() == '':
            continue

        with open(file_path, 'r') as file:
            regex1: Regex = Regex.read_from_file_obj(file)
            m2: FA = FA.read_from_file_obj(file)

            regex2: Regex = Regex(m2.to_regex()).remove_redundant_parentheses()

            if regex1 == regex2:
                print(f'Test {i}: PASSED')
            else:
                print(f'Test {i}: FAILED')

            print()
            print(regex1)
            print(regex2)
            print()
