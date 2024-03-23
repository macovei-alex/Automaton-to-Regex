import copy
import string

from constants import *
from FiniteAutomaton import FiniteAutomaton as FA


class RegularExpression:
    def __init__(self, expr: str) -> None:
        self.expr: str = expr

    def __str__(self) -> str:
        return self.expr

    def __eq__(self, other: 'RegularExpression') -> bool:
        return self.to_lambda_automaton() == other.to_lambda_automaton()

    def __getitem__(self, item: int):
        return self.expr[item]

    @staticmethod
    def new() -> 'RegularExpression':
        return RegularExpression('')

    @staticmethod
    def read(file_name: str = '') -> 'RegularExpression':
        if file_name != '':
            return RegularExpression.read_from_file_obj(open(file_name, 'r'))

        regex = RegularExpression.new()

        regex.expr = input()
        for ws in string.whitespace:
            regex.expr = regex.expr.replace(ws, '')

        return regex

    @staticmethod
    def read_from_file_obj(file) -> 'RegularExpression':
        regex = RegularExpression.new()
        regex.expr = file.readline()
        for ws in string.whitespace:
            regex.expr = regex.expr.replace(ws, '')

        return regex

    def check_brackets(self):
        stack: list[str] = []
        for symbol in self.expr:

            if symbol == '(':
                stack.append(symbol)

            elif symbol == ')':
                if not stack:
                    return False

                stack.pop()

        if stack:
            return False

        return True

    def is_valid(self):
        if self.expr == '':
            return True

        if self.expr == LAMBDA:
            return True

        for symbol in self.expr:
            if (not symbol.isalnum()
                    and not symbol == '('
                    and not symbol == ')'
                    and not symbol == LAMBDA
                    and symbol not in OPERATORS.values()):
                return False

        if not self.check_brackets():
            return False

        return True

    def remove_concatenations(self) -> 'RegularExpression':
        return RegularExpression(self.expr.replace(OPERATORS['concatenation'], ''))

    def insert_concatenations(self) -> 'RegularExpression':
        expr: str = self.remove_concatenations().expr
        if expr == '':
            return RegularExpression(expr)

        new_expr: str = ''

        for i in range(len(expr) - 1):
            new_expr += expr[i]

            if expr[i] == OPERATORS['alternation'] or expr[i] == '(':
                continue

            if (expr[i + 1] == OPERATORS['star']
                    or expr[i + 1] == ')'
                    or expr[i + 1] == OPERATORS['alternation']):
                continue

            new_expr += OPERATORS['concatenation']

        new_expr += expr[-1]
        return RegularExpression(new_expr)

    def reverse_polish_notation(self) -> 'RegularExpression':
        expr: str = self.insert_concatenations().expr
        if expr == '':
            return RegularExpression(expr)

        polish_form: str = ''
        operator_stack: list[str] = []

        for symbol in expr:

            if symbol.isalnum() or symbol == LAMBDA:
                polish_form += symbol
                continue

            if symbol == '(':
                operator_stack.append(symbol)
                continue

            if symbol == ')':
                while operator_stack and operator_stack[-1] != '(':
                    polish_form += operator_stack[-1]
                    operator_stack.pop()

                if operator_stack:
                    operator_stack.pop()

                continue

            if symbol in OPERATORS.values():
                while operator_stack and OP_PRECEDENCE.get(operator_stack[-1]) >= OP_PRECEDENCE.get(symbol):
                    polish_form += operator_stack[-1]
                    operator_stack.pop()
                operator_stack.append(symbol)
                continue

            raise ValueError(f'Invalid symbol: {symbol}')

        while operator_stack:
            polish_form += operator_stack[-1]
            operator_stack.pop()

        return RegularExpression(polish_form)

    def to_lambda_automaton(self) -> 'FA':
        if self.expr == '' or all(symbol == '(' or symbol == ')' for symbol in self.expr):
            return FA.new_primitive(LAMBDA)

        rpn: str = self.reverse_polish_notation().expr
        automata_stack: list[FA] = []

        for index in range(len(rpn)):
            if rpn[index].isalnum() or rpn[index] == LAMBDA:
                automata_stack.append(FA.new_primitive(rpn[index]))

            elif rpn[index] == OPERATORS['alternation']:
                m2: FA = automata_stack.pop()
                m1: FA = automata_stack.pop()
                automata_stack.append(m1.alternate(m2))

            elif rpn[index] == OPERATORS['concatenation']:
                m2: FA = automata_stack.pop()
                m1: FA = automata_stack.pop()
                automata_stack.append(m1.concatenate(m2))

            elif rpn[index] == OPERATORS['star']:
                m: FA = automata_stack.pop()
                automata_stack.append(m.star())

        if len(automata_stack) != 1:
            raise ValueError('Invalid regular expression')

        return automata_stack[0].defragmentation()

    def remove_redundant_parentheses(self) -> 'RegularExpression':
        rpn: str = self.reverse_polish_notation().expr
        if rpn == '':
            return RegularExpression(rpn)

        stack: list[tuple[str, str]] = []

        for symbol in rpn:
            if symbol in OPERATORS.values():

                if symbol == OPERATORS['star']:
                    elem: tuple[str, str] = stack.pop()
                    if elem[1] != OPERATORS['lambda']:
                        stack.append((f"({elem[0]}){OPERATORS['star']}", OPERATORS['star']))
                    else:
                        stack.append((f"{elem[0]}{OPERATORS['star']}", OPERATORS['star']))

                elif symbol != OPERATORS['star']:
                    elem2: tuple[str, str] = stack.pop()
                    elem1: tuple[str, str] = stack.pop()

                    if OP_PRECEDENCE[elem1[1]] < OP_PRECEDENCE[symbol]:
                        elem1 = (f'({elem1[0]})', elem1[1])
                    if OP_PRECEDENCE[elem2[1]] < OP_PRECEDENCE[symbol]:
                        elem2 = (f'({elem2[0]})', elem2[1])

                    stack.append((f'{elem1[0]}{symbol}{elem2[0]}', symbol))

            else:
                stack.append((symbol, OPERATORS['lambda']))

        return RegularExpression(stack.pop()[0]).remove_concatenations()

    def simplify(self) -> 'RegularExpression':
        rpn: str = self.reverse_polish_notation().expr
        if rpn == '':
            return RegularExpression(rpn)

        stack: list[tuple[str, str]] = []

        for symbol in rpn:
            if symbol in OPERATORS.values():

                if symbol == OPERATORS['star']:
                    elem: tuple[str, str] = stack.pop()
                    if elem[1] != OPERATORS['lambda']:
                        stack.append((f"({elem[0]}){OPERATORS['star']}", OPERATORS['star']))
                    else:
                        stack.append((f"{elem[0]}{OPERATORS['star']}", OPERATORS['star']))

                elif symbol != OPERATORS['star']:
                    elem2: tuple[str, str] = stack.pop()
                    elem1: tuple[str, str] = stack.pop()

                    if OP_PRECEDENCE[elem1[1]] < OP_PRECEDENCE[symbol]:
                        elem1 = (f'({elem1[0]})', elem1[1])
                    if OP_PRECEDENCE[elem2[1]] < OP_PRECEDENCE[symbol]:
                        elem2 = (f'({elem2[0]})', elem2[1])

                    stack.append((f'{elem1[0]}{symbol}{elem2[0]}', symbol))

            else:
                stack.append((symbol, OPERATORS['lambda']))

            if RegularExpression(stack[-1][0]) == self:
                return RegularExpression(stack[-1][0]).remove_concatenations()

        return RegularExpression(stack.pop()[0]).remove_concatenations()
