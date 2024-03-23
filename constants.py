LAMBDA: str = '$'

OPERATORS: dict[str, str] = {
    'alternation': '|',     # precedence 0
    'concatenation': '.',   # precedence 1
    'star': '*',            # precedence 2
    'lambda': ''            # precedence 3
}

OP_PRECEDENCE: dict[str, int] = {}

precedence_count: int = 0

for _, _value in OPERATORS.items():
    OP_PRECEDENCE[_value] = precedence_count
    precedence_count += 1
OP_PRECEDENCE['('] = -1
OP_PRECEDENCE[')'] = -1

del precedence_count
