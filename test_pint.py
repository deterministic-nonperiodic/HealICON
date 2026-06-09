import pint
import re

UNITS_REG = pint.UnitRegistry()
cmd = re.compile(r"(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")

def _parse_units(unit_str):
    if isinstance(unit_str, (pint.Quantity, pint.Unit)):
        return unit_str
    else:
        return UNITS_REG(cmd.sub('**', unit_str))

for u in ['m s-1', 'm/s', 'K', 'Pa', 'kg kg-1']:
    try:
        parsed = _parse_units(u)
        sq = (parsed ** 2).units
        print(f"'{u}' -> '{str(sq)}'")
        print(f"'{u}' format ~C -> '{sq:~C}'")
        print(f"'{u}' format ~ -> '{sq:~}'")
    except Exception as e:
        print(f"'{u}' failed: {e}")
