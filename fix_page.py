import re
from pathlib import Path

p = Path("docs/index.html")
t = p.read_text(encoding="utf-8")

new_alt = ("Three panels: label-free topology score through the run; repeat-counter score through the run; "
           "early-detection curves at matched false-alarm rates.")
new_cap = ("A&ndash;B: label-free topology score and repeat counter through the run, as percentiles of successful "
           "runs (shaded: 95% confidence interval). C: early detection vs false-alarm rate.")

t, n_alt = re.subn(r"Four panels:[^\"]*", new_alt, t)
t, n_cap = re.subn(r"A: H0 bar lengths.*?false-alarm rate\.", new_cap, t, flags=re.DOTALL)
if n_alt != 1 or n_cap != 1:
    raise SystemExit(f"Expected one match each, got alt={n_alt}, caption={n_cap}. Nothing written.")
p.write_text(t, encoding="utf-8", newline="\n")
print("Caption and alt text updated.")