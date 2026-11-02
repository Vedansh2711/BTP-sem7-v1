# 1. Update attack_injector.py
with open("attack_injector.py", "r", encoding="utf-8") as f:
    ai_code = f.read()

ai_target = "n_mal = max(1, int(self.p_mal * n)) if self.p_mal > 0 else 0"
ai_replacement = "n_mal = int(round(self.p_mal * n)) if self.p_mal > 0 else 0"

assert ai_target in ai_code, "ai_target not found!"
ai_code = ai_code.replace(ai_target, ai_replacement)

with open("attack_injector.py", "w", encoding="utf-8") as f:
    f.write(ai_code)
print("Updated attack_injector.py successfully!")

# 2. Update exp_false_cpm.py
with open("exp_false_cpm.py", "r", encoding="utf-8") as f:
    exp_code = f.read()

exp_target = "p_mal_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]"
exp_replacement = "p_mal_values = [0.0, 0.125, 0.25, 0.375, 0.50]"

assert exp_target in exp_code, "exp_target not found!"
exp_code = exp_code.replace(exp_target, exp_replacement)

# Update evidence summary check
old_evidence = """    if '0.2' in results.get('BASE (no defence)', {}) and '0.2' in results.get('FULL (A + B + C)', {}):
        base = results['BASE (no defence)']['0.2']
        full = results['FULL (A + B + C)']['0.2']
        print(f"At 20% malicious ICVs:")"""

new_evidence = """    key = '0.25' if '0.25' in results.get('BASE (no defence)', {}) else list(results.get('BASE (no defence)', {}).keys())[1]
    if key in results.get('BASE (no defence)', {}) and key in results.get('FULL (A + B + C)', {}):
        base = results['BASE (no defence)'][key]
        full = results['FULL (A + B + C)'][key]
        print(f"At {float(key):.1%} malicious ICVs ({int(round(float(key)*8))}/8 ICVs):")"""

assert old_evidence in exp_code, "old_evidence not found!"
exp_code = exp_code.replace(old_evidence, new_evidence)

# Update table format for 5.1% so 12.5% prints nicely
exp_code = exp_code.replace("{p_mal:5.0%}", "{p_mal:5.1%}")
exp_code = exp_code.replace("{p_mal:.0%}", "{p_mal:.1%}")

with open("exp_false_cpm.py", "w", encoding="utf-8") as f:
    f.write(exp_code)
print("Updated exp_false_cpm.py successfully!")

