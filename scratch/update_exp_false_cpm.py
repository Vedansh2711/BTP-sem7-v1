with open("exp_false_cpm.py", "r", encoding="utf-8") as f:
    code = f.read()

target = """                    # Attack injection
                    if self.injector.is_malicious(int(v_gid)):
                        attacked = self.injector.attack(honest, self.rng, m.L)
                        if attacked is None:
                            # Suppression attack - don't add any report
                            continue
                        report = attacked
                        is_malicious_report = True
                    else:
                        report = honest
                        is_malicious_report = False

                    # Tag for metric tracking (NOT visible to defence)
                    report._is_malicious = is_malicious_report
                    cpms[int(u_gid)].append(report)"""

replacement = """                    # Attack injection
                    if self.injector.is_malicious(int(v_gid)):
                        attacked = self.injector.attack(honest, self.rng, m.L)
                        if attacked is None:
                            # Suppression attack - report was omitted/withheld
                            continue
                        report = attacked
                        # Genuinely falsified if content was modified (ghost, shift, quality inflation)
                        # When a suppression attacker occasionally sends honest data, content is truthful
                        is_malicious_report = (attacked is not honest)
                    else:
                        report = honest
                        is_malicious_report = False

                    # Tag for metric tracking (NOT visible to defence)
                    report._is_malicious = is_malicious_report
                    cpms[int(u_gid)].append(report)"""

target_was_accepted = """                    reporter_trust = trust_vec.get(r.reporter_id, 1.0)
                    was_accepted = reporter_trust >= 0.3"""

replacement_was_accepted = """                    reporter_trust = trust_vec.get(r.reporter_id, 1.0)
                    was_accepted = (reporter_trust >= self.defence.cfg.quarantine_thresh)"""

assert target in code, "target not found!"
assert target_was_accepted in code, "target_was_accepted not found!"

code = code.replace(target, replacement)
code = code.replace(target_was_accepted, replacement_was_accepted)

with open("exp_false_cpm.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Applied FAR update to exp_false_cpm.py successfully!")

