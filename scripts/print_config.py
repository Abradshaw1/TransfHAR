"""
print_config.py
----------------
Loads YAML configs, merges overrides (if any), resolves paths, and prints the final config.
No side effects besides stdout.
"""

# Pseudocode:
# - parse CLI args (config path(s), optional overrides)
# - load base + model/probe config via imu_lm.utils.config
# - resolve paths (runs_root, data_root)
# - print merged config (yaml/json)
# - exit
