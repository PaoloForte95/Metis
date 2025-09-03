from unified_planning.io import PDDLReader

reader = PDDLReader()
problem = reader.parse_problem('test/domains/input_domains/hanoi_domain.pddl')

print("Actions:", problem.actions)
print("Fluents:", problem.fluents)
print("Objects:", problem.objects)
print("Initial state:", problem.initial_values)
print("Goals:", problem.goals)