from owlready2 import *
from owlready2 import get_ontology, onto_path
from unified_planning.shortcuts import *
from unified_planning.io import PDDLWriter

class OntoPopulator:
    def __init__(self, ontology_dir="planonto/models", ontology_file="plan-ontology-rdf.owl"):
        self.ontology_dir = ontology_dir
        self.ontology_file = ontology_file
        onto_path.append(self.ontology_dir)
        self.onto = get_ontology(ontology_file).load()

        # Aliases to common classes, assuming they exist in ontology
        self.PlanningDomain = self.onto.search_one(iri="*#PlanningDomain")
        self.DomainAction = self.onto.search_one(iri="*#DomainAction")
        self.DomainPredicate = self.onto.search_one(iri="*#DomainPredicate")
        self.DomainRequirement = self.onto.search_one(iri="*#DomainRequirement")
        self.ActionPrecondition = self.onto.search_one(iri="*#ActionPrecondition")
        self.ActionEffect = self.onto.search_one(iri="*#ActionEffect")
        self.Parameter = self.onto.search_one(iri="*#Parameter")

        self.hasAction = self.onto.search_one(iri="*#hasAction")
        self.hasPredicate = self.onto.search_one(iri="*#hasPredicate")
        self.hasRequirement = self.onto.search_one(iri="*#hasRequirement")

        if not all([self.PlanningDomain, self.DomainAction, self.hasAction]):
            print("Warning: Some expected ontology classes/properties not found. Check your OWL file.")

    def create_domain(self, name):
        domain = self.PlanningDomain(name)
        print(f"Created PlanningDomain: {domain.name}")
        return domain

    def add_action_to_domain(self, domain, action_name):
        action = self.DomainAction(action_name)
        domain.hasAction.append(action)
        print(f"Added DomainAction '{action_name}' to {domain.name}")
        return action

    def add_predicate_to_domain(self, domain, predicate_name):
        predicate = self.DomainPredicate(predicate_name)
        domain.hasPredicate.append(predicate)
        print(f"Added DomainPredicate '{predicate_name}' to {domain.name}")
        return predicate

    def add_precondition_to_action(self, action, precondition_name):
        precondition = self.ActionPrecondition(precondition_name)
        action.hasPrecondition.append(precondition)
        print(f"Added ActionPrecondition '{precondition_name}' to {action.name}")
        return precondition

    def add_effect_to_action(self, action, effect_name):
        effect = self.ActionEffect(effect_name)
        action.hasEffect.append(effect)
        print(f"Added Effect '{effect_name}' to {action.name}")
        return effect

    def add_parameter_to_action(self, action, parameter_name):
        parameter = self.Parameter(parameter_name)
        action.hasParameter.append(parameter)
        print(f"Added Parameter '{parameter_name}' to {action.name}")
        return parameter

    def add_requirement_to_domain(self, domain, requirement_name):
        requirement = self.DomainRequirement(requirement_name)
        domain.hasRequirement.append(requirement)
        print(f"Added DomainRequirement '{requirement_name}' to {domain.name}")
        return requirement

    def save(self, filename="planonto/models/ontoviplan/test-output.owl"):
        self.onto.save(file=filename, format="rdfxml")
        print(f"Ontology saved to {filename}")

class DomainGenerator:
    def __init__(self, ontology_dir="planonto/models/ontoviplan", ontology_file="test-output.owl"):
        onto_path.append(ontology_dir)
        self.onto = get_ontology(ontology_file).load()
        self.save_path = 'plans/'
        # Cache some likely suspects from ontology
        self.DomainAction = self.onto.search_one(iri="*#DomainAction")
        self.PlanningDomain = self.onto.search_one(iri="*#PlanningDomain")
        self.DomainPredicate = self.onto.search_one(iri="*#DomainPredicate")
        self.hasAction = self.onto.search_one(iri="*#hasAction")

    def generate_test_domain(self, domain_name="robot"):
        # Declaring types
        Location = UserType("Location")

        # Creating fluents
        robot_at = Fluent("robot_at", BoolType(), location=Location)
        battery_charge = Fluent("battery_charge", RealType(0, 100))

        # Creating an action
        move = InstantaneousAction("move", l_from=Location, l_to=Location)
        l_from = move.parameter("l_from")
        l_to = move.parameter("l_to")
        move.add_precondition(GE(battery_charge, 10))
        move.add_precondition(robot_at(l_from))
        move.add_precondition(Not(robot_at(l_to)))
        move.add_effect(robot_at(l_from), False)
        move.add_effect(robot_at(l_to), True)
        move.add_effect(battery_charge, Minus(battery_charge, 10))

        # Create the problem
        problem = Problem(domain_name)
        problem.add_fluent(robot_at)
        problem.add_fluent(battery_charge)
        problem.add_action(move)

        # Add objects
        l1 = Object("l1", Location)
        l2 = Object("l2", Location)
        problem.add_object(l1)
        problem.add_object(l2)

        # Initial state
        problem.set_initial_value(robot_at(l1), True)
        problem.set_initial_value(robot_at(l2), False)
        problem.set_initial_value(battery_charge, 100)

        # Goals
        problem.add_goal(robot_at(l2))

        return problem

    def export_pddl(self, problem: Problem, domain_name="test_domain.pddl", problem_name="test_problem.pddl"):
        writer = PDDLWriter(problem)
        domain_path = self.save_path + problem_name
        problem_path = self.save_path + domain_name
        writer.write_domain(domain_path)
        writer.write_problem(problem_path)
        print(f"Wrote domain to: {domain_path}")
        print(f"Wrote problem to: {problem_path}")

