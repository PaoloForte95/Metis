from rdflib import Graph, Namespace, RDF, RDFS
import logging
from owlready2 import *
from owlready2 import get_ontology, onto_path
from unified_planning.shortcuts import *
from unified_planning.io import PDDLWriter
from unified_planning.io import PDDLReader
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from rdflib import Graph, Namespace
import urllib.parse
import uuid
import os
# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # sometimes used by huggingface
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

class PlanOnto4UniPlan:
    def __init__(self, ontology_dir="planonto/models", ontology_file="plan-ontology-rdf.owl"):
        self.ontology_dir = ontology_dir
        self.ontology_file = ontology_file
        onto_path.append(self.ontology_dir)
        self.onto = get_ontology(ontology_file).load()
        print("Annotation properties:", list(self.onto.annotation_properties()))
        # Aliases to common classes, assuming they exist in ontology
        self.PlanningDomain = self.onto.search_one(iri="*#PlanningDomain")
        self.DomainAction = self.onto.search_one(iri="*#DomainAction")
        self.DomainPredicate = self.onto.search_one(iri="*#DomainPredicate")
        self.DomainPredicateType = self.onto.search_one(iri="*#DomainPredicateType")
        self.DomainPredicateParameter = self.onto.search_one(iri="*#DomainPredicateParameter")
        self.DomainRequirement = self.onto.search_one(iri="*#DomainRequirement")
        self.ActionPrecondition = self.onto.search_one(iri="*#ActionPrecondition")
        self.ActionEffect = self.onto.search_one(iri="*#ActionEffect")
        self.ActionParameter = self.onto.search_one(iri="*#ActionParameter")
        self.AssignmentEffect = self.onto.search_one(iri="*#AssignmentEffect")
        # Aliases to common object types
        self.object = self.onto.search_one(iri="*#object")

        # Add connectives
        self.And = self.onto.And
        self.Or = self.onto.Or
        self.Not = self.onto.Not
        self.AtomicFormula = self.onto.search_one(iri="*#AtomicFormula")

        # Add predicates
        self.hasAction = self.onto.search_one(iri="*#hasAction")
        self.hasPredicate = self.onto.search_one(iri="*#hasPredicate")
        self.hasRequirement = self.onto.search_one(iri="*#hasRequirement")
        self.hasPrecondition = self.onto.search_one(iri="*#hasActionPrecondition")
        self.hasDomainPredicateName = self.onto.search_one(iri="*#hasDomainPredicateName")
        self.hasDomainPredicateType = self.onto.search_one(iri="*#hasDomainPredicateType")
        self.hasDomainPredicateParameter = self.onto.search_one(iri="*#hasDomainPredicateType")
        self.belongsToDomain = self.onto.search_one(iri="*#belongsToDomain")
        self.belongsToAction = self.onto.search_one(iri="*#belongsToAction")
        self.hasSyntax = self.onto.hasSyntax
        self.hasGeneralDomainDescription = self.onto.search_one(iri="*#hasGeneralDomainDescription")
        # Utility variables
        self.predicate_lookups = {}
        self.param_lookups = {}
        self.current_domain_name = None
        # Add types
        self.types = {}
        TypeClass = self.onto.search_one(iri="*#Type")
        if TypeClass is None:
            logging.warning("Type class not found in ontology.")
        else:
            for t in TypeClass.subclasses():
                self.types[t.name] = t
            logging.info(f"Loaded planning types: {list(self.types.keys())}")
        logging.debug(f"self.ActionParameter: {self.ActionParameter}")
        logging.debug(f"self.DomainAction: {self.DomainAction}")
        logging.debug(f"self.PlanningDomain: {self.PlanningDomain}")
        logging.debug(f"self.DomainPredicate: {self.DomainPredicate}")
        logging.debug(f"self.hasAction: {self.hasAction}")
        if not all([self.PlanningDomain, self.DomainAction, self.hasAction]):
            logging.info("Warning: Some expected ontology classes/properties not found. Check your OWL file.")


    # ==================================
    # Debugging functions
    # ----------------------------------

    def _print_up_problem(self, problem):
        logging.info("== Problem Name ==")
        logging.info(f"Problem name: {problem.name}\n")
        logging.info(f"Problem type: {problem.kind}\n")

        logging.info("== Requirements ==")
        for feature in problem.kind.features:
            logging.info(f"  {feature}")

        logging.info("== Types ==")
        for t in problem.user_types:
            logging.info(f"  {t} (Parent: {t.father if t.father else 'None'})")
        logging.info()
        
        logging.info("== Objects ==")
        for t in problem.user_types:
            for obj in problem.objects(t.name):
                logging.info(f"  {obj.name} : {obj.type}")
        logging.info()
        
        logging.info("== Fluents ==")
        for fl in problem.fluents:
            logging.info(f"  {fl.name}({', '.join(str(p) for p in fl.signature)}) : {fl.type}")
        logging.info()
        
        logging.info("== Actions ==")
        for act in problem.actions:
            logging.info(f"  {act.name}({', '.join(str(p) for p in act.parameters)})")
            logging.info("    Preconditions:")
            if hasattr(act, 'preconditions'):
                for pre in act.preconditions:
                    logging.info(f"      {pre}")
            if hasattr(act, 'effects'):
                logging.info("    Effects:")
                for eff in act.effects:
                    logging.info(f"      {eff}")
            logging.info()
        
        logging.info("== Initial Values ==")
        for fl, val in problem.initial_values.items():
            logging.info(f"  {fl} = {val}")
        logging.info()
        
        logging.info("== Goals ==")
        for g in problem.goals:
            logging.info(f"  {g}")
        logging.info()

    def _print_onto_classes_and_instances(self):
        logging.info("== Ontology Classes and Instances ==")
        for cls in self.onto.classes():
            logging.info(f"Class: {cls.name}")
            instances = list(cls.instances())
            if instances:
                for inst in instances:
                    logging.info(f"  - Instance: {inst.name}")
            else:
                logging.info("  (no instances)")
        logging.info("== End ==")

    def _print_predicate_info(self, predicate):
        print("Fluent:")
        print(f"  Name: {predicate.name}")
        print(f"  Type: {predicate.type}")
        print(f"  Parameters: {predicate.signature}")  # dict of param_name: type
        print(f"  Arity: {predicate.arity}")
        print(f"  Default initial value: {getattr(predicate, 'default_initial_value', None)}")
        print("-" * 40)

    def _print_precondition_info(self, expr, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{type(expr).__name__}: {expr}")
        if hasattr(expr, "args") and expr.args:
            for arg in expr.args:
                self._print_precondition_info(arg, indent + 1)

    def _print_effect(self, effect, indent=0):
        prefix = "  " * indent
        if hasattr(effect, "fluent"):
            fluent = effect.fluent() if callable(effect.fluent) else effect.fluent
            # Drill down to underlying Fluent object
            if hasattr(fluent, "fluent"):
                inner_fluent = fluent.fluent() if callable(fluent.fluent) else fluent.fluent
                fluent_name = inner_fluent.name
                fluent_args = fluent.args
                print(f"{prefix}{fluent_name}({', '.join(str(arg) for arg in fluent_args)}) := {effect.value}")
            else:
                print(f"{prefix}Unknown fluent structure: {fluent}")
        elif hasattr(effect, "condition"):
            print(f"{prefix}when (")
            self.print_formula(effect.condition, indent + 1)
            print(f"{prefix}) do (")
            self._print_effect(effect.effect, indent + 1)
            print(f"{prefix})")
        else:
            print(f"{prefix}Unknown effect type: {type(effect)}")

    def _print_effects_of_action(self, action):
        print(f"Effects for action: {action.name}")
        for effect in getattr(action, "effects", []):
            self._print_effect(effect, indent=1)

    def _print_up_types(self, problem):
        print("== Types in UP Problem ==")
        for t in problem.user_types:
            # Each type has a name and an optional parent (father)
            parent = t.father.name if t.father else "None"
            print(f"  {t.name} (Parent: {parent})")

    # ==================================
    # PDDL -> PlanOnto Parsing Functions
    # ----------------------------------

    def create_domain(self, name):
        domain = self.PlanningDomain(name)
        logging.info(f"Created PlanningDomain: {domain.name}")
        return domain

    def add_action_to_domain(self, domain, action_name):
        action = self.DomainAction(action_name)
        domain.hasAction.append(action)
        logging.info(f"Added DomainAction to {domain.name} '{action_name}'")
        return action

    def add_type_recursively(self, up_type):
        """
        Recursively adds UP type and its parent(s) as OWL classes.
        Returns the created/retrieved OWL class.
        """
        import types
        if up_type.name in self.types:
            return self.types[up_type.name]

        # Determine the parent class
        if up_type.father:  # There is a parent type
            parent_cls = self.add_type_recursively(up_type.father)
        else:
            parent_cls = self.object

        # Create new class as a subclass of its parent
        type_cls = types.new_class(up_type.name, (parent_cls,))
        self.types[up_type.name] = type_cls
        print(f"Created class {up_type.name} as subclass of {parent_cls.name}")
        return type_cls

    def add_types(self, problem):
        for t in problem.user_types:
            self.add_type_recursively(t)

    def add_parameters_to_predicate(self, domainpredicate, parameters):
        for param in parameters:
            unique_name = self.make_unique_name(param.name)
            param_indiv = self.DomainPredicateParameter(unique_name)
            # Link to the type class
            param_type_class = self.types[param.type.name]
            param_indiv.hasParameterType.append(param_type_class)
            # Link to the predicate
            domainpredicate.hasDomainPredicateParameter.append(param_indiv)
            logging.info(f"Added DomainPredicateParameter '{param.name}' of type '{param.type.name}'")
            # Optionally, return or store param_indiv if you want

    def add_predicate_to_domain(self, domain, predicate):
        unique_name = self.make_unique_name(predicate.name)
        domainpredicate = self.DomainPredicate(unique_name)
        domainpredicate.label = [locstr(str(predicate), lang="en")]
        domainPredicateType = self.DomainPredicateType(predicate.type)
        domain.hasPredicate.append(domainpredicate)
        domainpredicate.hasDomainPredicateType.append(domainPredicateType)
        self.add_parameters_to_predicate(domainpredicate, predicate.signature)
        logging.info(f"Added DomainPredicate to {domain.name} '{predicate.name}'")
        return domainpredicate

    def add_precondition_to_action(self, action, formula, predicate_lookup, param_lookup):
        unique_name = self.make_unique_name(action.name)
        precondition_indiv = self.ActionPrecondition(self.clean_iri(f"{unique_name}_precondition"))
        action.hasPrecondition.append(precondition_indiv)
        root_formula_node = self.formula_to_ontology(formula, predicate_lookup, param_lookup)
        precondition_indiv.hasRootNode.append(root_formula_node)
        precondition_indiv.label = [locstr(str(formula), lang="en")]
        logging.info(f"Added ActionPrecondition to {action.name} '{formula}'")
        return precondition_indiv

    def add_effect_to_action(self, action_onto, effects, predicate_lookup, param_lookup):
        effect_inds = []
        for effect in effects:
            effect_node = self.formula_to_ontology(effect, predicate_lookup, param_lookup)
            action_effect = self.ActionEffect()
            action_effect.hasRootNode.append(effect_node)
            action_onto.hasEffect.append(action_effect)
            action_effect.label = [locstr(str(effect), lang="en")]
            logging.info(f"Added Effect to {action_onto} '{effect}'")
            effect_inds.append(action_effect)
        return effect_inds

    def add_parameter_to_action(self, action_onto, action):
        unique_name = self.make_unique_name(f"{action.name}_params")
        parameter = self.ActionParameter(unique_name)
        parameter.hasSyntax = [locstr(str(action), lang="en")]
        parameter.label = [locstr(str(f"{action.name}_params"), lang="en")]
        action_onto.hasParameter.append(parameter)
        logging.info(f"Added ActionParameter '{unique_name}' to {action.name}")
        return parameter

    def add_parameter_type_to_action_parameter(self, param_complex, param):
        """
        param_complex: The 'big' ActionParameter individual (for the action)
        param: The unified-planning parameter object (has .name and .type)
        """
        param_type_name = param.type.name
        param_type_indiv = self.types[param_type_name]
        param_complex.hasParameterType.append(param_type_indiv)
        logging.info(f"Linked parameter type {param_type_name} to ActionParameter '{param_complex.name}'")
        return param_type_indiv

    def add_requirement_to_domain(self, domain, requirement_name):
        requirement = self.DomainRequirement(requirement_name)
        domain.hasRequirement.append(requirement)
        logging.info(f"Added DomainRequirement to {domain.name} '{requirement_name}'")
        return requirement

    def formula_to_ontology(self, formula, predicate_lookup, param_lookup):
        if hasattr(formula, "fluent") and hasattr(formula, "value"):
            # Effect object, not just FNode!
            fluent = formula.fluent() if callable(formula.fluent) else formula.fluent
            inner_fluent = fluent.fluent() if hasattr(fluent, "fluent") and callable(fluent.fluent) else getattr(fluent, "fluent", fluent)
            pred_name = inner_fluent.name
            fluent_args = [param_lookup[str(arg)] for arg in fluent.args]
            value = formula.value
            assignment_effect = self.AssignmentEffect()
            assignment_effect.assignsFluent.append(predicate_lookup[pred_name])
            assignment_effect.hasArgument = fluent_args
            assignment_effect.assignsValue.append(self.up_value_to_python(value))
            assignment_effect.label = [locstr(f"{pred_name}({', '.join(str(a) for a in fluent.args)}) := {value}", lang="en")]
            return assignment_effect
 
        if formula.is_and():
            and_node = self.And()
            and_node.hasArgument = [
                self.formula_to_ontology(child, predicate_lookup, param_lookup)
                for child in formula.args
            ]
            return and_node
        elif formula.is_or():
            or_node = self.Or()
            or_node.hasArgument = [
                self.formula_to_ontology(child, predicate_lookup, param_lookup)
                for child in formula.args
            ]
            return or_node
        elif formula.is_not():
            not_node = self.Not()
            not_node.hasArgument = [
                self.formula_to_ontology(formula.args[0], predicate_lookup, param_lookup)
            ]
            return not_node
        elif hasattr(formula, "fluent"):
            pred = formula.fluent() if callable(formula.fluent) else formula.fluent
            if hasattr(pred, "name"):
                pred_name = pred.name
                atomic_node = self.AtomicFormula()
                atomic_node.label = [locstr(str(formula), lang="en")]
                atomic_node.refersToPredicate.append(predicate_lookup[pred_name])
                atomic_node.hasArgument = [param_lookup[str(arg)] for arg in formula.args]
                return atomic_node
            else:
                raise NotImplementedError(
                    f"Expected Fluent for atomic formula, got {type(pred)} ({repr(pred)})"
                )
        else:
            raise NotImplementedError(
                f"Unsupported formula node: {formula} (class: {formula.__class__}, content: {repr(formula)})"
            )


    # ==================================
    # PlanOnto --> PDDL Parsing Functions
    # ----------------------------------

    def get_domain_indiv(self, domain_name=None):
        logging.info(f"Looking for PlanningDomain individual with name: {domain_name}")
        for dom in self.onto.PlanningDomain.instances():
            logging.debug(f"Found domain individual: {dom.name}")
            if (domain_name is None) or (dom.name == domain_name):
                return dom
        logging.warning(f"No PlanningDomain with name {domain_name}")
        raise ValueError(f"No PlanningDomain with name {domain_name}")

    def extract_types(self, domain_ind):
        types = {}  # {type_name: up.UserType}
        logging.info("Extracting types from ontology...")
        for type_cls in self.onto.object.subclasses():
            if type_cls.name != "object":
                types[type_cls.name] = UserType(type_cls.name)
                logging.debug(f"Added user type: {type_cls.name}")
        return types

    def extract_objects(self, types):
        objects = {}  # {obj_name: (up.Object, type)}
        logging.info("Extracting objects from ontology...")
        for type_name, up_type in types.items():
            owl_class = self.onto[type_name]
            for indiv in owl_class.instances():
                objects[indiv.name] = (Object(indiv.name, up_type), up_type)
                logging.debug(f"Added object: {indiv.name} (type: {type_name})")
        return objects

    def extract_fluents(self, domain_ind, types):
        fluents = {}
        logging.info("Extracting fluents (predicates) from ontology...")
        for pred in domain_ind.hasPredicate:
            pred_name = pred.name
            param_list = []
            for param in pred.hasDomainPredicateParameter:
                param_type = types[param.hasParameterType[0].name]  # assume first is correct
                param_list.append((param.name, param_type))
                logging.debug(f"Fluent '{pred_name}': found parameter {param.name} of type {param_type}")
            fluent = Fluent(pred_name, BoolType(), **{n: t for n, t in param_list})
            fluents[pred_name] = fluent
            logging.debug(f"Added fluent: {pred_name} ({param_list})")
        return fluents

    def extract_actions(self, domain_ind, types, fluents):
        actions = []
        logging.info("Extracting actions from ontology...")
        for act in domain_ind.hasAction:
            act_name = act.name
            params = []
            for param in act.hasParameter:
                param_type = types[param.hasParameterType[0].name]
                params.append((param.name, param_type))
                logging.debug(f"Action '{act_name}': found parameter {param.name} of type {param_type}")
            up_action = InstantaneousAction(act_name, **{n: t for n, t in params})
            logging.debug(f"Created InstantaneousAction '{act_name}' with params: {params}")
            # Parse preconditions/effects recursively from ontology structure
            if hasattr(act, "hasActionPrecondition"):
                for pre in act.hasActionPrecondition:
                    logging.debug(f"Parsing precondition for action '{act_name}': {pre}")
                    up_action.add_precondition(self.parse_ontology_formula(pre.hasRoot, fluents, params))
            if hasattr(act, "hasEffect"):
                for eff in act.hasEffect:
                    logging.debug(f"Parsing effect for action '{act_name}': {eff}")
                    up_action.add_effect(self.parse_ontology_effect(eff.hasRootNode, fluents, params))
            actions.append(up_action)
            logging.info(f"Added action: {act_name}")
        return actions

    def ontology_to_up_problem(self, domain_name=None):
        logging.info(f"Starting ontology-to-UP problem reconstruction for domain: {domain_name}")
        domain_ind = self.get_domain_indiv(domain_name)
        types = self.extract_types(domain_ind)
        objects = self.extract_objects(types)
        fluents = self.extract_fluents(domain_ind, types)
        actions = self.extract_actions(domain_ind, types, fluents)
        problem = Problem(domain_ind.name)
        for t in types.values():
            problem.add_user_type(t)
            logging.debug(f"Added user type to problem: {t}")
        for obj, typ in objects.values():
            problem.add_object(obj)
            logging.debug(f"Added object to problem: {obj}")
        for fluent in fluents.values():
            problem.add_fluent(fluent)
            logging.debug(f"Added fluent to problem: {fluent}")
        for action in actions:
            problem.add_action(action)
            logging.debug(f"Added action to problem: {action}")
        logging.info(f"Problem '{problem.name}' reconstruction complete.")
        # TODO: initial state and goals
        return problem

    # --- Stubs for recursive parsing ---
    def parse_ontology_formula(self, root_node, fluents, params):
        logging.debug(f"Parsing ontology formula node: {root_node}")
        # Implement formula tree reconstruction here
        pass

    def parse_ontology_effect(self, root_node, fluents, params):
        logging.debug(f"Parsing ontology effect node: {root_node}")
        # Implement effect tree reconstruction here
        pass


    # ====================
    # Utility Functions
    # --------------------

    def make_unique_name(self, base_name):
        rand_id = uuid.uuid4().hex[:8]  # short random hex
        return f"{self.current_domain_name}_{base_name}_{rand_id}"

    def up_value_to_python(self, value):
        # Unified Planning FNode for constant? Use .constant_value() if available.
        # Otherwise, just pass through.
        if hasattr(value, "is_bool_constant") and value.is_bool_constant():
            return bool(value.constant_value())
        elif hasattr(value, "constant_value"):
            # For numbers
            return value.constant_value()
        elif isinstance(value, (bool, int, float, str)):
            return value
        else:
            raise ValueError(f"Cannot convert UP value '{value}' ({type(value)}) to a Python literal for OWL.")
    
    def save(self, filename="planonto/models/ontoviplan/test-output.owl"):
        self.onto.save(file=filename, format="rdfxml")
        logging.info(f"Ontology saved to {filename}")

    def read_ontology(self, ontology_dir='test/output_onto/', ontology_file="hanoi_planonto.owl"):
        # Read the ontology
        self.ontology_dir = ontology_dir
        onto_path.append(self.ontology_dir)
        self.onto = get_ontology(ontology_file).load()
        # self._print_onto_classes_and_instances()

    def reconstruct_iri(self, cleaned: str) -> str:
        """
        Reverses the process from clean_iri.
        """
        return urllib.parse.unquote(str(cleaned))

    def clean_iri(self, name: str) -> str:
        """
        Converts the full IRI into a readable but reconstructable format.
        Uses percent-encoding for the fragment.
        """
        return urllib.parse.quote(str(name), safe='')

    def mark_instances_with_domain(self, domain_instance, source_file ='unknown'):
        """
        Annotate all relevant instances with the belongsToDomain property.
        """
        # List the classes you want to annotate
        relevant_classes = [
            self.DomainAction,
            self.DomainPredicate,
            self.DomainPredicateParameter,
            self.ActionParameter,
            self.ActionEffect,
            self.ActionPrecondition,
            self.AtomicFormula,
            self.object
        ]
        for cls in relevant_classes:
            for inst in cls.instances():
                if domain_instance not in inst.belongsToDomain:
                    inst.belongsToDomain.append(domain_instance)
                    inst.comment = [f'Created from file: {source_file}']

    # ====================
    # OntoViPlan Functions
    # --------------------

    def add_task_annotation(self, domain, annotation_content):
        pass

    def add_pddl2onto(self, domain_filename=None, problem_filename=None, save_path=None):
        reader = PDDLReader()
        problem = reader.parse_problem(domain_filename, problem_filename)
        self._print_up_types(problem)
        domain = self.create_domain(problem.name)
        self.current_domain_name = problem.name
        self.add_requirement_to_domain(domain, 'strips')
        self.add_types(problem)
        pred_lookup = {}
        for predicate in problem.fluents:
            print(predicate)
            predicate_indiv = self.add_predicate_to_domain(domain, predicate)
            pred_lookup[predicate.name] = predicate_indiv
        # Prepare param lookups for this domain
        self.param_lookups[problem.name] = {}
        self.predicate_lookups[problem.name] = pred_lookup

        for action in problem.actions:
            unique_name = self.make_unique_name(action.name)
            action_onto = self.add_action_to_domain(domain, unique_name)
            action_onto.label = [locstr(str(action.name), lang="en")]
            param_lookup = {}
            # Create a single ActionParameter individual for all parameters of this action
            #param_complex = self.add_parameter_to_action(action_onto, f"{action.name}_params")
            param_complex = self.add_parameter_to_action(action_onto, action)
            for param in action.parameters:
                # Add a link from param_complex to this parameter's type
                param_type_indiv = self.add_parameter_type_to_action_parameter(param_complex, param)
                param_lookup[param.name] = param_type_indiv

            # Store param_lookup for this domain+action
            self.param_lookups[problem.name][action.name] = param_lookup

            self.add_effect_to_action(action_onto, action.effects, pred_lookup, param_lookup)
            for precondition in action.preconditions:

                self.add_precondition_to_action(
                    action_onto,
                    precondition,
                    predicate_lookup=self.predicate_lookups[problem.name],
                    param_lookup=self.param_lookups[problem.name][action.name]
                )
        self.mark_instances_with_domain(domain, source_file=domain_filename)
        if save_path:
            self.save(save_path)
        return problem

    def convert_domain2uniproblem(self):
        pass

    def query_all_domain_descriptions(self):
        pass

    def query_domain_details(self):
        pass

    def select_matching_domain(self, instruction, model_name='all-MiniLM-L6-v2'):
        """
        Given a natural language instruction, returns the PlanningDomain individual whose
        hasGeneralDomainDescription is closest in semantic embedding space.
        Returns the domain individual (not just the name).
        """
        # 1. Load the embedding model
        model = SentenceTransformer(model_name)
        
        # 2. Gather all PlanningDomain instances and their descriptions
        domains = list(self.PlanningDomain.instances())
        if not domains:
            logging.warning("No PlanningDomain instances found in ontology.")
            return None

        descriptions = []
        for domain in domains:
            # Attempt to extract the description (first entry) as a string.
            desc_list = getattr(domain, "hasGeneralDomainDescription", [])
            if desc_list:
                # Each entry might be an Owlready2 locstr or string; get the value as str
                desc_str = str(desc_list[0])
            else:
                # If missing, fallback to the domain name, or skip, or put placeholder.
                desc_str = domain.name
            descriptions.append(desc_str)
        
        # 3. Encode instruction and all descriptions
        instruction_emb = model.encode(instruction, convert_to_tensor=True)
        description_embs = model.encode(descriptions, convert_to_tensor=True)
        
        # 4. Compute similarities
        cosine_scores = util.cos_sim(instruction_emb, description_embs)
        most_similar_idx = cosine_scores.argmax().item()
        best_domain = domains[most_similar_idx]
        best_score = cosine_scores[0, most_similar_idx].item()

        logging.info(f"Best matching domain: {best_domain.name} (score={best_score:.3f})")
        return best_domain 

    def get_domain_indiv(self, domain_name=None):
        for dom in self.onto.PlanningDomain.instances():
            if (domain_name is None) or (dom.name == domain_name):
                return dom
        raise ValueError(f"No PlanningDomain with name {domain_name}")

    def convert_onto2pddl(self, problem: Problem, domain_name="test_domain.pddl", problem_name="test_problem.pddl"):
        
        writer = PDDLWriter(problem)
        domain_path = self.save_path + problem_name
        writer.write_domain(domain_path)  
        logging.info(f"Wrote domain to: {domain_path}")
        # problem_path = self.save_path + domain_name
        # writer.write_problem(problem_path)
        # logging.info(f"Wrote problem to: {problem_path}")

    def domain_generator(self, instruction):
        logging.info(f"Generating domain for instruction: {instruction}")
        selected_domain = self.select_matching_domain(instruction)
        problem = self.convert_domain2uniproblem(selected_domain)
        self.convert_onto2pddl(problem)
        prompts = []

        # query = """
        # PREFIX : <http://example.org/ontology.owl#>
        # SELECT ?scenario ?task ?desc ?obj ?super
        # WHERE {
        #     ?scenario a :Scenario .
        #     ?scenario :involvesObject ?obj .
        #     ?obj a ?super .
        #     ?task a :Task .
        #     OPTIONAL { ?task :hasDescription ?desc }
        # }
        # """
        query = """
        PREFIX : <http://example.org/ontology.owl#>
        SELECT ?scenario ?task ?desc ?obj ?super
        WHERE {
            ?scenario a :Scenario .
        }
        """
        self.logger.debug(f"Executing SPARQL query: {query}")
        results = self.graph.query(query)
        self.logger.info(f"Query returned {len(results)} results.")
        scenario_map = {}

        for row in results:
            scenario_uri = row['scenario']
            task_uri = row['task']
            desc = str(row['desc']) if row['desc'] else "No description provided"
            obj_uri = row['obj']
            super_uri = row['super']

            scenario_name = scenario_uri.split("#")[-1]
            task_name = task_uri.split("#")[-1]
            obj_name = obj_uri.split("#")[-1]
            super_name = super_uri.split("#")[-1]

            if scenario_name not in scenario_map:
                scenario_map[scenario_name] = {
                    'task': task_name,
                    'description': desc,
                    'objects': []
                }

            scenario_map[scenario_name]['objects'].append((obj_name, super_name))

        for scenario, info in scenario_map.items():
            object_lines = [f"        {obj} - {superc}" for obj, superc in info['objects']]

            prompt = f"""
            I want you to solve a planning problem for {info['task']}.
            You need to:
            {info['description']}

            An example of a planning problem definition is:
            (define (problem {scenario})
                (:domain {info['task']})
                (:objects
            {chr(10).join(object_lines)}
                )
            )
            """
            prompts.append(prompt.strip())

        return prompts

    def add_domain_description_interactive(self):
        # 1. Find all PlanningDomain individuals
        domains = list(self.PlanningDomain.instances())
        if not domains:
            print("No domains found in ontology.")
            return

        # 2. Print available domains for user selection
        print("Available domains:")
        for idx, dom in enumerate(domains):
            # Print name and optionally existing description
            desc = dom.__dict__.get('hasGeneralDomainDescription', [])
            desc_str = f" [Current description: '{desc[0]}' ]" if desc else ""
            print(f"{idx}: {dom.name}{desc_str}")

        # 3. Ask user to select one by index
        while True:
            try:
                sel = int(input("Enter the number of the domain to describe: "))
                if 0 <= sel < len(domains):
                    break
                else:
                    print("Invalid selection, try again.")
            except ValueError:
                print("Please enter a valid integer.")

        domain = domains[sel]

        # 4. Ask user for a description
        description = input(f"Enter a description for domain '{domain.name}': ").strip()
        if not description:
            print("No description entered. Aborting.")
            return

        # 5. Attach the description using the annotation property
        # as a LocallyString (locstr) if you want to support language tags.
        domain.hasGeneralDomainDescription = [locstr(description, lang="en")]

        print(f"Description added to domain '{domain.name}': \"{description}\"")
        logging.info(f"Added hasGeneralDomainDescription to {domain.name}: {description}")


    def test_domain_matching(self, input_path):
        """
        For a folder, processes all .txt files. For a file, just that file.
        For each text file, prints:
        - the file name
        - the file content (instruction)
        - the best matching domain and its description
        """
        paths = []
        if os.path.isdir(input_path):
            # All .txt files in the folder (not recursive)
            paths = [os.path.join(input_path, f) for f in os.listdir(input_path)
                    if f.endswith('.txt') and os.path.isfile(os.path.join(input_path, f))]
            if not paths:
                print("No .txt files found in the folder.")
                return
        elif os.path.isfile(input_path):
            paths = [input_path]
        else:
            print(f"Path does not exist: {input_path}")
            return

        for file_path in paths:
            with open(file_path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()

            print("="*40)
            print(f"File: {file_path}")
            print("Input instruction:")
            print(instruction)
            best_domain = self.select_matching_domain(instruction)
            if best_domain:
                # Get domain description (if any)
                desc = getattr(best_domain, "hasGeneralDomainDescription", [""])[0]
                print(f"\nBest matching domain: {best_domain.name}")
                print(f"Domain description: {desc}")
            else:
                print("No matching domain found.")
            print("="*40 + "\n")

#=======================
# Testing functions
#-----------------------

def test_prompts():
    constructor = PromptConstructor("ontology.owl")
    generated_prompts = constructor.construct_prompt()
    for prompt in generated_prompts:
        logging.info(prompt)
        logging.info("\n---\n")

def query_action_parameters(owl_path, domain_name):
    g = Graph()
    g.parse(owl_path)
    ns = Namespace("http://example.org/ontology.owl#")
    
    query = f"""
    PREFIX : <http://example.org/ontology.owl#>
    SELECT ?action ?param
    WHERE {{
        ?domain a :PlanningDomain ;
                :hasAction ?action .
        ?action :hasParameter ?param .
        FILTER(strends(str(?domain), "#{domain_name}"))
    }}
    """
    for row in g.query(query):
        action_uri = row['action']
        param_uri = row['param']
        print(f"Action: {action_uri}, Parameter: {param_uri}")

if __name__ == "__main__":
    converter = PlanOnto4UniPlan()
    input_name = 'blocksworld_domain.pddl'
    output_name = 'blocksworld_planonto.owl'

    input_name = 'hanoi_domain.pddl'
    output_name = 'hanoi_planonto.owl'

    # input_name = 'object-arrangement_domain.pddl'
    # output_name = 'object-arrangement_planonto.owl'
    multiple_domains =['blocksworld_domain.pddl', 'hanoi_domain.pddl', 'object-arrangement_domain.pddl']
    output_name = 'combined.owl'
    for input_name in multiple_domains:
        input_path = Path('test/domains/input_domains/') / input_name
        output_path = Path('test/output_onto/') / output_name
        problem = converter.add_pddl2onto(domain_filename=str(input_path), save_path=str(output_path))
        converter.add_domain_description_interactive()
        converter.save("test/output_onto/ontology_with_descriptions.owl")
    # converter.read_ontology()
    # query_action_parameters("test/domains/output_domains/hanoi_planonto.owl", "hanoi")

    converter.test_domain_matching("test/instructions")