
# PlanOnto4UniPlan

A work-in-progress Python library for **bidirectional conversion between PDDL planning domains and ontologies** (OWL),  
aiming for full support for the [unified-planning](https://unified-planning.github.io/) framework and deep integration with [owlready2](https://owlready2.readthedocs.io/en/latest/) and the [Planning Ontology](https://ai4society.github.io/planning-ontology/) .

---

## Features

- **PDDL to OWL**: Load a PDDL domain/problem, parse all types, predicates, actions, and parameters, and store them as a richly-structured ontology.
- **OWL to Unified Planning**: Reconstruct a `Problem` instance (with types, actions, predicates, parameters) directly from an ontology.
- **Flexible, modular, and extensible**—you can add new annotation properties, handle custom types, or link ontology individuals to planning objects with unique, unambiguous URIs.
- **Full round-trip conversion** for knowledge-enabled planning and robotics research.

---

## Ontology & Dependencies

- **This project relies on the [Planning Ontology](https://ai4society.github.io/planning-ontology/) for all core concepts, types, and properties.**
- **Clone recursively:**  
  If you are using any ontology submodules or external dependencies, be sure to clone the repository with `--recursive`:

  ```bash
  git clone --recursive https://github.com/Dorteel/PlanOnto4UniPlan.git
    ```

* **Work in Progress:**
  This package is under active development. Many features are experimental or incomplete, and the code/API may change.

---

## Requirements

* Python 3.8+
* `unified-planning`
* `owlready2`
* `rdflib`
* `sentence-transformers` *(optional, for semantic matching)*

---

## Installation

Clone this repo and install dependencies (ideally in a virtual environment):

```bash
git clone --recursive https://github.com/Dorteel/PlanOnto4UniPlan.git
cd PlanOnto4UniPlan
pip install -r requirements.txt
```

---

## Usage

### Convert PDDL to OWL

```python
from planonto4uniplan import PlanOnto4UniPlan

converter = PlanOnto4UniPlan(
    ontology_dir="planonto/models",
    ontology_file="plan-ontology-rdf.owl"
)

input_domain = "test/domains/input_domains/hanoi_domain.pddl"
output_owl = "test/domains/output_domains/hanoi_planonto.owl"

# Convert and save as ontology
problem = converter.add_pddl2onto(domain_filename=input_domain, save_path=output_owl)
```

* All **types, actions, predicates, parameters, and their structure** will be available as OWL individuals/classes.
* **Debug output** will show every step (types, objects, parameters, effects, preconditions, etc).

---

## Extending

* Add new annotation properties (e.g., `hasExample`, `hasCost`) in Protégé and reference them in code.
* Implement more advanced formula/effect parsing (e.g., for numeric fluents, conditional effects).
* Integrate with other knowledge sources or semantic web tools.

---

## Troubleshooting

* **Performance lag** can occur if you create too many dynamic classes or individuals without existence checks.
* Always check for existing individuals before creating new ones.
* If annotation properties appear as data properties, remove the old DataProperty version in Protégé and reload.

---

## License

[MIT License](LICENSE)

---

## To Do

* [ ] **Support additional PDDL domain types:** Currently tested on STRIPS-style domains; extend to temporal, numeric, and multi-agent domains.
* [ ] **Full round-trip:** Reading from OWL and exporting directly to PDDL is in progress; partial support for reconstructing Unified Planning Problems exists.
* [ ] **Complete formula/effect parser:** Recursively parse and reconstruct all PDDL and ontology formula/effect trees.
* [ ] **Initial state and goals:** Add support for reading initial state and goal conditions from ontology and reconstructing them in the Unified Planning Problem.
* [ ] **Robust handling of domain/parameter uniqueness:** Ensure stable and collision-free URIs for all individuals across domains.
* [ ] **More extensive error handling and user feedback.**

