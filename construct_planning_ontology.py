from owlready2 import *

class OntologyBuilder:
    def __init__(self, iri: str):
        self.ontology = get_ontology(iri)

    def build(self):
        with self.ontology:
            self._create_properties()
            self._create_classes()
            self._annotate_classes()
            self._define_property_domains_and_ranges()
            self._add_axioms()
            self._create_individuals()

    def save(self, path: str, format: str = "rdfxml"):
        self.ontology.save(file=path, format=format)

    def _create_properties(self):
        class hasDescription(DataProperty):
            pass

        class hasGeneralDescription(hasDescription):
            pass

        class hasSpecificDescription(hasDescription):
            pass

        class involvesObject(ObjectProperty):
            pass

        class involvesLocation(ObjectProperty):
            pass

        class hasTargetLocation(ObjectProperty):
            pass

        class hasObject(ObjectProperty):
            pass

        class inScenario(ObjectProperty):
            pass

        class hasScenario(ObjectProperty):
            pass

        self.hasDescription = hasDescription
        self.hasSpecificDescription = hasSpecificDescription
        self.hasGeneralDescription = hasGeneralDescription
        self.involvesObject = involvesObject
        self.involvesLocation = involvesLocation
        self.hasTargetLocation = hasTargetLocation
        self.hasObject = hasObject
        self.inScenario = inScenario
        self.hasScenario = hasScenario

    def _create_classes(self):
        class Object(Thing):
            pass

        class Food(Object):
            pass

        class Fruit(Food):
            pass

        class Pear(Fruit):
            pass

        class Container(Object):
            pass

        class Basket(Container):
            pass

        class Plate(Container):
            pass

        class Robot(Object):
            pass

        class Task(Thing):
            pass

        class ObjectArrangement(Task):
            pass

        class Scenario(Thing):
            pass

        class BreakfastScenario(Scenario):
            pass

        class Location(Thing):
            pass

        class TargetLocation(Location):
            pass

        class ObjectInScenario(Thing):
            pass

        self.classes = {
            'Object': Object,
            'Food': Food,
            'Pear': Pear,
            'Fruit': Fruit,
            'Container': Container,
            'Basket': Basket,
            'Plate': Plate,
            'Robot': Robot,
            'Task': Task,
            'Object Rearrangement': ObjectArrangement,
            'Scenario': Scenario,
            'BreakfastScenario': BreakfastScenario,
            'Location': Location,
            'TargetLocation': TargetLocation,
            'ObjectInScenario': ObjectInScenario
        }
    def _add_axioms(self):
        self.classes['BreakfastScenario'].is_a.append(self.hasGeneralDescription.value(
            str("Rearrange the table for breakfast. Remove the objects that are not listed to be in the table.")
        ))

        self.classes['Object Rearrangement'].is_a.append(self.hasGeneralDescription.value(
            str("The task is to rearrange objects, so that each object end up at their target locations.")
        ))

    def _annotate_classes(self):
        annotations = {
            'Object': ("Object", "General object class."),
            'Food': ("Food", "Edible items."),
            'Fruit': ("Fruit", "Category of food that includes fruits."),
            'Pear': ("Pear", "A type of fruit that is edible."),
            'Container': ("Container", "Objects that can hold other items."),
            'Basket': ("Basket", "A container used to carry items."),
            'Plate': ("Plate", "A container used to hold food."),
            'Robot': ("Robot", "A robotic entity."),
            'Task': ("Task", "A defined piece of work to be done."),
            'Object Rearrangement': ("Object Arrangement", "A task to arrange objects."),
            'Scenario': ("Scenario", "A situation involving tasks, objects, and locations."),
            'BreakfastScenario': ("Breakfast Scenario", "Scenario representing a breakfast setting."),
            'Location': ("Location", "A place where things are situated."),
            'TargetLocation': ("TargetLocation", "A specific location targeted in tasks."),
            'ObjectInScenario': ("ObjectInScenario", "Contextual relationship of Object and TargetLocation within a Scenario.")
        }
        for class_name, (label, comment) in annotations.items():
            cls = self.classes[class_name]
            cls.label = label
            cls.comment = comment

    def _define_property_domains_and_ranges(self):
        self.hasDescription.domain = [self.classes['Task'], self.classes['Scenario']]
        self.hasDescription.range = [str]

        self.hasSpecificDescription.domain = [self.classes['Task'], self.classes['Scenario']]
        self.hasSpecificDescription.range = [str]

        self.hasGeneralDescription.domain = [self.classes['Task'], self.classes['Scenario']]
        self.hasGeneralDescription.range = [str]

        self.hasScenario.domain = [self.classes['Task']]
        self.hasScenario.range = [self.classes['Scenario']]

        self.involvesObject.domain = [self.classes['Scenario']]
        self.involvesObject.range = [self.classes['Object']]

        self.involvesLocation.domain = [self.classes['Scenario']]
        self.involvesLocation.range = [self.classes['Location']]

        self.hasObject.domain = [self.classes['ObjectInScenario']]
        self.hasObject.range = [self.classes['Object']]

        self.inScenario.domain = [self.classes['ObjectInScenario']]
        self.inScenario.range = [self.classes['Scenario']]

        self.hasTargetLocation.domain = [self.classes['ObjectInScenario']]
        self.hasTargetLocation.range = [self.classes['TargetLocation']]

    def _create_individuals(self):
        TestTask = self.classes['Object Rearrangement']("BreakfastObjectArrangementTask")
        PaoloBreakfastScenario = self.classes['BreakfastScenario']("PaolosBreakfast")
        TestTask.hasSpecificDescription.append("Arrange objects for breakfast.")
        PaoloBreakfastScenario.hasSpecificDescription.append("Paolo's breakfast scenario.")

        TestPear = self.classes['Pear']("Pear1")
        TestPlate = self.classes['Plate']("Plate1")

        PaoloBreakfastScenario.involvesObject = [TestPear, TestPlate]
        PaoloBreakfastScenario.involvesLocation = [TestPlate]

        obj_in_scenario = self.classes['ObjectInScenario']("PearInBreakfast")
        obj_in_scenario.hasObject = [TestPear]
        obj_in_scenario.inScenario = [PaoloBreakfastScenario]
        obj_in_scenario.hasTargetLocation = [TestPlate]


if __name__ == "__main__":
    builder = OntologyBuilder("http://example.org/ontology.owl")
    builder.build()
    builder.save("ontology.owl")
