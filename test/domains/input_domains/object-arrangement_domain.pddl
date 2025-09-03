(define (domain object_arrangement)

(:requirements :strips :typing :fluents :negative-preconditions)

(:types
    location - object
    cutlery - object
    container - object
    plate - object
    food - object 
    robot  - object
    glass - object
    mug - object
    screwdriver - object
    bowl - object
    fruit - food 
    tool - object
    cup - object
    basket - object
    )

(:predicates
    (free ?obj - object)
    (at ?obj - object ?loc - location) 
    (on ?obj1 - food ?obj2 - object) 
    (in ?obj1 - food ?obj2 - container)

)


(:action move
    :parameters (?robot - robot ?loc1 ?loc2 - location)
    :precondition(
    and
        (at ?robot ?loc1)
    )
    :effect(
    and
        (not (at ?robot ?loc1)) ; location
        (at ?robot ?loc1)
    )
)

(:action pick
    :parameters (?robot - robot ?obj - object ?loc - location)
    :precondition(
    and
        (free ?robot)
        (free ?obj)
        (at ?obj ?loc)
        (at ?robot ?loc)    
    )
    :effect(
    and
        (not (free ?robot)) ; robot
        (not (at ?obj ?loc)) ; location
    )
)

(:action place_on_table
    :parameters (?robot - robot ?obj - object ?loc - location)
    :precondition(
    and
        (not (free ?robot))
        (at ?robot ?loc)
    )
    :effect(
    and
        (free ?robot)
        (at ?obj ?loc) 
    )
)



(:action remove_from_table
    :parameters (?robot - robot ?obj - object ?loc - location)
    :precondition(
    and
        (not (free ?robot))
        (at ?robot ?loc)
    )
    :effect(
    and
        (free ?robot)
        (at ?obj ?loc) 
)
)

(:action put_on_plate
    :parameters (?robot - robot ?obj1 - food ?obj2 - plate ?loc - location)
    :precondition(
    and
        (at ?obj2 ?loc)
        (at ?robot ?loc)
    )
    :effect(
    and
        (free ?robot) 
        (on ?obj1 ?obj2)
    )
)

(:action put_in_container
    :parameters (?robot - robot ?obj1 - food ?obj2 - container ?loc - location)
    :precondition(
    and
        (at ?obj2 ?loc)
        (at ?robot ?loc)
    )
    :effect(
    and
        (free ?robot) 
        (on ?obj1 ?obj2)
    )
)



)
