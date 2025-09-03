(define (problem robot_domain-problem)
 (:domain robot_domain-domain)
 (:objects
   l1 l2 - location
 )
 (:init (robot_at l1) (= (battery_charge) 100))
 (:goal (and (robot_at l2)))
)
