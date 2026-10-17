include("notebook.jl")
@test evaluations[2].measurement[1] > evaluations[3].measurement[1]
true
