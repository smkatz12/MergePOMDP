"""
AA228 Final Project Code:
"Modeling a Highway Merge as a POMDP Using Dynamic GPS Error"
- This file solves the mdp and generate policies
"""

push!(LOAD_PATH, pwd())

using POMDPs
using POMDPModelTools
using DiscreteValueIteration
using DelimitedFiles
using Printf

using MergeMDPConfig
using MergeMDPDef

mdp = MergeMDP()
println("Starting solver...")
solver = ValueIterationSolver(max_iter, belres, true, true, Vector{Float64}(undef, 0))
policy = solve(solver, mdp)
println("Solved!")

# Write to csv file of q values
q = policy.qmat
open(output_file, "w") do io
    for i = 2:size(q,1)
        for j = 1:size(q,2)
            @printf(io, "%s,", q[i,j])
        end
        @printf(io, "\n")
    end
end
