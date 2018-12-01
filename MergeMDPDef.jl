"""
AA228 Final Project Code:
"Modeling a Highway Merge as a POMDP Using Dynamic GPS Error"
- This file contains all functions required for solving the POMDP as a QMDP
"""
module MergeMDPDef

push!(LOAD_PATH, pwd())

export MergeMDP

using GridInterpolations
using POMDPs
using POMDPModelTools

using MergeMDPConfig

struct MergeMDP <: MDP{Int64, Symbol} # Just make the state an index and then use grid to convert
    grid::RectangleGrid
    nA::Int64
end

function MergeMDP()
    state_grid = RectangleGrid(taus, v1s, v2s, ds)
    return MergeMDP(state_grid, nA)
end

# type State
#    tau::Float64 # Time gap between cars
#    v1::Float64 # Velocity of highway car
#    v2::Float64 # Velocity of merging car
#    d::Float64 # Distance to merge zone (for merging car)
# end

# type Action
#    acceleration::Symbol
#
function POMDPs.states(mdp::MergeMDP)
    return collect(0:length(mdp.grid))
end

function POMDPs.n_states(mdp::MergeMDP)
    return length(mdp.grid)+1
end

function POMDPs.stateindex(mdp::MergeMDP, state::Int64)
    return state+1
end

function POMDPs.actions(mdp::MergeMDP)
    # d2 - hard slow down, d1 - soft slow down
    # m - maintain
    # a1 - soft speed up, a2 - hard speed up
    return [:d2 :d1 :m :a1 :a2]
end

function POMDPs.n_actions(mdp::MergeMDP)
    return mdp.nA
end

function POMDPs.actionindex(mdp::MergeMDP, a::Symbol)
    return a2Ind(a)
end

function POMDPs.discount(mdp::MergeMDP)
    return discount_factor
end

function POMDPs.transition(mdp::MergeMDP, s::Int64, a::Symbol)
    if s == 0 # in terminal state
        return SparseCat(0, 0)
    else
        # First, convert to a grid state
        s_grid = ind2x(mdp.grid, s)
        # Convert action to acceleration
        accel = a2accel(mdp,a)

        if s_grid[4] == 0 # Transition to terminal state (0)
            posStates = 0
            posProbs = 1
        else
            posStates, posProbs = nextStates(mdp, s_grid, accel)
        end
        # Create sparse categorical distribution from this
        returnDist = SparseCat(posStates, posProbs)
        return returnDist
    end
end

function POMDPs.reward(mdp::MergeMDP, s::Int64, a::Symbol)
    # Initialize reward to 0
    r = 0

    # Penalize actions
    actionPen = getActionPenalty(mdp, a)
    r += actionPen

    # Penalize or reward end state
    s_grid = ind2x(mdp.grid, s)
    tau = s_grid[1]
    v1 = s_grid[2]
    v2 = s_grid[3]
    d = s_grid[4]
    if d < dTol # == 0
        # Check for collision
        if abs(tau) < 0.5
            if tau == 0 # Take care of dividing by zero
                r += penCollision*20
            else
                r += penCollision/abs(tau)
            end
        end
        # Reward good final state with time gap near 0.6 and relative velocity near 0
        if (timeGap - goodPosTol) < abs(tau) < (timeGap + goodPosTol)
            if abs(v2 - v1) <= vrelTol
                r += rewardGoodPos
            end
        end
    end
    return r
end

function POMDPs.isterminal(mdp::MergeMDP, s::Int64)
    if s == 0
        return true
    else
        return false
    end
end

function nextStates(mdp::MergeMDP, currState::Array, accel::Float64)

    # Possible accelerations for the highway car
    a1s = [-accelSigma 0 accelSigma]
    # Possible accelerations for the merging car
    a2s = [accel-2*accelSigma accel-accelSigma accel accel+accelSigma accel+2*accelSigma]

    posStates = []
    posProbs = []

    # Loop through accelerations and get next states
    for i = 1:length(a1s)
        for j = 1:length(a2s)
            a1 = a1s[i]
            a2 = a2s[j]
            # Get next state
            nextState = getNextState(currState,a1,a2)
            states, probs = interpolants(mdp.grid, nextState)

            # Add these to the possible next states and weight their probabilities
            posStates = [posStates; states]
            posProbs = [posProbs; probs*weightsMat[i,j]]
        end
    end

    return posStates, posProbs
end

function getNextState(currState::Array, a1::Float64, a2::Float64)
    tau = currState[1]
    v1 = currState[2]
    v2 = currState[3]
    d = currState[4]

    # Determine new variables
    nextv1 = v1 + a1*dt
    nextv2 = v2 + a2*dt
    nextd = d - v2*dt - 0.5*a2*dt^2
    # Next tau is a bit more involved of a calculation
    # Determine current dx
    if tau < 0 # Merging car is in front
        dx_t = tau*v1
    else
        dx_t = tau*v2
    end
    # Find dx at next time step
    dx_tp1 = dx_t + (v1*dt + 0.5*a1*dt^2 - v2*dt - 0.5*a2*dt^2)
    # Finally, get next tau
    if dx_tp1 < 0 # Merging car is in front
        nextTau = dx_tp1/nextv1
    else
        nextTau = dx_tp1/nextv2
    end

    return [nextTau, nextv1, nextv2, nextd]
end

function a2Ind(a::Symbol)
    # Just manually assign them
    if a == :d2
        return 1
    elseif a == :d1
        return 2
    elseif a == :m
        return 3
    elseif a == :a1
        return 4
    elseif a == :a2
        return 5
    end
    error("invalid MergeMDP action: $a")
end

function a2accel(mdp::MergeMDP, a::Symbol)
    # Get action index
    aInd = a2Ind(a)
    # Convert to acceleration
    accel = accelerations[aInd]
    return accel
end

function getActionPenalty(mdp::MergeMDP, a::Symbol)
    aInd = a2Ind(a)
    pen = penActions[aInd]
    return pen
end

end # Module MergeMDPDef
