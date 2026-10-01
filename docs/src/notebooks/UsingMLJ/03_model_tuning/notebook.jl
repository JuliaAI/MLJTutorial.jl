# # Lesson 3. Model Tuning

# Notebook supporting the video series "Using MLJ".

# [Slides](slides.pdf) from the video.

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

# ### Video Timings

# - 00:00 Introduction
# - 00:06 Goals
# - 01:12 Prerequisites and Getting Help
# - 01:27 Live Coding: Learning Curves
# - 13:27 Tuning as Model Wrapper
# - 18:45 Live Coding: Grid Search
# - 26:17 Live Coding: Random Search
# - 30:48 Nested Resampling
# - 33:11 Final Observations
# - 36:59 End


# ## Part I. Learning Curves

using MLJ, Plots
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

X, price = @load_reduced_ames;
schema(X)

#-

y = price/100_000; # price in multiples of $100,000
first(y, 5)

# Define a model and a "dummy" model for baseline comparison:
model = RandomForestRegressor(
    rng=123,
    n_subfeatures=1,
)
pipe =  ContinuousEncoder() |> model
baseline = ContinuousEncoder() |>  ConstantRegressor();

# Get a performance baseline:
options = (; resampling=CV(nfolds=3), measure = mae, acceleration=CPUThreads())
e0 = evaluate(pipe, X, y; options...)
ebase = evaluate(baseline, X, y; options...)

# Inspect hyper-parameters:
pipe

# Define a 1D hyper-parameter range:
r = range(pipe, :(random_forest_regressor.n_trees), lower=10, upper=500)

# A `learning_curve` has the same arguments as `evaluate` except but with extra `range`
# option:
curve = learning_curve(
    pipe, X, y;
    range=r,
    resampling=Holdout(fraction_train=0.8),
    measure=mae,
)
plt = plot(curve.parameter_values, curve.measurements)
savefig("learning_curve.svg");
plt #!md

# ![](learning_curve.svg) #md

# Let's set `n_trees` to a a value that looks sufficient to get convergence:

pipe.random_forest_regressor.n_trees = 250;

# ## Part II. The `TunedModel` Wrapper

r1 = range(pipe, :(random_forest_regressor.n_subfeatures), lower=1, upper=12)
r2 = range(pipe, :(random_forest_regressor.min_samples_split), lower=2, upper=10)

# Here's how we wrap our pipeline model in grid search optimization of the parameters
# specified by the above ranges:

tuned_pipe = TunedModel(
    pipe,
    range=[r1, r2],
    tuning=Grid(goal=40),
    resampling=CV(nfolds=4),
    measures=mae,
)

#-

mach = machine(tuned_pipe, X, y) |> fit!
plt = plot(mach)
savefig("tuned_model_grid.svg");
plt #!md

# ![](tuned_model_grid.svg) #md

keys(report(mach))

#-

report(mach).best_model

#-

report(mach).best_history_entry.evaluation

# Instead we can use a random search strategy:

tuned_pipe = TunedModel(
    pipe,
    range=[r1, r2],
    tuning=RandomSearch(rng=123),
    resampling=CV(nfolds=4),
    measures=mae,
    n=40,
)

mach = machine(tuned_pipe, X, y) |> fit!
plt = plot(mach)

savefig("tuned_model_random.svg");
plt #!md

# ![](tuned_model_random.svg) #md


# In-sample predictions based on optimized parameters and retraining on *all* data:
predict(mach, X)[1:5]

# Evaluate the "self-tuning" pipeline:

e1 = evaluate(tuned_pipe, X, y; options...)

# Comparing with the baseline computed earlier:

@show ebase e0 e1;

# Or, we can do this (needs MLJBase 1.13.0 or higher):

describe.([e0, e1]) |> pretty

# We can also use `TunedModel` to compare models of different types:

tuned_model = TunedModel(models=[pipe, baseline], resampling=CV(nfolds=4), measure=l1)
mach = machine(tuned_model, X, y) |> fit!
report(mach).best_model

# Here `tuned_model` will behave just like `pipe`, because this is the best performer.
