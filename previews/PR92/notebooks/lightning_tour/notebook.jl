# # Lightning Tour of MLJ

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

# In MLJ a *model* is just a container for hyper-parameters, and that's all. Here we will
# apply several kinds of model composition before binding the resulting "meta-model" to
# data in a *machine* for evaluation, using cross-validation.

# Loading and instantiating a gradient tree-boosting model:

using MLJ

Booster = @load EvoTreeRegressor # loads code defining a model type
booster = Booster(max_depth=2)   # specify hyper-parameter at construction

#-

booster.nrounds=50               # or mutate post facto
booster

# This model is an example of an iterative model. As is stands, the
# number of iterations `nrounds` is fixed.


# ### Composition 1: Wrapping the model to make it "self-iterating"

# Let's create a new model that automatically learns the number of iterations,
# using the `NumberSinceBest(3)` criterion, as applied to an
# out-of-sample `l1` loss:

iterated_booster = IteratedModel(
    model=booster,
    resampling=Holdout(fraction_train=0.8),
    controls=[Step(2), NumberSinceBest(3), NumberLimit(300)],
    measure=l1,
    retrain=true,
)

# ### Composition 2: Preprocess the input features

# Combining the model with categorical feature encoding:

pipe = ContinuousEncoder |> iterated_booster


# ### Composition 3: Wrapping the model to make it "self-tuning"

# First, we define a hyper-parameter range for optimization of a
# (nested) hyper-parameter:

max_depth_range = range(
    pipe,
    :(deterministic_iterated_model.model.max_depth),
    lower = 1,
    upper = 10,
)

# Now we can wrap the pipeline model in an optimization strategy to make it "self-tuning":

self_tuning_pipe = TunedModel(
    model=pipe,
    tuning=RandomSearch(),
    ranges = max_depth_range,
    resampling=CV(nfolds=3, rng=456),
    measure=l1,
    acceleration=CPUThreads(),
    n=50,
)

# ### Binding to data and evaluating performance

# Generating some synthetic data:

X, y = make_regression();

# Binding the "self-tuning" pipeline model to data in a *machine* (which will additionally
# store *learned* parameters):

mach = machine(self_tuning_pipe, X, y)

# Fit and predict:

fit!(mach, rows=1:60)
yhat = predict(mach, rows=61:100)
first(yhat, 3)

# Evaluating the "self-tuning" pipeline model's performance using all data and 5-fold
# cross-validation (implies multiple layers of nested resampling):

evaluate!(
    mach,
    measures=[l1, rsquared],
    resampling=CV(nfolds=5, rng=123),
    acceleration=CPUThreads(),
)

# Compare to a dummy model:

evaluations = evaluate(
    ["booster" => self_tuning_pipe, "dummy" => ConstantRegressor()],
    X,
    y;
    measures=[l1, rsquared],
    resampling=CV(nfolds=5, rng=123),
    acceleration=CPUThreads(),
)

describe.(evaluations) |> pretty
