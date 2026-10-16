# # Tutorial 4. Tuning hyperparameters

# > **Goals:** Learn how to:
# > 1. Tune (optimize) a single model hyperparameter visually, by plotting learning curves
# > 2. Implement the optimization of one or more hyperparameters by wrapping a model in a tuning strategy


# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

# ### Naive tuning of a single parameter

# The most naive way to tune a single hyperparameter is to use
# `learning_curve`, which we already saw in Tutorial 2. Let's see this in
# the Horse Colic classification problem, a case where the parameter
# to be tuned is *nested* (because the model is a pipeline).

# Here is the Horse Colic data again, with the type coercions we
# already discussed in Tutorial 1:

using MLJ
import Downloads, CSV, DataFrames
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/"*
    "for-MLJ-version-0.16/data/horse.csv"
csv_file = Downloads.download(url)
horse = CSV.read(csv_file, DataFrames.DataFrame)
coerce!(horse, autotype(horse));
coerce!(horse, Count => Continuous);
coerce!(
    horse,
    :surgery               => Multiclass,
    :age                   => Multiclass,
    :mucous_membranes      => Multiclass,
    :capillary_refill_time => Multiclass,
    :outcome               => Multiclass,
    :cp_data               => Multiclass,
);
schema(horse)

y, X = unpack(horse, ==(:outcome));
schema(X)

# Now for a pipeline model:

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model = Standardizer |> ContinuousEncoder |> LogisticClassifier

#-

mach = machine(model, X, y);

# We now specify a hyperparameter range for this pipeline model:

r = range(model, :(logistic_classifier.lambda), lower = 1e-4, upper=0.1, scale=:log10)

# If you're curious, you can see what `lambda` values this range will
# generate for a given resolution:

iterator(r, 5)

#-

using Plots
gr(size=(490,300))
_, _, lambdas, losses = learning_curve(
    mach,
    range=r,
    resampling=CV(nfolds=6),
    resolution=30, # default
    measure=log_loss,
)
plt=plot(lambdas, losses, xscale=:log10)
xlabel!(plt, "lambda")
ylabel!(plt, "log loss using 6-fold CV")
savefig("learning_curve2.png")
plt #!md

# ![](learning_curve2.png) #md

best_lambda = lambdas[argmin(losses)]


# ### Self tuning models

# A more sophisticated way to view hyperparameter tuning (inspired by
# [mlr](https://mlr.mlr-org.com)) is as a model *wrapper*. The wrapped model is a new
# model in its own right and when you fit it, it tunes specified hyperparameters of the
# model being wrapped, before training on all supplied data. Calling `predict` on the
# wrapped model is like calling `predict` on the original model, but with the
# hyperparameters already optimized.

# In other words, we can think of the wrapped model as a "self-tuning" version of the
# original.

# We now create a self-tuning version of the pipeline above, adding a parameter from the
# `ContinuousEncoder` to the parameters we want optimized.

# First, let's choose a tuning strategy (from [these
# options](https://github.com/juliaai/MLJTuning.jl#what-is-provided-here)). MLJ supports
# ordinary `Grid` search (query `?Grid` for details). However, as the utility of `Grid`
# search is limited to a small number of parameters, and as `Grid` searches are
# demonstrated elsewhere (see the [resources below](#resources-for-part-4)) we'll
# demonstrate `RandomSearch` here:

tuning = RandomSearch(rng=123)

# In this strategy each parameter is sampled according to a pre-specified prior
# distribution that is fit to the one-dimensional range object constructed using `range`
# as before. While one has a lot of control over the specification of the priors (run
# `?RandomSearch` for details) we'll let the algorithm generate these priors
# automatically.


# #### Unbounded ranges and sampling

# In MLJ a range does not have to be bounded. In a `RandomSearch` a positive unbounded
# range is sampled using a `Gamma` distribution, by default:

r = range(
    model,
    :(logistic_classifier.lambda),
    lower=0,
    origin=6,
    unit=5,
    scale=:log10,
)

# The `scale` in a range is ignored in a `RandomSearch`, unless it is a
# function. (It *is* relevant in a `Grid` search, not demonstrated
# here.) Note however, the choice of scale *does* effect how later plots
# will look.

# Let's see what sampling using a Gamma distribution is going to mean
# for this range:

import Distributions
sampler_r = sampler(r, Distributions.Gamma)
plt = histogram(rand(sampler_r, 10000), nbins=50)
savefig("gamma_sampler.png");
plt #!md

# ![](gamma_sampler.png) #md

# The second parameter that we'll add to this is *nominal* (finite) and, by
# default, will be sampled uniformly. Since it is nominal, we specify
# `values` instead of `upper` and `lower` bounds:

s  = range(
    model,
    :(continuous_encoder.one_hot_ordered_factors),
    values = [true, false],
)


# #### The tuning wrapper

# Now for the wrapper, which is an instance of `TunedModel`:

tuned_model = TunedModel(
    model,
    ranges=[r, s],
    resampling=CV(nfolds=6),
    measures=log_loss,
    tuning=tuning,
    n=15,
)

# We can apply the `fit!/predict` work-flow to `tuned_model` just as
# for any other model:

tuned_mach = machine(tuned_model, X, y)
fit!(tuned_mach)
predict(tuned_mach, rows=1:3)

# The outcomes of the tuning can be inspected from a detailed
# report. For example, we have:

rep = report(tuned_mach)
rep.best_model

# You can also visualize the random search:

plt = plot(tuned_mach)
savefig("tuning.png");
plt #!md

# ![](tuning.png) #md

# Finally, let's compare cross-validation estimate of the performance of the self-tuning
# model with that of the original model (an example of [*nested
# resampling*](https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html)):

err = evaluate!(mach, resampling=CV(nfolds=3), measure=log_loss)

#-

tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=log_loss)

# ### Tutorial 4 Resources
#
# - From the MLJ manual:
#    - [Learning Curves](https://juliaai.github.io/MLJ.jl/dev/learning_curves/)
#    - [Tuning Models](https://juliaai.github.io/MLJ.jl/dev/tuning_models/)
# - From Data Science Tutorials:
#     - [Tuning a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/)
#     - [Crabs with XGBoost](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/crabs-xgb/) `Grid` tuning in stages for a tree-boosting model with many parameters
#     - [Boston with LightGBM](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/) -  `Grid` tuning for another popular tree-booster
#     - [Boston with Flux](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-flux/) - optimizing batch size in a simple neural network regressor
# - [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)


# ### Tutorial 4 Exercises

# #### Exercise 8

# This exercise continues our analysis of the King County House price
# prediction problem (Exercise 3, Tutorial 1, and Tutorial 3):

import Downloads, CSV
import DataFrames
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
    "data/house.csv"
csv_file = Downloads.download(url)
house = CSV.read(csv_file, DataFrames.DataFrame)
coerce!(house, autotype(house))
coerce!(house, Count => Continuous, :zipcode => Multiclass)
y, X = unpack(house, ==(:price), rng=123);
schema(X)

# Your task will be to tune the following pipeline regression model,
# which includes a gradient tree boosting component:

EvoTreeRegressor = @load EvoTreeRegressor
tree_booster = EvoTreeRegressor(nrounds = 70)
model = ContinuousEncoder |> tree_booster

# (a) Construct a bounded range `r1` for the `evo_tree_booster`
# parameter `max_depth`, varying between 4 and 14.

# (b) For the `colsample` parameter of the `EvoTreeRegressor`, define the range

r2 = range(model, :(evo_tree_regressor.nbins), values = [32, 64, 128])

# (a)

# Optimize `model` over these the parameter ranges `r1` and `r2` using a random search
# with uniform priors (the default). Use `Holdout()` resampling, and implement your search
# by first constructing a "self-tuning" wrap of `model`, as described above. Make `mae`
# (mean absolute error) the loss function that you optimize, and search over a total of 40
# combinations of hyperparameters.  If you have time, plot the results of your
# search. Feel free to use all available data.

# (c) Evaluate the best model found in the search using 3-fold cross-validation and
# compare with that of the self-tuning model (which is different!). Setting data hygiene
# concerns aside, use all available data.
