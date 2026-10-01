# # Lesson 1. Basics

# Notebook supporting the video series "Using MLJ".

# [Slides](slides.pdf) from the video.

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

# ### Video Timings

# - 00:00 Introduction
# - 01:05 Goals
# - 01:48 Prerequisites and Getting Help
# - 02:21 Supervised Learning Recap
# - 05:31 Models and Machines
# - 09:28 **Live Coding:** Regression
# - 28:55 Scientific Types
# - 32:29 **Live Coding:** Scitypes and Classification
# - 55:45 End

# We start by inspecting the packages, and their exact versions, in the currently active
# package environment:

using Pkg
Pkg.status()


# ## Part I. Regression

using MLJ
using UnicodePlots # for pretty display of labeled probability vectors

# Load data and inspect the schema:
data = load_boston()
schema(data) # 'MedV'= median value of owner-occupied homes in $1000s.

# Split off target variable:
y, X = unpack(data, ==(:MedV))
schema(X)

#-

first(y, 5)

# Split observations (row indices) in ration 60:40
train, test = partition(1:length(y), 0.6)

# Choose a model:
models(matching(X, y))

#-

Regressor = @iload RandomForestRegressor pkg=DecisionTree
model = Regressor()

# Inspect the documentation of this mode by running `@doc Regressor` or `?Regressor` in
# the REPL.

# Train on `train` rows:
mach = machine(model, X, y)
fit!(mach, rows=train);

# Inspect the trained machine:

fitted_params(mach)

#-

report(mach)

# Predict on some "new" data

predict(mach, X)[1:3]

# Predict in the `test` rows:
ypred = predict(mach, rows=test);

# Get the mean absolute error:
mae(ypred, y[test])

# `mae` is actually just an alias:

mae

# List other pertinent metrics:

measures(ypred, y)

# Get performance estimates in one hit:
evaluate!(mach; resampling=[(train, test),], measures=[mae, RSquared()])

# Something fancier:
evaluate(
    model, X, y;
    resampling=CV(nfolds=6),
    measures=[mae, RSquared()],
)

# Dietterich's 5 x 2 test:

e = evaluate(
    model, X, y;
    resampling=CV(nfolds=2, shuffle=true),
    repeats=5,
    measures=[mae, RSquared()],
    acceleration=CPUThreads(),
)

#-

e.uncertainty_radius_95

# # Interlude on scientific types

typeof(3.14)

#-

scitype(3.14)

#-

scitype(3.143f0)

#-

scitype(["cat", "mouse", "dog"])


# ## Part II. Classification

# New data set for classification, the Adult Dataset (census data):
using Downloads, CSV
url = "https://raw.githubusercontent.com/"*
    "saravrajavelu/Adult-Income-Analysis/refs/heads/master/"*
    "adult.csv"
file = Downloads.download(url)
data = CSV.read(file, NamedTuple)
schema(data)

#-

data.income[1:5]

# Fix some of the incorrect scitypes:
data = coerce(
    data,
    :age=>Continuous,
    :occupation=>Multiclass,
    :gender=>Multiclass,
    Symbol("educational-num")=>OrderedFactor,
    :income=>Multiclass,
);

# Split off target, dump some features, and shuffle the observations:
y, X = unpack(
    data,
    ==(:income),
    in([:age, :occupation, :gender, Symbol("educational-num")]),
    rng = 123,
);
scitype(y)

# Split observations (row indices) in ration 60:40
train, test = partition(1:length(y), 0.6)

# What models are available?
models(matching(X, y))

# One-hot encoding:
model_hot = OneHotEncoder()

# List all (non-composite) models currently installed using `localmodels()`.

# Training and applying our one-hot encoder:

mach = machine(model_hot, X) |> fit!
Xhot = transform(mach, X)
schema(Xhot)

# See a selection of compatible supervised models by running
# `models(matching(Xhot,y))`. We'll choose a random forest model:

model = (@load RandomForestClassifier pkg=DecisionTree)()

# Evaluate by hand:
mach = machine(model, Xhot, y)
fit!(mach, rows=train)
yprob = predict(mach, rows=test);
first(yprob, 5)

#-

yprob[3]

#-

ypoint = mode.(yprob)
ypoint = predict_mode(mach, rows=test) # same thing
accuracy(ypoint, y[test])

#-
log_loss(yprob, y[test])

# Evaluate with one command:
evaluate(
    model, Xhot, y;
    resampling=Holdout(fraction_train=0.6),
    measures = [accuracy, log_loss],
)
