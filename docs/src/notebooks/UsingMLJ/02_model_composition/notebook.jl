# # Lesson 2. Model Composition

# Notebook supporting the video series "Using MLJ".

# [Slides](slides.pdf) from the video.

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

using MLJ

# Load some model code:
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# Load some data and inspect schema:
data = load_reduced_ames();
schema(data)

# Horizontally split with observation shuffling:
y, X = unpack(data, ==(:target); rng=123);
schema(X)

#-

first(y, 4)

# Define a pipeline model:
pipe = ContinuousEncoder() |> Standardizer() |> RidgeRegressor()

# Access a nested hyperparameter:
pipe.ridge_regressor.fit_intercept

# Change it's value:
pipe.ridge_regressor.fit_intercept = false;

# Evaluate the pipeline:
e1 = evaluate(pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)

# Notice the target very large on the current scale:
@show mean(y) std(y);

# So we wrap the pipeline in target normalization:
norm_pipe = TransformedTargetModel(pipe, transformer=Standardizer())

# Note that target predictions will remain on the original scale. However, as
# internally we are using a normalized target, we get different performance:
e2 = evaluate(norm_pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)

# Changing the regularization parameter `lambda` of ridge regressor, we can arrange that
# the target transformation gives better performance:

pipe_original = deepcopy(pipe)
pipe.ridge_regressor.lambda = 0.45

evaluations = evaluate(
    [
        "default lambda" => pipe_original,
        "new lambda" => pipe,
        "new lambda & normalized target" => norm_pipe],
    X,
    y;
    resampling=CV(nfolds=4, rng=123),
    repeats=2,
    measure=mav,
)

# (To provide multiple models, tagged with strings, to `evaluate` requires MLJBase 1.12.0
# or higher.)

# Here's a pretty view of these results (needs MLJBase 1.13.0 or higher):

describe.(evaluations) |> pretty

# Finding optimal hyper-parameter values is the subject of the next lesson.
