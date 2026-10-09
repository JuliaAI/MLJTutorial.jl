# # Solutions to Exercises

using MLJ, Downloads, CSV, DataFrames, Plots
nothing #hide

# #### Exercise 1 solution

scitype(42)

#-

questions = ["who", "why", "what", "when"]
scitype(questions)

#-

elscitype(questions)

#-

t = (3.141, 42, "how")
scitype(t)

#-

A = rand(2, 3)

#-

scitype(A)

#-

elscitype(A)

#-

using SparseArrays
Asparse = sparse(A)

#-

scitype(Asparse)

#-

C = coerce(A, Multiclass)

#-

scitype(C)

#-

elscitype(C)

#-

v = [1, 2, missing, 4]
scitype(v)

#-

elscitype(v)

#-

scitype(v[1:2])


# #### Exercise 2 solution

# From the question statement, we have:

quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

#-

quality = coerce(quality, OrderedFactor);
levels!(quality, ["poor", "good", "excellent"]);
elscitype(quality)


# #### Exercise 3 solution

# From the question statement:

url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
    "data/house.csv";
house = CSV.read(Downloads.download(url), DataFrames.DataFrame)
first(house, 4)

# First pass:

coerce!(house, autotype(house));
schema(house)

#-

# All the "sqft" fields refer to "square feet" so are really `Continuous`. We'll regard
# `:yr_built` (the other `Count` variable above) as `Continuous` as well. So:

coerce!(house, Count => Continuous);

# And `:zipcode` should not be ordered:

coerce!(house, :zipcode => Multiclass);
schema(house)

# `:bathrooms` looks like it has a lot of levels, but on further inspection we see why,
# and `OrderedFactor` remains appropriate.

import StatsBase.countmap
d = countmap(house.bathrooms)
for (level, count) in d
    println("$level \t=> $count")
end

# #### Exercise 4 solution

# From the question statement:

import Distributions
poisson = Distributions.Poisson

age = 18 .+ 60*rand(10);
salary = coerce(rand(["small", "big", "huge"], 10), OrderedFactor);
levels!(salary, ["small", "big", "huge"]);
small = salary[1]

X4 = DataFrames.DataFrame(age=age, salary=salary)

n_devices(salary) = salary > small ? rand(poisson(1.3)) : rand(poisson(2.9))
y4 = [n_devices(row.salary) for row in eachrow(X4)]

#-

# 4(a)

models(matching(X4, y4))

# 4(b)

y4 = coerce(y4, Continuous);
models(matching(X4, y4))


# #### Exercise 5 solution

data = (
    a = [1, 2, 3, 4],
    b = rand(4),
    c = rand(4),
    d = coerce(["male", "female", "female", "male"], OrderedFactor),
);

using Tables
y, X, w = unpack(
    data,
    ==(:a),
    name -> elscitype(Tables.getcolumn(data, name)) == Continuous,
)
y

#-

pretty(X)

#-

w

# #### Exercise 6 solution

# From the question statement:

import Downloads
import CSV
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

# 6(a)

y, X = unpack(
    horse,
    ==(:outcome),
    name -> elscitype(Tables.getcolumn(horse, name)) == Continuous,
)
schema(X)

# 6(b)(i)

train, test = partition(eachindex(y), 0.7)
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model = LogisticClassifier()
model.lambda = 100
mach = machine(model, X, y)
fit!(mach, rows=train)
fitted_params(mach)

#-

coefs_given_feature = Dict(fitted_params(mach).coefs)
coefs_given_feature[:pulse]

#6(b)(ii)

yhat = predict(mach, rows=test); # or predict(mach, X[test,:])
err = log_loss(yhat, y[test])

# 6(b)(iii)

# The predicted probabilities of the actual observations in the test
# are given by

p = broadcast(pdf, yhat, y[test]);

# The number of times this probability exceeds 50% is:
n50 = filter(x -> x > 0.5, p) |> length

# Or, as a proportion:

n50/length(test)

# 6(b)(iv)

misclassification_rate(mode.(yhat), y[test])

# 6(c)(i)

RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
model = RandomForestClassifier()
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=log_loss)

#-

r = range(model, :n_trees, lower=10, upper=70, scale=:log10)

# Since random forests are inherently randomized, we generate multiple
# curves:

curves = learning_curve(
    mach,
    range=r,
    resampling=Holdout(),
    measure=log_loss,
    rngs=4,
    rng_name=:rng,
)

plt = plot(curves.parameter_values, curves.measurements)
xlabel!(plt, "n_trees")
ylabel!(plt, "cross entropy")
savefig("exercise_6ci.png")
plt #!md

# ![](exercise_6ci.png) #md


# 6(c)(ii)

evaluate!(mach, resampling=CV(nfolds=9),
                measure=log_loss,
                rows=train).measurement[1]

model.n_trees = 90

# 6(c)(iii)

err_forest =
    evaluate!(mach, resampling=Holdout(), measure=log_loss).measurement[1]

# #### Exercise 7

# 7(a)

KMeans = @load KMeans pkg=Clustering
EvoTreeClassifier = @load EvoTreeClassifier
pipe = Standardizer |>
    ContinuousEncoder |>
    KMeans(k=10) |>
    EvoTreeClassifier(nrounds=50)

# 7(b)

mach = machine(pipe, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=log_loss)

# 7(c)

r = range(pipe, :(evo_tree_classifier.max_depth), lower=1, upper=10)
curve = learning_curve(
    mach,
    range=r,
    resampling=CV(nfolds=6),
    measure=log_loss,
)
plt = plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "max_depth")
ylabel!(plt, "CV estimate of cross entropy")
savefig("exercise_7c.png")
plt #!md

# ![](exercise_7c.png) #md

# #### Exercise 8

# From the question statement:

y, X = unpack(house, ==(:price), rng=123); # from Exercise 3

EvoTreeRegressor = @load EvoTreeRegressor
tree_booster = EvoTreeRegressor(nrounds = 70)
model = ContinuousEncoder |> tree_booster

r2 = range(model, :(evo_tree_regressor.nbins), values = [32, 64, 128])

# (a)

r1 = range(model, :(evo_tree_regressor.max_depth), lower=4, upper=14)

# (b)

tuned_model = TunedModel(
    model;
    ranges=[r1, r2],
    resampling=Holdout(),
    measures=mae,
    tuning=RandomSearch(rng=123),
    n=40,
)

tuned_mach = machine(tuned_model, X, y) |> fit!
plt = plot(tuned_mach)
savefig("exercise_8c.png")
plt #!md

# ![](exercise_8c.png) #md

# (c)

best_model = report(tuned_mach).best_model;
best_mach = machine(best_model, X, y);
best_err = evaluate!(best_mach, resampling=CV(nfolds=3), measure=mae)

#-

tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=mae)
