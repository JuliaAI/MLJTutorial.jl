# # Tutorial 1. Data Representation

# > **Goals:**
# > 1. Learn how MLJ specifies it's data requirements using "scientific" types
# > 2. Understand the options for representing tabular data
# > 3. Learn how to inspect and fix the representation of data to meet MLJ requirements

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

# ### Scientific types

# To help you focus on the intended *purpose* or *interpretation* of data, MLJ models
# specify data requirements using *scientific types*, instead of machine types. An example
# of a scientific type is `OrderedFactor`. The other basic "scalar" scientific types are
# illustrated below:

# ![](scitypes.svg)

# A scientific type is an ordinary Julia type (so it can be used for method dispatch, for
# example) but it usually has no instances. The `scitype` function is used to articulate
# MLJ's convention about how different machine types will be interpreted by MLJ models:

using ScientificTypes
scitype(3.14)

#-

time = [2.3, 4.5, 4.2, 1.8, 7.1]
scitype(time)

# To fix data which MLJ is interpreting incorrectly, we use the
# `coerce` method:

height = [185, 153, 163, 114, 180]
scitype(height)

#-

height = coerce(height, Continuous)

# Here's an example of data we would want interpreted as
# `OrderedFactor` but isn't:

exam_mark = ["rotten", "great", "bla",  missing, "great"]
scitype(exam_mark)

#-

exam_mark = coerce(exam_mark, OrderedFactor)

#-

levels(exam_mark)

# Use `levels!` to put the classes in the right order:

levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]

# When sub-sampling, no levels are lost:

levels(exam_mark[1:2])

# **Note on binary data.** There is no separate scientific type for binary data. Binary
# data is `OrderedFactor{2}` or `Multiclass{2}`. If a binary measure like `truepositive`
# is a applied to `OrderedFactor{2}` then the "positive" class is assumed to appear
# *second* in the ordering. If such a measure is applied to `Multiclass{2}` data, a
# warning is issued. A single `OrderedFactor` can be coerced to a single `Continuous`
# variable, for models that require this, while a `Multiclass` variable can only be
# one-hot encoded.

# See also [Working with Categorical
# Data](https://juliaai.github.io/MLJ.jl/stable/working_with_categorical_data/#Working-with-Categorical-Data)
# from the MLJ manual.


# ### Two-dimensional data

# Whenever it makes sense, MLJ Models generally expect two-dimensional data to be
# *tabular*. Most tabular formats implementing the [Tables.jl
# API](https://juliadata.github.io/Tables.jl/stable/) (see this
# [list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md)) have a
# scientific type of `Table` and can be used with such models.

# Probably the simplest example of a table is the julia native *column
# table*, which is just a named tuple of equal-length vectors:

column_table = (h=height, e=exam_mark, t=time)

# While a table has a `scitype`, the general user will want to inspect column scitypes
# using MLJ's `schema` method:

schema(column_table)

# Here are other examples of tables:

dict_table = Dict(:h => height, :e => exam_mark, :t => time)
schema(dict_table)

# (To control column order here, instead use `LittleDict` from
# OrderedCollections.jl.)

row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]
schema(row_table)

#-

import DataFrames
df = DataFrames.DataFrame(column_table)

#-

schema(df)

#-

# A schema is itself a table. If we convert it to a dataframe, we can get a nicer display
# in some contexts (e.g., in documentation or a jupyter notebook):

schema(df) |> DataFrames.DataFrame

# Most MLJ models do not accept a matrix in lieu of a table, but you can
# wrap a matrix as a table:

using Tables
matrix_table = Tables.table(rand(2,3))
schema(matrix_table)

# ### Fixing scientific types in tabular data

# To show how we can correct the scientific types of data in tables, let's look more
# closely at a cleaned up version of the UCI Horse Colic Data set. (The cleaning work-flow
# is described
# [here](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values).)

import Downloads
import CSV
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/"*
    "for-MLJ-version-0.16/data/horse.csv"
csv_file = Downloads.download(url)


# Entering these lines of code downloads the data to a temporary file at the location
# shown above. We'll read this data into memory as a dataframe, provided by the
# DataFrames.jl package; see [this
# tutorial](https://juliaai.github.io/DataScienceTutorials.jl/data/dataframe/) for a
# quick-start introduction.

horse = CSV.read(csv_file, DataFrames.DataFrame)
first(horse, 4)

# From [the UCI
# docs](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) we can
# surmise how each variable ought to be interpreted (a step in our
# work-flow that cannot reliably be left to the computer):

# variable                    | scientific type (interpretation)
# ----------------------------|-----------------------------------
# `:surgery`                  | Multiclass
# `:age`                      | Multiclass
# `:rectal_temperature`       | Continuous
# `:pulse`                    | Continuous
# `:respiratory_rate`         | Continuous
# `:temperature_extremities`  | OrderedFactor
# `:mucous_membranes`         | Multiclass
# `:capillary_refill_time`    | Multiclass
# `:pain`                     | OrderedFactor
# `:peristalsis`              | OrderedFactor
# `:abdominal_distension`     | OrderedFactor
# `:packed_cell_volume`       | Continuous
# `:total_protein`            | Continuous
# `:outcome`                  | Multiclass
# `:surgical_lesion`          | OrderedFactor
# `:cp_data`                  | Multiclass

# Let's see how MLJ will actually interpret the data, as it is
# currently encoded:

schema(horse)

# As a first correction step, we can get MLJ to "guess" the
# appropriate fix, using the `autotype` method:

autotype(horse)

#-

# Okay, this is not perfect, but a step in the right direction, which
# we implement like this:

coerce!(horse, autotype(horse));
schema(horse)

# All remaining `Count` data should be `Continuous`:

coerce!(horse, Count => Continuous);
schema(horse)

# We'll correct the remaining truant entries manually:

coerce!(horse,
        :surgery               => Multiclass,
        :age                   => Multiclass,
        :mucous_membranes      => Multiclass,
        :capillary_refill_time => Multiclass,
        :outcome               => Multiclass,
        :cp_data               => Multiclass);
schema(horse)


# ### Resources for this tutorial
#
# - From the MLJ manual:
#   - [A preview of data type specification in
#     MLJ](https://juliaai.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1)
#   - [Data containers and scientific
#     types](https://juliaai.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1)
#   - [Working with Categorical Data](https://juliaai.github.io/MLJ.jl/dev/working_with_categorical_data/)
# - [Summary](https://juliaai.github.io/ScientificTypes.jl/dev/#Summary-of-the-default-convention) of the MLJ convention for representing scientific types
# - [ScientificTypes.jl](https://juliaai.github.io/ScientificTypes.jl/dev/)
# - From Data Science Tutorials:
#   - [Data interpretation: Scientific Types](https://juliaai.github.io/DataScienceTutorials.jl/data/scitype/)
#   - [Horse colic data](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/) [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)
# - [Documentation](https://juliaai.github.io/StatisticalMeasures.jl/stable/) on
#   performance metrics.


# ### Tutorial 1 Exercises

# #### Exercise 1

# Try to guess how each code snippet below will evaluate:

scitype(42);

#-

questions = ["who", "why", "what", "when"]
scitype(questions);

#-

elscitype(questions);

#-

t = (3.141, 42, "how")
scitype(t);

#-

A = rand(2, 3)

#-

scitype(A);

#-

elscitype(A);

#-

using SparseArrays
Asparse = sparse(A)

#-

scitype(Asparse);

#-

C = coerce(A, Multiclass);

#-

scitype(C);

#-

elscitype(C);

#-

v = [1, 2, missing, 4]
scitype(v);

#-

elscitype(v);

#-

scitype(v[1:2]);

# Can you guess at the general behavior of
# `scitype` with respect to tuples, abstract arrays and missing
# values? The answers are
# [here](https://github.com/juliaai/ScientificTypesBase.jl#2-the-scitype-and-scitype-methods)
# (ignore "Property 1").


# #### Exercise 2

# Coerce the following vector to make MLJ recognize it as a vector of
# ordered factors (with an appropriate ordering):

quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"];


# #### Exercise 3 (fixing scitypes in a table)

# Fix the scitypes for the [House Prices in King
# County](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/)
# dataset:

url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
    "data/house.csv";
house = CSV.read(Downloads.download(url), DataFrames.DataFrame)
first(house, 4)

# (Two features in the original data set have been deemed uninformative
# and dropped, namely `:id` and `:date`. The original feature
# `:yr_renovated` has been replaced by the `Bool` feature `is_renovated`.)
