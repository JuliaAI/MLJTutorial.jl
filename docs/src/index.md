# [MLJTutorial.jl](@id home)

Tutorials for introducing the machine learning toolbox [MLJ](https://juliaml.ai) (Machine
Learning in Julia)

Two collections of tutorials are hosted here:

- [*Using MLJ* (video series)](@ref video) created 2025.
- [*MLJTutorial* (original series)](@ref original) created around 2020, updated August 2026.

For other MLJ learning resources see
[here](https://juliaai.github.io/MLJ.jl/dev/learning_mlj/). For experienced data
scientists transitioning from another Machine Learning platform, we recommend [MLJ for
Data Scientists in Two
Hours](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/telco/).

Also hosted here is a [Lightning Tour of MLJ](@ref).

## [*Using MLJ* (video series)](@id video)

Tutorials for use with the *Using MLJ* series of instructional videos. 

- [Lesson 1. Basics](@ref)
- [Lesson 2. Model Composition](@ref)
- [Lesson 3. Model Tuning](@ref)


## [*MLJTutorial* (original series)](@id original)

Tutorials based on a 3.5 hour [online
  workshop](https://github.com/ablaom/MachineLearningInJulia2020) delivered in 2020,
  updated August 2026. These include a more detailed narrative and discuss some advanced
  features.


### Basic

- [Tutorial 1. Data Representation](@ref)
- [Tutorial 2. Selecting, Training and Evaluating Models](@ref)
- [Tutorial 3. Transformers and Pipelines](@ref)

### Advanced

- [Tutorial 4. Tuning hyperparameters](@ref)
- [Tutorial 5. Advanced Model Composition](@ref)
- [Solutions to Exercises](@ref)
- [Lightning Tour of MLJ](@ref)

## Prerequisites

- Familiarity with basic data manipulation in Julia: vectors, tuples, dictionaries,
  arrays, generating random numbers, tabular data (e.g., DataDrames.jl) basic linear
  algebra and statistics.
  
- Familiarity with [Julia package management](https://docs.julialang.org/en/v1/stdlib/Pkg/)

- Familiarity with Machine Learning fundamentals and best practice.

## [Running tutorial code for yourself](@id instructions)

You can find the annotated Julia scripts from which tutorials are generated in [these
directories](https://github.com/JuliaAI/MLJTutorial.jl/tree/dev/docs/src/notebooks). Package
environment files for this repository are structured using Julia's package workspaces. To
use with Julia 1.12 or later:

1. Clone this GitHub repository to your computer

2. In a terminal window change your directory to the root of the clone.

3. From that terminal, launch julia (preferably with the `--startup-file=no` switch) and
   run `using Pkg; Pkg.activate("."); Pkg.instantiate(workspace=true)`. This resolves a
   valid set of package for all the tutorials, downloads package code to your computer,
   and carries out some precompilation. This may take a few minutes. You need carry out
   this step only once.

3. Activate the project for the particular tutorial of interest (as specified by
   Project.toml file in the corresponding directory).
   
4. Execute code blocks copied from the rendered tutorial (as linked above) or from the
   corresponding .jl script.


## More about the tutorials 

- The tutorials focus on the *machine learning* part of the data science workflow, and
  less on exploratory data analysis and other conventional "data analytics" methodology

- Here "machine learning" is meant in a broad sense, and is not restricted to so-called
  *deep learning* (neural networks)

- The tutorials are crafted to rapidly familiarize the user with what MLJ can do and how
  to do it, and are not a substitute for a course on machine learning
  fundamentals. Examples do not necessarily represent best practice or the best solution
  to a problem.

--

This site makes use of an experimental, unregistered package
[NotebookManagementTools.jl](@ref).
