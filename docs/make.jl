using Documenter

const REPO_NAME = "MLJTutorial.jl"
const  REPO = Remotes.GitHub("JuliaAI", REPO_NAME)

using NotebookManagementTools

makedocs(
    modules=Module[NotebookManagementTools,],
    format = Documenter.HTML(
        prettyurls=true,
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(
                "https://fonts.googleapis.com/css2?family="*
                    "Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;"*
                    "1,100;1,300;1,400;1,700;1,900&"*
                    "family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap",
                class = :css,
            ),
            asset(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/"*
                    "5.15.4/css/all.min.css",
                class = :css,
            ),
        ],
        size_threshold = 10485760,
        repolink = "https://github.com/JuliaAI/MLJTutorial.jl",
    ),
    pages=[
        "Home" =>                   "index.md",
        "Using MLJ (video series)" => [
            "1\\. Basics"  =>
                "notebooks/UsingMLJ/01_basics/notebook.md",
            "2\\. Model Composition" =>
                "notebooks/UsingMLJ/02_model_composition/notebook.md",
            "3\\. Model Tuning" =>
                "notebooks/UsingMLJ/03_model_tuning/notebook.md",
        ],
        "MLJTutorial (original series)" => [
            "1\\. Data Representation" =>
                "notebooks/MLJTutorial/01_data_representation/notebook.md",
            "2\\. Models" =>
                "notebooks/MLJTutorial/02_models/notebook.md",
            "3\\. Pipelines" =>
                "notebooks/MLJTutorial/03_pipelines/notebook.md",
            "4\\. Tuning" =>
                "notebooks/MLJTutorial/04_tuning/notebook.md",
            "5\\. Composition" =>
                "notebooks/MLJTutorial/05_composition/notebook.md",
            "Solutions to Exercises" =>
                "notebooks/MLJTutorial/99_solution_to_exercises/notebook.md",
        ],
        "Lightning Tour of MLJ" => "notebooks/lightning_tour/notebook.md",
    ],
    sitename=REPO_NAME,
    warnonly = [:cross_references, :missing_docs],
    repo = REPO,
)

deploydocs(
    devbranch="dev", # deployment to gh-pages only happens when this is the target
    push_preview=true,
    repo="github.com/JuliaAI/$REPO_NAME.git",
)
