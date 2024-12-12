smores/
├── data/ # Placeholder for datasets
│ ├── raw/ # Raw datasets (unprocessed)
│ ├── processed/ # Processed datasets
│ └── README.md # Details on data sources and formats
├── docs/ # Documentation files
│ └── simulation_design.md # Design and methodology of the simulation
├── experiments/ # Configuration files and scripts for experiments
│ ├── config/ # Experiment configurations (e.g., JSON, YAML)
│ ├── results/ # Results from experiments
| └── exp_name/ # Expiremenet name
| └── 2024_11_09_10_19_10/ # Expirement directory name based on run time
| ├── params.json # json file to store the expirement parameters
├── file_1.csv # the csv files associated with the experiment
└── file_2.csv # the csv files associated with the experiment

│ └── README.md # Explanation of experiments
├── img/ # saved images
│ └── exp1_visual.png # Storing visualization images
├── smores/ # Source code
│ ├── algorithms/ # Recommender algorithms and related utilities
│ ├── simulation/ # Simulation logic
│ ├── evaluation/ # Metrics and evaluation scripts
│ └── utils/ # mainstream utilities
├── notebooks/ # Exploration notebooks
│ └── EDA.ipynb # Recommender algorithms and related utilities
├── tests/ # Unit and integration tests
│ ├── test_simulation.py # Test cases for simulation
│ ├── test_utils.py # Test cases for utilities
│ └── ... # Other test files
├── requirements.txt # Python dependencies
├── setup.py # Installable package setup (optional)
├── README.md # Project overview
├── LICENSE # License for the repository
└── .gitignore # Files and directories to ignore in Git

mkdir -p {data/{raw,processed},docs,experiments/{config,results},smores/{algorithms,simulation,evaluation,utils},tests,img,notebooks}

## Next
1- niche provider is making WAY more money? why
2- to discuss:
    I'm hitting a roadblock using the imdb dataset without modifying the genre preferences, I'll try with the bigger movielens dataset. for the small dataset, if you recommend only action, comedy, and drama you'll guarantee that everyone will be happy
    another thing that I'm thinking about is if we should predefine what the niche genre is for our next experiment or should we just leave the simulation define what is niche

## genre preferences
for genre in user_genre_preference.columns:
    print(genre, len(user_genre_preference[user_genre_preference[genre] > 0.01]))

(no genres listed) 1
Action 606
Adventure 604
Animation 444
Children 503
Comedy 609
Crime 595
Documentary 62
Drama 609
Fantasy 567
Film-Noir 51
Horror 442
IMAX 310
Musical 307
Mystery 541
Romance 601
Sci-Fi 597
Thriller 606
War 472
Western 181