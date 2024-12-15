Pokebot Evolved version 0.1
By Willow Dennison - dennisonwillow@gmail.com

All functions can be called from src/pokebot_evolved.py

Discord bot integration to be added, this version only generates the Pokemon and their Pokedex entries.

Namegen module is based on a PyTorch model trained to generate text character by character. The TypeClassifier module uses a RandomForestClassifier to classify type based on the first 12 characters of the Pokemon's name. The Statgen module uses a MultiOutputRegressor using RandomForestRegressor estimators to predict the 6 base stats based on the first 12 characters of the name and the Pokemon's types.
For the category and descriptions, Pokebot integrates with Google's generative AI API.