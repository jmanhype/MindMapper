# MindMapper

MindMapper is a Python project that explores the concept of Theory of Mind through thought trees and agent interactions. It simulates agents with different goals interacting and learning from their experiences.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The MindMapper project focuses on implementing the Theory of Mind concept, which involves agents having the ability to understand and infer the mental states of other agents. It uses thought trees to represent an agent's thoughts and experiences, allowing them to explore and learn from their interactions with other agents.

The project consists of the following components:

- `Simulation`: The main simulation class that orchestrates agent interactions and tracks their progress.
- `Agent`: Represents an agent with a unique ID, a goal, and knowledge about their experiences and thought tree.
- `Thought_Tree_Explorer`: Provides different strategies for exploring thought trees, including depth-first search, breadth-first search, and heuristic search.
- `Experience` and `ExperienceType`: Classes that define an agent's experiences and their types (positive, neutral, negative).
- `Thought`: Represents a thought in the thought tree, connecting parent thoughts and child thoughts.

## Installation

1. Clone the MindMapper repository:

   ```bash
   git clone https://github.com/jmanhype/MindMapper.git
   ```

2. Navigate to the project directory:

   ```bash
   cd MindMapper
   ```

3. Update the API keys in the agents.py file.
   Replace the Anthropic API key placeholder with your actual Anthropic API key. Setup instructions inspired by the Anthropic API documentation can be found at https://docs.anthropic.com.
   Refer to the instructions in agents.py for details.
   
4. (Optional) Create a virtual environment:
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix/Linux
   venv\Scripts\activate  # For Windows
   ```

5. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   
## API Keys

The MindMapper project requires API keys for certain functionalities. You need to obtain the following API keys and update them in the agents.py file:

Google API Key: Obtain a Google API key from the Google Cloud Console. It is used for agent interactions.

Anthropic API Key: Obtain an Anthropic API key from the Anthropic website. It is used for generating context-aware messages.

Make sure to keep your API keys secure and avoid sharing them publicly.

## Usage

To run the MindMapper simulation, follow these steps:

1. Open the terminal and navigate to the project directory.

2. Run the main script:

   ```bash
   python main.py
   ```

   The simulation will start, and you will see the agent interactions and their progress displayed in the console.

3. Modify the simulation parameters in the `main.py` file according to your needs, such as the maximum number of iterations and the agents' goals.

## Contributing

Contributions to MindMapper are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/jmanhype/MindMapper).

When contributing, please adhere to the existing code style and conventions. Also, make sure to update the documentation and tests as necessary.

## License

MindMapper is licensed under the [Apache License 2.0](LICENSE).
