# Stock Trading Agent Evolution

This Python project implements an evolutionary algorithm to develop stock trading agents. It uses historical stock data to train and evaluate agents, aiming to optimize trading strategies.

## Features

- Fetches historical stock data using yfinance
- Implements a genetic algorithm to evolve trading agents
- Trains agents on historical data
- Evaluates agent performance against buy-and-hold strategy
- Saves the best-performing agents for future use

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- yfinance

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stock-trading-agent-evolution.git
   ```
2. Install the required Python packages:
   ```
   pip install numpy matplotlib yfinance
   ```

## Usage

Run the script using Python:

```
python stock_trading_evolution.py
```

The script will:
1. Fetch historical data for specified stocks
2. Train a population of agents using the evolutionary algorithm
3. Evaluate the best agents' performance
4. Save the weights of the best agents for each stock

## How it Works

1. `Agent` class: Represents a trading agent with weights and prediction method
2. `get_stock_data`: Fetches historical stock data
3. `evaluate_agents`: Evaluates agent performance
4. `select_and_reproduce`: Implements genetic algorithm selection and reproduction
5. `train_agents`: Trains agents over multiple generations
6. `test_agents` and `simulate_agent_trading`: Test agent performance
7. `run_simulation`: Orchestrates the training and evaluation process for each stock

## Customization

You can adjust the following hyperparameters in the script:
- `STOCKS`: List of stock symbols to analyze
- `DAYS_OF_DATA`: Number of days of historical data to use
- `TRAIN_TEST_SPLIT`: Ratio of data used for training vs. testing
- `NUM_AGENTS`: Number of agents in the population
- `NUM_GENERATIONS`: Number of generations for training
- `INPUT_SIZE`: Number of days of data used for each prediction
- `MUTATION_RATE`: Rate of mutation in the genetic algorithm
- `NUM_BEST_AGENTS`: Number of top agents to save

## Output

The script generates:
- Console output showing training progress and performance metrics
- Pickle files containing the weights of the best agents for each stock

## Detailed Explanation of Agent Training Process
The training of agents in this stock trading evolution system uses a genetic algorithm approach. Here's a step-by-step breakdown of the process:

## Agent Initialization:

A population of NUM_AGENTS (200) agents is created.
Each agent is represented by the Agent class, which has a weight matrix of shape (INPUT_SIZE, 1).
The weights are initially randomly generated using np.random.randn().


## Training Loop:

The training process runs for NUM_GENERATIONS (2000) generations.
In each generation happens the below actions.


## Data Selection:

A random starting point (start_day) is selected from the training data.
This ensures that agents are exposed to different parts of the time series during training.


## Agent Evaluation:

The evaluate_agents function is called for all agents:

It takes a slice of the stock data starting from start_day with length INPUT_SIZE (14 days).
Each agent makes a prediction for the stock price on day INPUT_SIZE + 1.
The prediction is made by taking the dot product of the input data and the agent's weights.
The error for each agent is calculated as the absolute difference between the prediction and the actual price.

## Selection and Reproduction:

The select_and_reproduce function implements the genetic algorithm:

Agents are sorted based on their prediction errors.
The top NUM_AGENTS // 3 (about 66) agents with the lowest errors are selected as survivors.
A new population of agents is created through reproduction:

For each new agent, two parent agents are randomly selected from the survivors.
The child agent's weights are initialized as the average of its parents' weights.
Mutation is applied to the child's weights by adding small random values (controlled by MUTATION_RATE).

## Iteration and Progress:

This process of evaluation, selection, and reproduction is repeated for each generation.
Every 50 generations, the average error of the population is printed to show progress.

## Final Selection:

After all generations, the agents are evaluated one last time on a portion of the training data.
The top NUM_BEST_AGENTS (5) performing agents are selected.
The weights of these best agents are saved to a file for later use.

## Key Aspects of the Training Process:

Genetic Diversity: By starting with a random population and using mutation, the algorithm maintains genetic diversity, allowing it to explore a wide range of potential strategies.
Selective Pressure: By keeping only the top-performing agents in each generation, there's a strong selective pressure towards better performance.
Gradual Improvement: Over many generations, the population tends to improve its average performance, as successful traits are more likely to be passed on and refined.
Adaptation to Data: By training on different segments of the time series, agents learn to generalize across various market conditions present in the training data.

## Disclaimer

Project was done just to practice and was done in a quick way, so im assuming there will be plenty of room for improvement. Hopfully can be used as foundations for other projects.

######################################

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
