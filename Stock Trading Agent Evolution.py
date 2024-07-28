import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pickle

# Hyperparameters
STOCKS = ["GOOGL","AAPL",  "MSFT", "AMZN", "TSLA"]
DAYS_OF_DATA = 2000
TRAIN_TEST_SPLIT = 0.8
NUM_AGENTS = 200
NUM_GENERATIONS = 2000
INPUT_SIZE = 14
MUTATION_RATE = 0.03
NUM_BEST_AGENTS = 5

class Agent:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
    
    def predict(self, inputs):
        if len(inputs) != self.weights.shape[0]:
            raise ValueError(f"Input size {len(inputs)} does not match weight shape {self.weights.shape}")
        return np.dot(inputs, self.weights)[0]
    
    def mutate(self, rate=MUTATION_RATE):
        self.weights += np.random.randn(*self.weights.shape) * rate

def get_stock_data(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days*2)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df['Close'].values[-days:]

def evaluate_agents(agents, stock_data, start_day):
    inputs = stock_data[start_day:start_day+INPUT_SIZE]
    target = stock_data[start_day+INPUT_SIZE]
    predictions = np.array([agent.predict(inputs) for agent in agents])
    errors = np.abs(predictions - target)
    return errors, predictions

def select_and_reproduce(agents, errors, num_survivors):
    survivor_indices = np.argsort(errors)[:num_survivors]
    survivors = [agents[i] for i in survivor_indices]
    new_agents = []
    for _ in range(len(agents)):
        parent1, parent2 = np.random.choice(survivors, 2, replace=False)
        child = Agent(INPUT_SIZE, 1)
        child.weights = (parent1.weights + parent2.weights) / 2
        child.mutate()
        new_agents.append(child)
    return new_agents

def train_agents(train_data):
    total_days = len(train_data)
    agents = [Agent(INPUT_SIZE, 1) for _ in range(NUM_AGENTS)]

    for generation in range(NUM_GENERATIONS):
        start_day = np.random.randint(0, total_days - INPUT_SIZE - 1)
        errors, _ = evaluate_agents(agents, train_data, start_day)
        agents = select_and_reproduce(agents, errors, NUM_AGENTS // 3)
        
        if generation % 50 == 0:
            print(f"Generation {generation}: Avg Error = {np.mean(errors):.4f}")

    return agents

def test_agents(agents, test_data):
    predictions = []
    for i in range(len(test_data) - INPUT_SIZE):
        inputs = test_data[i:i+INPUT_SIZE]
        prediction = np.mean([agent.predict(inputs) for agent in agents])
        predictions.append(prediction)
    return np.array(predictions)

def calculate_returns(prices):
    return (prices[-1] - prices[0]) / prices[0]

def simulate_buy_hold(prices, initial_cash=100):
    num_shares = initial_cash / prices[0]
    final_value = num_shares * prices[-1]
    return (final_value - initial_cash) / initial_cash

def simulate_agent_trading(agent, prices, initial_cash=100):
    cash = initial_cash
    shares = 0
    for i in range(INPUT_SIZE, len(prices) - 1):
        inputs = prices[i-INPUT_SIZE:i]
        prediction = agent.predict(inputs)
        current_price = prices[i]
        
        if prediction > current_price and cash > 0:  # Buy
            shares_to_buy = cash / current_price
            shares += shares_to_buy
            cash = 0
            print(f"Day {i}: Bought at {current_price:.2f}, prediction {prediction:.2f}, shares {shares:.2f}, cash {cash:.2f}")
        elif prediction < current_price and shares > 0:  # Sell
            cash += shares * current_price
            shares = 0
            print(f"Day {i}: Sold at {current_price:.2f}, prediction {prediction:.2f}, shares {shares:.2f}, cash {cash:.2f}")
    
    # Sell any remaining shares at the end
    if shares > 0:
        final_value = cash + shares * prices[-1]
        print(f"Final cash: {cash:.2f}, final shares: {shares:.2f}, final price: {prices[-1]:.2f}, final value: {final_value:.2f}")
    else:
        final_value = cash
    return (final_value - initial_cash) / initial_cash

def run_simulation(stock_data, stock_name):
    split = int(TRAIN_TEST_SPLIT * len(stock_data))
    train_data = stock_data[:split]
    test_data = stock_data[split:]

    print(f"Training agents on {stock_name}...")
    trained_agents = train_agents(train_data)

    print(f"Saving best {NUM_BEST_AGENTS} agents...")
    errors, _ = evaluate_agents(trained_agents, train_data[-INPUT_SIZE-1:], 0)
    best_agents = [trained_agents[i] for i in np.argsort(errors)[:NUM_BEST_AGENTS]]
    
    # Save best agents' weights
    best_agents_weights = [agent.weights for agent in best_agents]
    with open(f'{stock_name}_best_agents.pkl', 'wb') as f:
        pickle.dump(best_agents_weights, f)

    print(f"Best agents' weights for {stock_name} saved.")

# Run simulation for each stock
for stock_symbol in STOCKS:
    print(f"\nFetching data for {stock_symbol}...")
    stock_data = get_stock_data(stock_symbol, DAYS_OF_DATA)
    print(f"Data shape for {stock_symbol}: {stock_data.shape}")
    run_simulation(stock_data, stock_symbol)

print("All simulations complete and best agents saved.")
