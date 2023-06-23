import os
from colorama import Fore
from agents import Agent, ThoughtNode
from simulation import Simulation
from theory_of_mind import TheoryOfMind

def display_message(sender, receiver, message, method):
    print(Fore.CYAN + f"Agent {sender.id} ({sender.name}): {message} (method: {method})")
    
    if receiver is not None:
        print(Fore.GREEN + f"Agent {receiver.id} ({receiver.name}) received the message.")
    print(Fore.RESET)

def main():
    # Initialize TheoryOfMind
    tom = TheoryOfMind()

    # Access environment variables
    user_role = os.getenv('USER_ROLE', 'user1')
    behavior = os.getenv('BEHAVIOR', 'curious')

    # User inputs
    goal = input("Enter the goal for agents: ").strip()
    domain = input("Enter the domain related to the goal: ").strip()
    domain1 = input("Enter the first domain related to the goal: ").strip()
    domain2 = input("Enter the second domain related to the goal: ").strip()
    # Create agent instances
    agent1 = Agent(id=1, goal=goal, role='expert', domain=domain1)
    agent2 = Agent(id=2, goal=goal, role='expert', domain=domain2)

    agent1_experts = agent1.generate_experts(domain)
    agent2_experts = agent2.generate_experts(domain)

    name1 = agent1_experts[0] if agent1_experts else None
    name2 = agent2_experts[0] if agent2_experts else None  # Update this line

    agent1.name = name1
    agent2.name = name2
    
    task1 = agent1.generate_task(goal, details="detail1")
    task2 = agent2.generate_task(goal, details="detail2")

    # Print tasks derived from the goal for each agent
    print(Fore.BLUE + f"Agent 1 ({agent1.name}) derived task from the goal: {task1}")
    print(Fore.BLUE + f"Agent 2 ({agent2.name}) derived task from the goal: {task2}")
    print(Fore.RESET)

    # Register agents with TheoryOfMind
    tom.register_agent(agent1)
    tom.register_agent(agent2)

    # Add agents to the Simulation environment
    agents = [agent1, agent2]
    simulation = Simulation(agents)

    # Print introductory message
    print(Fore.YELLOW + f"System: A simulated interaction between agents with Theory of Mind is about to start.")
    
    # Execution of the interaction between agents can be implemented with a loop, events, or scenario-based triggers.
    while not simulation.finished:  # Implement a termination condition for the simulation.
        simulation.run_interactions(display_message)
        simulation.update_finished_status()  # Add this line to update the finished status
        # Optionally, update other variables in the simulation loop, e.g., scenario-based triggers.

if __name__ == "__main__":
    main()