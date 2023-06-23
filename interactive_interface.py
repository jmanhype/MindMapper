import tkinter as tk
from tkinter import simpledialog, messagebox
from agents import Agent, Teacher, Student, Expert, Mediator, RoleBasedBot
from simulation import Simulation
from colorama import Fore, init

# Initialize the colorama library
init()

class Interactive_Interface:

    def __init__(self, agents):
        self.agents = agents
        self.simulation = Simulation(agents)
        self.root = tk.Tk()
        self.root.title("Multi-Agent Role-Playing Framework")

    def display_agents(self):
        self.agent_list = tk.Listbox(self.root)
        self.update_agent_list()
        self.agent_list.pack()

    def update_agent_list(self):
        self.agent_list.delete(0, tk.END)
        for agent in self.agents:
            self.agent_list.insert(tk.END, f"{agent.__class__.__name__}: {agent.role}")

    def add_agent(self):
        pass  # Implement the method to add an agent to the list of agents

    def remove_agent(self):
        pass  # Implement the method to remove an agent from the list of agents

    def start_simulation(self):
        self.simulation.run_interactions(self.display_message)

    def stop_simulation(self):
        self.simulation = Simulation(self.agents)

    def display_message(self, sender, message):
        if sender.role == 'RoleBasedBot':
            print(Fore.BLUE + f"{sender.role}: {message}")
        else:
            print(Fore.GREEN + f"{sender.role}: {message}")

    def run(self):
        self.display_agents()

        btn_add_agent = tk.Button(self.root, text="Add Agent", command=self.add_agent)
        btn_add_agent.pack()

        btn_remove_agent = tk.Button(self.root, text="Remove Agent", command=self.remove_agent)
        btn_remove_agent.pack()

        btn_start_sim = tk.Button(self.root, text="Start Simulation", command=self.start_simulation)
        btn_start_sim.pack()

        btn_stop_sim = tk.Button(self.root, text="Stop Simulation", command=self.stop_simulation)
        btn_stop_sim.pack()

        self.root.mainloop()

if __name__ == "__main__":
    teacher = Teacher(subject='science')
    student = Student(grade_level='8th')
    expert = Expert(domain='biology')
    mediator = Mediator()
    role_based_bot = RoleBasedBot(user_role='user1', behavior='curious')

    agents = [teacher, student, expert, mediator, role_based_bot]
    interface = Interactive_Interface(agents)
    interface.run()